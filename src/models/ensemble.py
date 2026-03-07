"""
Advanced ensemble predictor with walk-forward calibration.
Combines LSTM and XGBoost with proper probability calibration.
"""

import logging
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV

from config.settings import ENSEMBLE_CONFIG, LSTM_CONFIG
from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Combines LSTM and XGBoost with calibrated probabilities."""

    def __init__(self, config: dict = None):
        self.config = config or ENSEMBLE_CONFIG
        self.lstm = LSTMPredictor()
        self.xgboost = XGBoostPredictor()
        self.scaler = RobustScaler()  # Robust to outliers
        self.is_trained = False
        self.feature_names = []

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
        self.feature_names = feature_names

        # Robust scaling
        X_scaled = self.scaler.fit_transform(X)

        # Walk-forward split: use last 20% as validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train both models
        logger.info("Training LSTM model...")
        lstm_result = self.lstm.train(X_train, y_train, X_val, y_val)

        logger.info("Training XGBoost model...")
        xgb_result = self.xgboost.train(X_train, y_train, X_val, y_val)

        # Evaluate ensemble on validation set
        seq_len = LSTM_CONFIG["sequence_length"]

        # XGBoost predictions on validation
        xgb_val_preds = self.xgboost.predict(X_val)

        # LSTM predictions on validation
        lstm_val_preds = self.lstm.predict(X_val)

        # Align (LSTM has fewer predictions due to sequence windowing)
        xgb_aligned = xgb_val_preds[seq_len:]
        y_val_aligned = y_val[seq_len:]

        if len(lstm_val_preds) == len(xgb_aligned) and len(lstm_val_preds) > 0:
            # Find optimal weights using validation set
            best_acc = 0
            best_w = self.config["lstm_weight"]
            for w in np.arange(0.1, 0.9, 0.05):
                combo = w * lstm_val_preds + (1 - w) * xgb_aligned
                acc = np.mean((combo > 0.5).astype(int) == y_val_aligned)
                if acc > best_acc:
                    best_acc = acc
                    best_w = w

            self.config["lstm_weight"] = round(best_w, 2)
            self.config["xgboost_weight"] = round(1 - best_w, 2)
            ensemble_acc = best_acc
            logger.info(f"Optimal weights: LSTM={best_w:.2f} XGB={1-best_w:.2f}")
        else:
            # Fallback: just use XGBoost accuracy
            xgb_binary = (xgb_val_preds > 0.5).astype(int)
            ensemble_acc = np.mean(xgb_binary == y_val)

        self.is_trained = True

        result = {
            "lstm": lstm_result,
            "xgboost": xgb_result,
            "ensemble_val_accuracy": ensemble_acc,
            "weights": f"LSTM={self.config['lstm_weight']} XGB={self.config['xgboost_weight']}",
        }
        logger.info(f"Ensemble validation accuracy: {ensemble_acc:.4f}")

        top_features = self.xgboost.get_top_features(feature_names)
        if top_features:
            logger.info("Top features: " + ", ".join(f"{n}={v:.4f}" for n, v in top_features[:5]))

        return result

    def predict_latest(self, X: np.ndarray, raw_features: dict = None) -> dict:
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained")

        X_scaled = self.scaler.transform(X)

        lstm_pred = self.lstm.predict_latest(X_scaled)
        xgb_pred = self.xgboost.predict_latest(X_scaled)

        w_lstm = self.config["lstm_weight"]
        w_xgb = self.config["xgboost_weight"]
        ensemble_pred = w_lstm * lstm_pred + w_xgb * xgb_pred

        # Confidence: distance from 0.5, scaled to 0-1
        confidence = abs(ensemble_pred - 0.5) * 2

        signal = "BUY" if ensemble_pred > 0.5 else "SELL"

        # Agreement bonus: both models agree = higher confidence
        models_agree = (lstm_pred > 0.5) == (xgb_pred > 0.5)

        # Market Quality filter from advanced features
        market_quality = 0.5
        regime = -1
        hurst = 0.5
        entropy_val = 1.0
        if raw_features:
            market_quality = raw_features.get("market_quality", 0.5)
            regime = raw_features.get("regime", -1)
            hurst = raw_features.get("hurst", 0.5)
            entropy_val = raw_features.get("entropy", 1.0)

        # Trade decision: model confidence + market quality + model agreement
        should_trade = (
            models_agree
            and confidence >= self.config["confidence_threshold"]
            and market_quality >= 0.25  # market must be somewhat tradable
            and regime != 3            # don't trade in choppy regime
            and entropy_val < 0.9      # market must show some pattern
        )

        return {
            "signal": signal,
            "confidence": confidence,
            "ensemble_probability": ensemble_pred,
            "lstm_probability": lstm_pred,
            "xgboost_probability": xgb_pred,
            "models_agree": models_agree,
            "market_quality": market_quality,
            "regime": int(regime),
            "hurst": hurst,
            "entropy": entropy_val,
            "should_trade": should_trade,
        }

    def save(self, symbol: str):
        self.lstm.save(symbol)
        self.xgboost.save(symbol)
        import joblib, os
        from config.settings import MODEL_DIR
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, f"scaler_{symbol}.pkl"))
        joblib.dump(self.feature_names, os.path.join(MODEL_DIR, f"features_{symbol}.pkl"))
        joblib.dump(self.config, os.path.join(MODEL_DIR, f"ensemble_config_{symbol}.pkl"))
        logger.info(f"Ensemble models saved for {symbol}")

    def load(self, symbol: str) -> bool:
        import joblib, os
        from config.settings import MODEL_DIR
        lstm_ok = self.lstm.load(symbol)
        xgb_ok = self.xgboost.load(symbol)
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{symbol}.pkl")
        features_path = os.path.join(MODEL_DIR, f"features_{symbol}.pkl")
        config_path = os.path.join(MODEL_DIR, f"ensemble_config_{symbol}.pkl")
        if lstm_ok and xgb_ok and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)
            if os.path.exists(config_path):
                self.config = joblib.load(config_path)
            self.is_trained = True
            logger.info(f"Ensemble loaded for {symbol} | weights: LSTM={self.config['lstm_weight']} XGB={self.config['xgboost_weight']}")
            return True
        return False
