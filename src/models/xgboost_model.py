"""
XGBoost model for forex price direction prediction.
"""

import os
import numpy as np
import logging
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from config.settings import XGBOOST_CONFIG, MODEL_DIR

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """XGBoost-based price direction predictor."""

    def __init__(self, config: dict = None):
        self.config = config or XGBOOST_CONFIG
        self.model = XGBClassifier(**self.config)
        self.is_trained = False
        self.feature_importance = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        self.is_trained = True
        self.feature_importance = self.model.feature_importances_

        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        result = {"train_accuracy": train_acc}

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            result["val_accuracy"] = val_acc
            logger.info(f"XGBoost training | train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        else:
            logger.info(f"XGBoost training | train_acc={train_acc:.4f}")

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]

    def predict_latest(self, X: np.ndarray) -> float:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        proba = self.model.predict_proba(X[-1:])
        return float(proba[0][1])

    def get_top_features(self, feature_names: list, top_n: int = 10) -> list:
        if self.feature_importance is None:
            return []
        indices = np.argsort(self.feature_importance)[::-1][:top_n]
        return [(feature_names[i], self.feature_importance[i]) for i in indices]

    def save(self, symbol: str):
        path = os.path.join(MODEL_DIR, f"xgboost_{symbol}.pkl")
        joblib.dump(self.model, path)
        logger.info(f"XGBoost model saved to {path}")

    def load(self, symbol: str) -> bool:
        path = os.path.join(MODEL_DIR, f"xgboost_{symbol}.pkl")
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True
            logger.info(f"XGBoost model loaded from {path}")
            return True
        return False
