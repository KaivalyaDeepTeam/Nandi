"""
LSTM model for forex price direction prediction.
"""

import os
import numpy as np
import logging

from config.settings import LSTM_CONFIG, MODEL_DIR

logger = logging.getLogger(__name__)

# Suppress TF warnings and set seeds for reproducibility
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


class LSTMPredictor:
    """LSTM-based price direction predictor."""

    def __init__(self, config: dict = None):
        self.config = config or LSTM_CONFIG
        self.model = None
        self.is_trained = False

    def _build_model(self, input_shape: tuple):
        model = Sequential([
            LSTM(self.config["units_1"], return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(self.config["dropout"]),

            LSTM(self.config["units_2"], return_sequences=False),
            BatchNormalization(),
            Dropout(self.config["dropout"]),

            Dense(32, activation="relu"),
            Dropout(self.config["dropout"] / 2),

            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model
        return model

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple:
        seq_len = self.config["sequence_length"]
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i - seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        X_seq, y_seq = self._create_sequences(X_train, y_train)

        if X_seq.shape[0] == 0:
            raise ValueError("Not enough data for the given sequence length")

        self._build_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            if X_val_seq.shape[0] > 0:
                validation_data = (X_val_seq, y_val_seq)

        if validation_data is None:
            validation_split = 0.2
        else:
            validation_split = 0.0

        history = self.model.fit(
            X_seq, y_seq,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=1,
        )

        self.is_trained = True
        final_loss = history.history["loss"][-1]
        final_acc = history.history["accuracy"][-1]
        logger.info(f"LSTM training complete | loss={final_loss:.4f} acc={final_acc:.4f}")

        return {
            "loss": final_loss,
            "accuracy": final_acc,
            "epochs_trained": len(history.history["loss"]),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        X_seq, _ = self._create_sequences(X, np.zeros(len(X)))
        if X_seq.shape[0] == 0:
            raise ValueError("Not enough data for prediction")
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()

    def predict_latest(self, X: np.ndarray) -> float:
        """Predict only the latest data point."""
        seq_len = self.config["sequence_length"]
        if len(X) < seq_len:
            raise ValueError(f"Need at least {seq_len} data points")
        X_latest = X[-seq_len:].reshape(1, seq_len, X.shape[1])
        pred = self.model.predict(X_latest, verbose=0)
        return float(pred[0][0])

    def save(self, symbol: str):
        path = os.path.join(MODEL_DIR, f"lstm_{symbol}.keras")
        self.model.save(path)
        logger.info(f"LSTM model saved to {path}")

    def load(self, symbol: str) -> bool:
        path = os.path.join(MODEL_DIR, f"lstm_{symbol}.keras")
        if os.path.exists(path):
            self.model = load_model(path)
            self.is_trained = True
            logger.info(f"LSTM model loaded from {path}")
            return True
        return False
