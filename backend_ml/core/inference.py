import joblib
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any
from backend_ml.core.architecture import LSTMAutoencoder
from backend_ml.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    def __init__(
        self,
        scaler_path: str,
        isolation_forest_path: str,
        lstm_weights_path: str,
        input_dim: int = 6,
        hidden_dim: int = 32,
        device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing InferenceEngine on device: {self.device} (input_dim={self.input_dim}, hidden_dim={self.hidden_dim})")

        try:
            self.scaler = joblib.load(scaler_path)
            self.isolation_forest = joblib.load(isolation_forest_path)
            logger.info("Loaded StandardScaler and IsolationForest artifacts.")

            self.lstm_model = LSTMAutoencoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                seq_len=30
            )

            state_dict = torch.load(
                lstm_weights_path,
                map_location=self.device,
                weights_only=True
            )
            self.lstm_model.load_state_dict(state_dict)
            self.lstm_model.to(self.device)
            self.lstm_model.eval()
            logger.info("Loaded PyTorch LSTM Autoencoder state dict.")

        except Exception as e:
            logger.exception("Failed to initialize InferenceEngine artifacts.")
            raise e

    def predict_sequence(
        self,
        features_df: pd.DataFrame,
        base_threshold: float = 1.4066,
        profile_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        try:
            if len(features_df) != 30:
                raise ValueError(f"Sequence length must be exactly 30 rows, received {len(features_df)}.")

            if features_df.shape[1] != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, received {features_df.shape[1]}.")

            # 1. Scale all 6 features for the LSTM Autoencoder
            scaled_features = self.scaler.transform(features_df)

            # 2. Isolation Forest point check:
            # Use the 3 features it was fitted on in the notebook (columns [2, 3, 4]: scoring_index, ratio_score, cluster_mean_speed)
            # If your IsoForest was fitted on 3 features, slice the 3 corresponding scaled columns:
            if hasattr(self.isolation_forest, "n_features_in_") and self.isolation_forest.n_features_in_ == 3:
                iso_features = scaled_features[:, [2, 3, 4]]
            else:
                iso_features = scaled_features

            if_preds = self.isolation_forest.predict(iso_features)
            has_point_anomaly = bool(-1 in if_preds)

            # 3. 3D tensor conversion for LSTM: shape (1, 30, 6)
            tensor_input = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 4. PyTorch forward reconstruction pass
            with torch.no_grad():
                reconstructed = self.lstm_model(tensor_input)
                mse_loss = float(torch.mean((tensor_input - reconstructed) ** 2).item())

            # 5. Threshold comparison against base_threshold
            effective_threshold = float(base_threshold * profile_multiplier)
            is_anomaly = bool((mse_loss > effective_threshold) or has_point_anomaly)

            return {
                "is_anomaly": is_anomaly,
                "reconstruction_loss": mse_loss,
                "effective_threshold": effective_threshold,
                "point_anomaly_detected": has_point_anomaly,
                "base_threshold": float(base_threshold),
                "profile_multiplier": float(profile_multiplier)
            }

        except Exception as e:
            logger.exception("Error executing sequence prediction.")
            raise e
        