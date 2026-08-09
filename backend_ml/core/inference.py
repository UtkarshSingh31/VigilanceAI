import joblib
import torch
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
        hidden_dim: int = 64,
        device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing InferenceEngine on device: {self.device} (input_dim={self.input_dim})")

        try:
            #  Load Scikit-Learn Scaler and Isolation Forest artifacts
            self.scaler = joblib.load(scaler_path)
            self.isolation_forest = joblib.load(isolation_forest_path)
            logger.info("Successfully loaded StandardScaler and IsolationForest artifacts.")

            # Instantiate PyTorch LSTM Autoencoder
            self.lstm_model = LSTMAutoencoder(input_dim=self.input_dim, hidden_dim=hidden_dim)
            self.lstm_model.load_state_dict(
                torch.load(
                    lstm_weights_path, 
                    map_location=self.device),
                    weights_only=True
                )
            self.lstm_model.to(self.device)
            self.lstm_model.eval()  # Freeze weights for forward pass
            logger.info("Successfully loaded PyTorch LSTMAutoencoder weights.")

        except Exception as e:
            logger.error(f"Failed to initialize InferenceEngine artifacts: {e}")
            raise e

    def predict_sequence(
        self, 
        features_df: pd.DataFrame, 
        base_threshold: float = 1.4448,
        profile_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Executes parallel point-anomaly scoring (Isolation Forest) and 
        3D PyTorch sequence reconstruction loss calculation (LSTM Autoencoder).
        
        Expects features_df to contain exactly 6 columns in training order:
        [network_speed_kb, packet_frequency, scoring_index, ratio_score, cluster_mean_speed,text_cluster]
        """
        try:
            logger.debug(f"Starting sequence inference. Input shape: {features_df.shape}")

            # Verify exact 6-feature input dimension
            if features_df.shape[1] != self.input_dim:
                logger.error(f"Feature dimension mismatch! Expected {self.input_dim} columns, got {features_df.shape[1]}.")
                raise ValueError(f"Inference requires exactly {self.input_dim} feature columns.")

            if len(features_df) != 30:
                raise ValueError(
                    f"Expected exactly 30 telemetry rows, got {len(features_df)}."
                )

            #  Scale 6 numerical features using pre-fitted StandardScaler
            scaled_features = self.scaler.transform(features_df)

            #  Parallel Point-Anomaly Check via Isolation Forest
            if_predictions = self.isolation_forest.predict(scaled_features)
            has_point_anomaly = -1 in if_predictions

            #  Reshape scaled features to 3D Tensor: (batch_size=1, sequence_length=30, input_dim=6)
            tensor_input = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)

            #  PyTorch Forward Pass (No Gradients)
            with torch.no_grad():
                reconstructed = self.lstm_model(tensor_input)
                # Compute Mean Squared Error (MSE) reconstruction loss
                mse_loss = torch.mean((tensor_input - reconstructed) ** 2).item()

            #  Dynamic Context Threshold Evaluation
            effective_threshold = base_threshold * profile_multiplier
            is_anomaly = (mse_loss > effective_threshold) or has_point_anomaly

            logger.info(
                f"Inference Complete | Loss: {mse_loss:.4f} | "
                f"Effective Threshold: {effective_threshold:.4f} | "
                f"Point Anomaly: {has_point_anomaly} | Flagged: {is_anomaly}"
            )

            return {
                "is_anomaly": bool(is_anomaly),
                "reconstruction_loss": float(mse_loss),
                "effective_threshold": float(effective_threshold),
                "point_anomaly_detected": bool(has_point_anomaly)
            }

        except Exception as e:
            logger.exception("Error during sequence inference execution.")
            raise e

        