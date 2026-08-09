import pandas as pd
from typing import Dict, Any, List
from backend_ml.core.embeddings import assign_text_clusters
from backend_ml.core.features import build_engineered_features
from backend_ml.core.inference import InferenceEngine
from backend_ml.logger import get_logger

logger = get_logger(__name__)

class PipelineOrchestrator:
    def __init__(
        self,
        room_configs:Dict[str,Dict[str,Any]],
        scaler_path: str,
        isolation_forest_path: str,
        lstm_weights_path: str,
        base_threshold: float = 1.7520,
        input_dim: int = 6,
        hidden_dim: int = 64,
        device: str = "cpu"
    ):
        """
        Initializes the Vigilance ML pipeline.

        room_configs contains room-specific:
            - HDBSCAN clusterer path
            - cluster_speed_map
            - global_train_mean

        Example:

        room_configs = {
            "room_101": {
                "clusterer_path": ".../room_101_hdbscan.joblib",
                "cluster_speed_map": {...},
                "global_train_mean": 842.31,
            },
            "room_102": {
                "clusterer_path": ".../room_102_hdbscan.joblib",
                "cluster_speed_map": {...},
                "global_train_mean": 521.77,
            },
        }
        """
        self.room_configs=room_configs
        self.base_threshold = base_threshold

        logger.info(
            f"Initializing PipelineOrchestrator for "
            f"{len(room_configs)} rooms."
        )

        self.inference_engine = InferenceEngine(
            scaler_path=scaler_path,
            isolation_forest_path=isolation_forest_path,
            lstm_weights_path=lstm_weights_path,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        logger.info("PipelineOrchestrator and InferenceEngine initialized successfully.")

    def process_telemetry_batch(
        self,
        room_id:str,
        telemetry_data: List[Dict[str, Any]],
        profile_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Executes the complete ML inference pipeline for one
        30-step telemetry batch belonging to a specific room.

        Flow:

        telemetry
            ↓
        room-specific HDBSCAN
            ↓
        room-specific cluster_speed_map
            ↓
        room-specific global_train_mean
            ↓
        6 engineered features
            ↓
        StandardScaler
            ↓
        Isolation Forest
            ↓
        LSTM Autoencoder
            ↓
        anomaly result.

        """
        try:

            logger.debug(
                f"Received telemetry batch for room={room_id} "
                f"with {len(telemetry_data)} items."
            )

            if room_id not in self.room_configs:
                raise ValueError(
                    f"No ML configuration found for room: {room_id}"
                )

            room_config = self.room_configs[room_id]

            clusterer_path = room_config["clusterer_path"]
            cluster_speed_map = room_config["cluster_speed_map"]
            global_train_mean = room_config["global_train_mean"]

            logger.debug(
                f"Using ML artifacts for room={room_id}"
            )
            
            raw_df = pd.DataFrame(telemetry_data)

            # Text Clustering via SentenceTransformers + HDBSCAN approximate_predict()
            clustered_df = assign_text_clusters(
                raw_df, 
                clusterer_path=clusterer_path
            )

            # Numeric Feature Engineering (Enforcing strict 6-feature contract)
            engineered_df = build_engineered_features(
                clustered_df,
                cluster_speed_map=cluster_speed_map,
                global_train_mean=global_train_mean
            )

            #  Inference Execution (StandardScaler -> Isolation Forest -> PyTorch LSTM)
            results = self.inference_engine.predict_sequence(
                features_df=engineered_df,
                base_threshold=self.base_threshold,
                profile_multiplier=profile_multiplier
            )

            logger.info(
                f"Pipeline execution complete | "
                f"Room: {room_id} | "
                f"Anomaly: {results['is_anomaly']} | "
                f"Loss: {results['reconstruction_loss']:.4f} | "
                f"Threshold: {results['effective_threshold']:.4f}"
            )

            return results

        except Exception:
            logger.exception(
                f"Error executing pipeline for room={room_id}."
            )
            raise
        