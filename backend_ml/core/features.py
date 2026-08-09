import pandas as pd
import numpy as np
from backend_ml.logger import get_logger

logger = get_logger(__name__)

def build_engineered_features(
    df: pd.DataFrame, 
    cluster_speed_map: dict=None, 
    global_train_mean: float=0.0
) -> pd.DataFrame:
    """
    Performs room-specific feature engineering.

    cluster_speed_map and global_train_mean must come from the
    training artifacts belonging to the current room.

    Expected input columns:
        network_speed_kb
        packet_frequency
        text_cluster

    Output features:
        network_speed_kb
        packet_frequency
        scoring_index
        ratio_score
        cluster_mean_speed
        text_cluster

    """
    try:
        logger.debug(f"Starting feature engineering on DataFrame shape: {df.shape}")

        # Verify required input columns
        required = ['network_speed_kb', 'packet_frequency']
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns for feature math: {missing}")
            raise KeyError(f"DataFrame missing columns: {missing}")

        epsilon = 1e-5

        # 1. Scoring Index (Composite network intensity)
        df['scoring_index'] = np.around(
            df['network_speed_kb'] * df['packet_frequency'], 2
        )

        # 2. Ratio Score (Speed vs Packet Density)
        df['ratio_score'] = df['network_speed_kb'] / (df['packet_frequency'] + epsilon)

        # 3. Cluster Mean Speed Mapping 

        cluster_mean_speed = (
            df["text_cluster"]
            .map(cluster_speed_map)
        )

        df["cluster_mean_speed"] = np.around(
            cluster_mean_speed,
            2,
        )
        
        df["cluster_mean_speed"] = (
            df["cluster_mean_speed"]
            .fillna(global_train_mean)
        )

        # exact numeric feature ordering matching training
        feature_columns = [
            'network_speed_kb',
            'packet_frequency',
            'scoring_index',
            'ratio_score',
            'cluster_mean_speed',
            'text_cluster'
        ]

        final_df = df[feature_columns].copy().astype(np.float32)

        logger.debug(f"Feature engineering successful. Columns: {list(final_df.columns)}")
        return final_df

    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}")
        raise
    