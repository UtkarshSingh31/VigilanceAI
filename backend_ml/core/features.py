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
    Executes the exact numerical feature engineering from the notebook:
    - scoring_index = network_speed_kb * packet_frequency
    - ratio_score = network_speed_kb / (packet_frequency + epsilon)
    - cluster_mean_speed mapping
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

        # 3. Cluster Mean Speed Mapping (if text_cluster exists)
        if 'text_cluster' in df.columns and cluster_speed_map is not None:
            df['cluster_mean_speed'] = (
                df['text_cluster']
                .map(cluster_speed_map)
                .fillna(global_train_mean)
            )
            df['cluster_mean_speed'] = np.around(df['cluster_mean_speed'], 2)

        # Select exact numeric feature ordering matching training
        feature_columns = [
            'network_speed_kb',
            'packet_frequency',
            'scoring_index',
            'ratio_score'
        ]

        if 'text_cluster' in df.columns:
            feature_columns.extend(['text_cluster', 'cluster_mean_speed'])

        final_df = df[feature_columns].copy().astype(np.float32)

        logger.debug(f"Feature engineering successful. Columns: {list(final_df.columns)}")
        return final_df

    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}")
        raise