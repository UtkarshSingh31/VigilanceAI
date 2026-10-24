import pandas as pd
import numpy as np
from backend_ml.logger import get_logger

logger = get_logger(__name__)


def build_engineered_features(
    df: pd.DataFrame,
    cluster_speed_map: dict,
    global_train_mean: float
) -> pd.DataFrame:
    try:
        required_columns = {'network_speed_kb', 'packet_frequency', 'text_cluster'}
        missing = required_columns - set(df.columns)
        if missing:
            raise KeyError(f"DataFrame missing required columns: {sorted(missing)}")

        if cluster_speed_map is None:
            raise ValueError("cluster_speed_map is required for feature engineering.")

        epsilon = 1e-5
        processed_df = df.copy()

        # Feature 3: scoring_index
        processed_df['scoring_index'] = np.around(
            processed_df['network_speed_kb'] * processed_df['packet_frequency'], 2
        )

        # Feature 4: ratio_score
        processed_df['ratio_score'] = (
            processed_df['network_speed_kb'] / (processed_df['packet_frequency'] + epsilon)
        )

        # Feature 5: cluster_mean_speed lookup
        processed_df['cluster_mean_speed'] = (
            processed_df['text_cluster']
            .map(cluster_speed_map)
            .fillna(global_train_mean)
        )
        processed_df['cluster_mean_speed'] = np.around(processed_df['cluster_mean_speed'], 2)

        # Authoritative 6-Feature contract
        feature_columns = [
            'network_speed_kb',
            'packet_frequency',
            'scoring_index',
            'ratio_score',
            'cluster_mean_speed',
            'text_cluster'
        ]

        return processed_df[feature_columns].copy().astype(np.float32)

    except Exception as e:
        logger.exception("Error in feature engineering pipeline.")
        raise e