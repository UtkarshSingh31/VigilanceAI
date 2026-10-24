import re 
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from backend_ml.logger import get_logger
from sentence_transformers import SentenceTransformer
from hdbscan.prediction import approximate_predict

logger=get_logger(__name__)

try:
    model=SentenceTransformer('BAAI/bge-small-en-v1.5')
    logger.info("SentenceTransformer model loaded successfully.")
    
except Exception as e:
    logger.error("Error loading model")
    raise e


def clean_text(text:str)->str:
    """
    clean text for title for every window title strings
    """
    regex_pattern = r'\s*-\s*(Google Chrome|Brave|Mozilla Firefox|Microsoft Edge)\s*$'
    cleaned=re.sub(regex_pattern,'',str(text))

    return cleaned.strip()

def assign_text_cluster(
        df:pd.DataFrame,
        clusterer_path: Optional[str] = None
)->pd.DataFrame:

    try:
        logger.info(f"Assigning text clusters for DataFrame with {len(df)} rows.")

        processed_df=df.copy()

        if 'window_title' not in processed_df.columns:
            logger.error("'window_title' column missing from input DataFrame.")
            raise KeyError("'window_title' column is required for text clustering.")

        # clean window title.
        processed_df['cleaned_window_title']=processed_df['window_title'].apply(clean_text)

        unique_titles = [str(t) for t in processed_df['cleaned_window_title'].unique()]
        logger.debug(f"Found {len(unique_titles)} unique cleaned window titles.")

        embeddings=model.encode(unique_titles,show_progress_bar=False)

        if clusterer_path:
            logger.debug(f"Loading HDBSCAN clusterer from: {clusterer_path}")
            clusterer = joblib.load(clusterer_path)
            cluster_labels, strengths = approximate_predict(clusterer, embeddings)
        else:
            logger.warning("No HDBSCAN clusterer path provided. Assigning noise label (-1) to titles.")
            cluster_labels = np.full(len(unique_titles), -1)


        # map cluster back to titles
        title_to_cluster=dict(zip(unique_titles, cluster_labels))
        processed_df['text_cluster']=processed_df['cleaned_window_title'].map(title_to_cluster).astype(int)

        logger.info("Text clustering completed successfully.")
        return processed_df

    except Exception as e:
        logger.error("Error occurred during text cluster assignment.")
        raise e
    



