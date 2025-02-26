import pandas as pd
from scipy import spatial
from app.embedding import get_embedding
import logging

logger = logging.getLogger(__name__)

def search_similar_strings(search_term: str, data_frame: pd.DataFrame, limit: int = None) -> tuple[list[str], list[float]]:
    """
    Search for strings in the data_frame that are most similar to the given search_term.
    """
    if limit is None:
        from app.config import CONFIG
        limit = CONFIG.get("result_limit", 100)

    try:
        search_embedding = get_embedding(search_term)
    except Exception as e:
        logger.error("Failed to obtain search embedding.")
        raise

    def similarity(x, y):
        return 1 - spatial.distance.cosine(x, y)

    try:
        results = [
            (row["text"], similarity(search_embedding, row["embedding"]))
            for _, row in data_frame.iterrows()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        texts, scores = zip(*results) if results else ([], [])
        logger.info("Search completed successfully.")
        return list(texts[:limit]), list(scores[:limit])
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise
