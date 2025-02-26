import openai
from app.config import CONFIG
import logging

logger = logging.getLogger(__name__)

def get_embedding(text: str) -> list:
    """
    Get embedding for a given text using OpenAI API.
    """
    try:
        response = openai.Embedding.create(
            model=CONFIG["embedding_model"],
            input=text,
            api_key=CONFIG["api_key"]
        )
        embedding = response["data"][0]["embedding"]
        logger.info("Embedding retrieved successfully.")
        return embedding
    except Exception as e:
        logger.error(f"Error fetching embedding: {e}")
        raise
