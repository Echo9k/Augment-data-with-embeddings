# app/data_loader.py
import os
import pandas as pd
import logging
from app.embedding import get_embedding

logger = logging.getLogger(__name__)

def load_data_from_directory(directory: str = "data/") -> pd.DataFrame:
    """
    Read text files from the specified directory, create embeddings for each file,
    and return a DataFrame with columns: 'filename', 'text', 'embedding'.
    """
    data = []
    if not os.path.exists(directory):
        logger.error(f"Data directory {directory} does not exist.")
        return pd.DataFrame(data)
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                embedding = get_embedding(text)
                data.append({"filename": filename, "text": text, "embedding": embedding})
                logger.info(f"Processed file: {filename}")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
    
    return pd.DataFrame(data)

# from app.data_loader import load_data_from_directory
