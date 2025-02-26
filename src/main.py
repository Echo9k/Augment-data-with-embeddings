import pandas as pd
import logging
import sys
from app.query import ask
from app.config import CONFIG

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

def load_sample_dataframe() -> pd.DataFrame:
    """
    Loads or creates a sample dataframe with text and embedding columns.
    In a real application, replace this with your data source.
    """
    # This is a dummy dataframe for demonstration.
    return pd.DataFrame([
        {"text": "LangChain documentation overview", "embedding": [0.1, 0.2, 0.3]},
        {"text": "Installation instructions for LangChain", "embedding": [0.2, 0.1, 0.4]},
    ])

def main():
    logger.info("Application starting up...")
    df = load_sample_dataframe()
    
    # Example query; in practice, obtain this dynamically or via CLI/HTTP endpoint.
    query_text = "How do I install LangChain?"
    
    try:
        answer = ask(query_text, df, print_message=True)
        print("Answer:", answer)
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
s