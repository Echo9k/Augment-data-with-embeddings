import logging
import sys
from data_loading.config import Config
from data_loading.directory_processor import process_directory

def setup_logging(level):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def main():
    # Load configuration
    config = Config()
    max_tokens = config.get("app.max_tokens", 400)
    html_extension = config.get("app.html_file_extension", ".html")
    encoding_model = config.get("app.encoding_model", "cl100k_base")
    log_level = config.get("app.log_level", "INFO")
    
    setup_logging(log_level)
    logging.info("Starting directory processing...")
    
    # For example, process the 'html_files' directory (adjust as needed)
    directory = r"G:/My Drive/wdir/repos/Manning/Augment data with embeddings/data"
    results = process_directory(directory, max_tokens, html_extension, encoding_model)
    
    # Display or further process results
    for file_path, chunks in results.items():
        logging.info(f"{file_path}: {len(chunks)} chunks extracted.")
    
main()
