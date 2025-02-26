import os
import logging
from data_loading.html_parser import extract_text_from_html
from data_loading.token_splitter import split_text_by_tokens

def process_directory(directory, max_tokens, html_extension, encoding_model):
    """
    Process HTML files in a directory, extracting text and splitting into token chunks.
    """
    results = {}
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(html_extension):
                file_path = os.path.join(dirpath, filename)
                logging.info(f"Processing file: {file_path}")
                text = extract_text_from_html(file_path)
                if text:
                    chunks = split_text_by_tokens(text, max_tokens, encoding_model)
                    results[file_path] = chunks
    return results
