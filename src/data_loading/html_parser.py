from bs4 import BeautifulSoup
import logging

def extract_text_from_html(file_path):
    """
    Extract text from an HTML file.
    """
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return ""
    
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()
