import tiktoken
import logging

def split_text_by_tokens(text, max_tokens, encoding_model):
    """
    Split text into chunks based on token count.
    """
    try:
        tokenizer = tiktoken.get_encoding(encoding_model)
    except Exception as e:
        logging.error(f"Error getting tokenizer for model {encoding_model}: {e}")
        return []
    
    chunks = []
    current_chunk = ""
    for word in text.split():
        if len(tokenizer.encode(current_chunk + " " + word)) <= max_tokens:
            current_chunk += " " + word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
