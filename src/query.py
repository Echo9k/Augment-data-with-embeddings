import openai
from app.config import CONFIG
from app.search import search_similar_strings
import logging

logger = logging.getLogger(__name__)

def build_query_message(query: str, df, model: str) -> str:
    """
    Construct the query message with relevant documentation.
    """
    try:
        strings, _ = search_similar_strings(query, df)
        introduction = "Use the below documentation on LangChain to answer the subsequent question. " \
                       "If the answer cannot be found in the documents, write 'I could not find an answer.'"
        documentation = f"Documentation: {strings}"
        full_message = f"{introduction}\n\nQuestion: {query}\n\n{documentation}"
        logger.info("Query message constructed successfully.")
        return full_message
    except Exception as e:
        logger.error(f"Error building query message: {e}")
        raise

def ask(query: str, df, model: str = None, print_message: bool = False) -> str:
    """
    Answer a query using GPT and a dataframe of relevant texts.
    """
    model = model or CONFIG["gpt_model"]
    try:
        message_text = build_query_message(query, df, model)
        if print_message:
            print(message_text)
        messages = [
            {"role": "system", "content": "You answer questions about Langchain."},
            {"role": "user", "content": message_text},
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            api_key=CONFIG["api_key"]
        )
        response_message = response["choices"][0]["message"]["content"]
        logger.info("Response received successfully.")
        return response_message
    except Exception as e:
        logger.error(f"Error during ask() execution: {e}")
        raise
