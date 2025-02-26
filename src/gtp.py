import openai
import pandas as pd
import tiktoken
from src.search import search_similar_strings

from data_loading.config import Config

cfg = Config()

def num_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(cfg.gpt_model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    message: str = "Here's the documentation I found:",
) -> str:
    """
    Return a message for GPT with relevant source texts pulled from a dataframe.
    The message includes an introduction, the documentation snippets, and the question.
    """
    strings, relatednesses = search_similar_strings(query, df)
    # Build the message using the relevant parts.
    introduction = (
        "Use the below documentation on LangChain to answer the subsequent question. "
        "If the answer cannot be found in the documents, write 'I could not find an answer.'"
    )
    documentation = f"\nDocumentation: {strings}"
    question = f"\n\nQuestion: {query}"
    return f"{introduction}\n{message}{documentation}{question}"


def ask(
    query: str,
    df: pd.DataFrame,
    print_message: bool = False,
) -> str:
    """
    Answers a query using GPT and a dataframe of relevant texts and embeddings.
    """
    message = query_message(query, df)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about LangChain."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=cfg.gpt_model,
        messages=messages,
        temperature=0,
        api_key=cfg.api_key
    )
    # Access the first choice returned by the API
    response_message = response["choices"][0]["message"]["content"]
    return response_message
