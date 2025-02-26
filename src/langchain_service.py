# app/langchain_service.py
import os
import logging
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from app.config import config

# Load environment variables from .env file
dotenv_path = os.getenv("DOTENV_PATH", ".env")
load_dotenv(dotenv_path)

# Configure logging
logging.basicConfig(level=config.get("logging", {}).get("level", "INFO"))
logger = logging.getLogger(__name__)

def create_langchain_chain():
    try:
        api_key = os.getenv("OPENAI_API_KEY", config.get("openai_api_key"))
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables or config.")

        model_name = config.get("model_name", "text-embedding-ada-002")
        logger.info("Initializing embeddings...")
        embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)

        chroma_config = config.get("chroma", {})
        logger.info("Setting up vectorstore...")
        vectorstore = Chroma(
            collection_name=chroma_config.get("collection_name", "langchain_store"),
            embedding_function=embeddings,
            persist_directory=chroma_config.get("persist_directory", "./chroma_db")
        )

        retriever = vectorstore.as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        logger.info("Creating conversational chain...")
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(), 
            chain_type="map_reduce", 
            retriever=retriever,
            memory=memory  
        )
        return chain
    except Exception as e:
        logger.error("Failed to create LangChain chain", exc_info=True)
        raise e
