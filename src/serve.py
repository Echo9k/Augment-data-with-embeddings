#!/usr/bin/env python
"""Example LangChain server exposes a retriever."""
from fastapi import FastAPI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import os
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


load_dotenv("/Users/pabloelgueta/Documents/manning_course/qa-using-langchain-and-vector-databases-lp-author/.env")

api_key = os.getenv('OPENAI_API_KEY') 
# Specify the model name to be used for embeddings
model_name = 'text-embedding-ada-002'


embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=api_key
)

vectorstore = Chroma(
    collection_name="langchain_store",
    embedding_function=embeddings,
    persist_directory="/Users/pabloelgueta/Documents/manning_course/qa-using-langchain-and-vector-databases-lp-author/chroma_db"
)

retriever = vectorstore.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(), 
    chain_type="map_reduce", 
    retriever=retriever,
    memory=memory  
)


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)