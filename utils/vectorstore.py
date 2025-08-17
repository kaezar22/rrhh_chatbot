from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
import streamlit as st

def create_vectorstore(docs):
    # 🔑 Cargar API key de Google
    google_api_key = "AIzaSyCoMFqkP0COT38Ik61sy44w1BRg5AlFBdk"

    # 🔹 Splitting de documentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 🔹 Embeddings con VertexAI
    embeddings = VertexAIEmbeddings(model="text-embedding-004")  

    # 🔹 Crear vectorstore con FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

