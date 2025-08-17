from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 🔑 Clave fija aquí
    embeddings = OpenAIEmbeddings(
        model="deepseek-reasoner",   
        openai_api_key="sk-900f90f07b2349d8ba65e95e1eabb2ff",
        openai_api_base="https://api.deepseek.com"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
