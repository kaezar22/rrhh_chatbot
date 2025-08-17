from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_vectorstore(docs, api_key: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="deepseek-embedding",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

