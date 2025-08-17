from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # usamos el wrapper OpenAI-compatible

def create_vectorstore(docs, api_key: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings usando DeepSeek vía API OpenAI-compatible
    embeddings = OpenAIEmbeddings(
        model="deepseek-reasoner",   # 👈 o "deepseek-embedding" si tu endpoint lo soporta
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com"  # 👈 importante
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
