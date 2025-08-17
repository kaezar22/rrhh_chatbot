from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def create_vectorstore(docs):
    # Dividir documentos en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Modelo de embeddings ligero y forzado a CPU
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # 👈 fuerza CPU para evitar el error
    )

    # Crear el vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

