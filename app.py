import streamlit as st
from utils.loader import load_files
from utils.vectorstore import create_vectorstore
from utils.llm import ask_gemini

# 👉 Configuración de la página
st.set_page_config(
    page_title="Caro Answers",
    page_icon="💬",
    layout="wide"
)
st.title("📄 Carolina-Bot")

# Inicializar
FILE_PATHS = ["data/reglamento.pdf", "data/recursos_humanos.txt"]

# Cargar documentos y vectorstore
if "vectorstore" not in st.session_state:
    st.write("🔄 Cargando documentos...")
    docs = load_files(FILE_PATHS)
    st.session_state.vectorstore = create_vectorstore(docs)

# Interfaz
user_input = st.text_input("💬 Pregunta algo sobre RRHH")
if user_input:
    retriever = st.session_state.vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(user_input)

    messages = [
        {"role": "system", "content": "Eres un asistente experto en políticas de RRHH."},
        {"role": "user", "content": user_input}
    ]

    answer = ask_gemini(messages)

    st.write("🤖 Respuesta:", answer)
