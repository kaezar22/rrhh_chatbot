import streamlit as st
from utils.loader import load_files
from utils.vectorstore import create_vectorstore
from utils.llm import ask_deepseek

# 👉 Configuración de la página
st.set_page_config(
    page_title="Caro Answers",
    page_icon="💬",
    layout="wide"
)
st.title("📄 Carolina-Bot")

# Inicializar
FILE_PATHS = ["data/reglamento.pdf", "data/recursos_humanos.txt"]

# 🔑 Clave fija aquí (no usa st.secrets)
API_KEY = "sk-900f90f07b2349d8ba65e95e1eabb2ff"

if "vectorstore" not in st.session_state:
    st.write("🔄 Cargando documentos...")
    docs = load_files(FILE_PATHS)
    st.session_state.vectorstore = create_vectorstore(docs, API_KEY)

# Entrada usuario
question = st.text_input("Qué duda tienes?")

if question:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    related_docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([d.page_content for d in related_docs])
    answer = ask_deepseek(question, context)

    st.subheader("Respuesta:")
    st.write(answer)

    with st.expander("📚 Contexto usado"):
        st.write(context)
