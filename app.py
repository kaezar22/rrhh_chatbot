import streamlit as st
from utils.loader import load_files
from utils.vectorstore import create_vectorstore
from utils.llm import ask_deepseek

# 👉 Esto debe ir al inicio del archivo, antes de st.title()
st.set_page_config(
    page_title="Caro Answers",
    page_icon="💬",
    layout="wide"
)
st.title("📄 Carolina-Bot")

# Inicializar
FILE_PATHS = ["data/reglamento.pdf", "data/recursos_humanos.txt"]
API_KEY = st.secrets["DEEPSEEK_API_KEY"]  # ⚡ carga el secreto desde .streamlit/secrets.toml

if "vectorstore" not in st.session_state:
    st.write("🔄 Cargando documentos...")
    docs = load_files(FILE_PATHS)
    st.session_state.vectorstore = create_vectorstore(docs, API_KEY)  # 👈 ahora con api_key


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
