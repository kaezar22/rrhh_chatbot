import streamlit as st
from utils.loader import load_files
from utils.vectorstore import create_vectorstore
from utils.llm import ask_deepseek

# 👉 Configuración de la app
st.set_page_config(
    page_title="Caro Answers",
    page_icon="💬",
    layout="wide"
)
st.title("📄 Carolina-Bot")

# 👉 Cargar API key de DeepSeek desde secrets
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

# Archivos a indexar
FILE_PATHS = ["data/reglamento.pdf", "data/recursos_humanos.txt"]

# Inicializar vectorstore solo una vez
if "vectorstore" not in st.session_state:
    st.write("🔄 Cargando documentos...")
    docs = load_files(FILE_PATHS)
    st.session_state.vectorstore = create_vectorstore(docs, api_key=DEEPSEEK_API_KEY)

# Entrada de usuario
question = st.text_input("Qué duda tienes?")

if question:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    related_docs = retriever.get_relevant_documents(question)

    # Construir contexto con los documentos más relevantes
    context = "\n\n".join([d.page_content for d in related_docs])

    # Preguntar a DeepSeek
    answer = ask_deepseek(question, context, api_key=DEEPSEEK_API_KEY)

    # Mostrar respuesta
    st.subheader("Respuesta:")
    st.write(answer)

    # Mostrar contexto
    with st.expander("📚 Contexto usado"):
        st.write(context)
