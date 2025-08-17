from langchain.document_loaders import PyPDFLoader, TextLoader

def load_files(file_paths):
    docs = []
    for path in file_paths:
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif path.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
        else:
            print(f"⚠️ Tipo de archivo no soportado: {path}")
    return docs
