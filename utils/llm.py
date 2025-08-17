from openai import OpenAI

# ⚠️ Poner aquí tu API key de DeepSeek directamente
DEEPSEEK_API_KEY = "sk-900f90f072b349d8ba65e95e1eabb2ff"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

def ask_deepseek(prompt: str, context: str = "") -> str:
    """Hace una consulta a DeepSeek pasando contexto de los PDFs"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un asistente que responde únicamente usando la información de los documentos proporcionados."},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {prompt}"}
            ],
            temperature=0.4,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error al consultar DeepSeek: {e}"
