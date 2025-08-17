import streamlit as st
from openai import OpenAI

def ask_gemini(messages):
    google_api_key = "AIzaSyCoMFqkP0COT38Ik61sy44w1BRg5AlFBdk"

    # Cliente OpenAI-compatible apuntando a Gemini
    gemini = OpenAI(
        api_key=google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = gemini.chat.completions.create(
        model="gemini-2.0-flash",  # 🚀 puedes probar también gemini-1.5-pro
        messages=messages
    )

    return response.choices[0].message.content

