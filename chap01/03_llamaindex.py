from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = GoogleGenAI(
    model=os.getenv("GEMINI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY")
)

response = llm.complete(
    "Explique ce qu'est un Agent IA en 5 lignes."
)

print(response.text)