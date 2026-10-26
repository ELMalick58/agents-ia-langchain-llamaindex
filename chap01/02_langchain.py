from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=float(os.getenv("TEMPERATURE"))
)

response = llm.invoke(
    "Explique ce qu'est un Agent IA en 5 lignes."
)

print(response.content)