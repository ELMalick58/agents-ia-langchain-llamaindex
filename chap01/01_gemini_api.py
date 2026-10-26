import google.generativeai as genai
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Initialiser Gemini
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    model_name=os.getenv("GEMINI_MODEL"),
    system_instruction="""
    Je souhaite écrire un livre sur le sujet correspondant à la description ci-dessous.

    Proposez-moi 3 titres accrocheurs pour ce livre.

    Répondez uniquement avec les titres, un par ligne, sans texte supplémentaire.
    """
)

response = model.generate_content(
    "Le sujet porte sur la tarte au citron.",
    generation_config=genai.GenerationConfig(
        temperature=float(os.getenv("TEMPERATURE")),
        max_output_tokens=int(os.getenv("MAX_TOKENS"))
    )
)

print(response.text)