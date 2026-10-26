from llama_index.core import Document, VectorStoreIndex, Settings

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Configuration du LLM Gemini
Settings.llm = GoogleGenAI(
    model=os.getenv("GEMINI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY")
)

# Configuration des embeddings Gemini
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-001",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Documents
documents = [
    Document(
        text="Garry Kasparov est né le 13 avril 1963 à Bakou, en Azerbaïdjan, et est devenu l'un des plus grands joueurs d'échecs de l'histoire."
    ),
    Document(
        text="En 1985, à seulement 22 ans, Kasparov est devenu le plus jeune champion du monde d'échecs."
    ),
    Document(
        text="Kasparov est célèbre pour ses matchs historiques contre des ordinateurs, notamment Deep Blue."
    ),
    Document(
        text="En 2005, Kasparov a pris sa retraite des compétitions d'échecs pour se consacrer à la politique."
    ),
    Document(
        text="Auteur prolifique et orateur, Kasparov continue d'influencer les domaines des échecs et de la politique."
    ),
    Document(
        text="Zinédine Zidane est né le 23 juin 1972 à Marseille, en France."
    ),
    Document(
        text="En 1998, Zidane a mené l'équipe de France à la victoire en Coupe du Monde."
    ),
    Document(
        text="Zidane a remporté le Ballon d'Or en 1998."
    ),
    Document(
        text="Après sa carrière de joueur, Zidane est devenu entraîneur du Real Madrid."
    ),
    Document(
        text="Zidane est reconnu pour sa vision du jeu, sa technique et son calme."
    ),
]

# Création de l'index vectoriel
index = VectorStoreIndex.from_documents(documents)

# Moteur de recherche
query_engine = index.as_query_engine()

# Questions
response1 = query_engine.query(
    "Qu'est-ce qu'il est devenu, Zinédine ?"
)

print("\nQUESTION 1")
print(response1)

response2 = query_engine.query(
    "Qui a fait de la politique ?"
)

print("\nQUESTION 2")
print(response2)

response3 = query_engine.query(
    "Quelles sont les personnes concernées par nos documents ?"
)

print("\nQUESTION 3")
print(response3)