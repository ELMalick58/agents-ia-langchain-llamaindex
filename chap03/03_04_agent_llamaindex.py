import os
import asyncio
from dotenv import load_dotenv

from tavily import TavilyClient

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage

load_dotenv()

# ==========================
# LLM Gemini
# ==========================

llm = GoogleGenAI(
    model=os.getenv("GEMINI_MODEL"),
    api_key=os.getenv("GEMINI_API_KEY")
)

# ==========================
# Tavily
# ==========================

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# ==========================
# Tool
# ==========================

def search_web(query: str) -> str:
    """
    Recherche des informations sur Internet avec Tavily.
    """
    print("\n Outil appelé : search_web")
    print(f"   Requête      : {query}")

    results = tavily.search(
        query=query,
        max_results=3
    )

    return str(results)

# ==========================
# Agent
# ==========================

agent = FunctionAgent(
    tools=[search_web],
    llm=llm,
    system_prompt="""
    Tu es un assistant utile et précis.

    Si une question nécessite des informations récentes
    ou provenant d'Internet, utilise l'outil search_web.

    Sinon répond directement.
    """
)

# ==========================
# Mémoire
# ==========================

chat_history = []


async def ask(question: str) -> str:

    print(f"\n{'='*50}")
    print(f" Question : {question}")
    print(f"{'='*50}")

    response = await agent.run(
        question,
        chat_history=chat_history
    )

    answer = str(response)

    chat_history.append(ChatMessage(role="user", content=question))
    chat_history.append(ChatMessage(role="assistant", content=answer))

    print(f"\n Réponse : {answer}")

    return answer


# ==========================
# Main
# ==========================

async def main():

    # ── Test 1 : Mémoire ──────────────────────────────
    await ask("Bonjour, je suis Elhadji Malick")
    await ask("Quel est mon nom ?")

    # ── Test 2 : Outil Tavily ─────────────────────────
    await ask("Qui est le PDG d'IBM en 2025 ?")

    # ── Test 3 : Réponse directe ──────────────────────
    await ask(
        "Traduis en anglais la phrase suivante : "
        "Les agents IA sont l'avenir."
    )


if __name__ == "__main__":
    asyncio.run(main())