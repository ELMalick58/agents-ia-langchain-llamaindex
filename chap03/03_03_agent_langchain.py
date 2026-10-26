import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

# LLM Gemini
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# Outil Tavily
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# LLM avec outils
llm_with_tools = llm.bind_tools(tools)

# Mémoire
chat_history = []


def ask(question: str):
    print(f"\n{'='*50}")
    print(f" Question : {question}")
    print(f"{'='*50}")

    # Messages avec historique
    messages = chat_history + [HumanMessage(content=question)]

    # Appel au LLM
    response = llm_with_tools.invoke(messages)

    # Si le LLM veut utiliser un outil
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"\n🔧 Outil appelé : {tool_call['name']}")
            print(f"   Arguments    : {tool_call['args']}")

            # Exécution de l'outil
            tool_result = search_tool.invoke(tool_call['args'])
            print(f"   Résultat     : {str(tool_result)[:200]}...")

            # Réponse finale
            messages_with_tool = messages + [
                response,
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call['id']
                )
            ]
            final_response = llm.invoke(messages_with_tool)
            output = final_response.content

    else:
        output = response.content

    # Mise à jour mémoire
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=output))

    print(f"\n Réponse : {output}")
    return output


# ── Test 1 : Mémoire ──────────────────────────────
ask("Bonjour, je suis Elhadji Malick")
ask("Quel est mon nom ?")

# ── Test 2 : Outil de recherche ───────────────────
ask("Qui est le PDG d'IBM en 2015 ?")

# ── Test 3 : Raisonnement direct ──────────────────
ask("Traduis en anglais la phrase suivante : Les agents IA sont l'avenir.")