import random
import json
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


load_dotenv()


# Fonction météo
@tool
def weather_tool(city: str) -> str:
    """Provides the weather information for a given city."""
    temperature = random.randint(1, 70)
    result = {
        "city": city,
        "temperature": temperature,
        "unit": "Celsius"
    }
    return json.dumps(result)


# LLM Gemini avec outils liés
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# Lier les outils au LLM
llm_with_tools = llm.bind_tools([weather_tool])

# Prompt
prompt = "Quel temps fait-il à Paris ?"
print(f"\n Question : {prompt}")

# Appel au LLM
response = llm_with_tools.invoke([HumanMessage(content=prompt)])

# Vérifier si le LLM a appelé un outil
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"\n Outil appelé : {tool_call['name']}")
        print(f"   Arguments    : {tool_call['args']}")

        # Exécuter l'outil
        result = weather_tool.invoke(tool_call['args'])
        print(f"   Résultat     : {result}")

        # Réponse finale
        final_response = llm.invoke([
            HumanMessage(content=prompt),
            response,
            HumanMessage(content=f"Résultat de l'outil : {result}")
        ])
        print(f"\n Réponse : {final_response.content}")
else:
    print(f"\n Réponse directe : {response.content}")