from langchain.agents import initialize_agent, AgentType
from tools import song_tool, ticketmaster_tool, yelp_tool, get_weather_tool, final_answer_tool
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests
from langchain_core.messages import HumanMessage, SystemMessage
import logging

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE = "https://openrouter.ai/api/v1"
# backbone for agent
llm = ChatOpenAI(
    openai_api_key= OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="gpt-4o-mini",
)

# tools for agent
tools = [song_tool, ticketmaster_tool, yelp_tool, get_weather_tool, final_answer_tool]

# configure agent w/ tooling and LLM
agent = initialize_agent(
    tools,
    llm,
    agent='chat-zero-shot-react-description',
    verbose=True,
)

def get_LLM_response(response: str) -> str:
    url = f"{BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "temperature": 0.8,
        "messages": [
            {"role":"system", "content": "You are a SoCal ABG. INCLUDE ALL DETAILS OF THE RESPONSE"},
            {"role":"user", "content": response}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        return response.json()
    except Exception as e:
        print(f"Error getting LLM response: {str(e)}")
        return None

def run_agent(query: str) -> str:
    # agent's response
    response = agent.run(query)
    logging.info(f"Agent response: {response}")
    llm_response = get_LLM_response(response)
    logging.info(f"LLM response: {llm_response}")
    # LLM's response
    return {
        "response": llm_response["choices"][0]["message"]["content"]
    }

