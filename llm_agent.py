from langchain.agents import initialize_agent, AgentType
from tools import song_tool, ticketmaster_tool, yelp_tool, get_weather_tool, final_answer_tool
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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

def run_agent(query: str) -> str:
    # TODO: pre-process/ clean up user query

    return agent.run(query)
