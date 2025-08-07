from langchain.agents import initialize_agent, AgentType
from tools import song_tool, ticketmaster_tool
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# personality prompt


# backbone for agent
llm = ChatOpenAI(
    openai_api_key= OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="gpt-4o-mini",
)

# tools for agent
tools = [song_tool, ticketmaster_tool]

# configure agent w/ tooling and LLM
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"system_message": "You are a SoCal ABG that raves, drinks boba, and loves Seaside Bakery"},
    max_iterations=2
)

def run_agent(query: str) -> str:
    return agent.run(query)
