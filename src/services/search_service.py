from models.state import State
import logging
from dotenv import load_dotenv
from tavily import TavilyClient
import os
# Configure logging with more detail
logging.basicConfig(level=logging.INFO)

# load environment variables
load_dotenv()


# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# search engine fallback for user query
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the search engine results
def search_web(state: State) -> State:
    """Search the web for the user query"""
    logging.info("[SEARCHING THE WEB] Searching the web for the user query")
    query = state["messages"][-1].content
    results = tavily_client.search(query, max_results=5)

    logging.info("[SEARCHING THE WEB] Search results: " + str(results))
    return {
        "search_results": results
    }
