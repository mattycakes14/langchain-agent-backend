from utils.embedding import get_embedding
from models.state import State
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
import logging
from arcadepy import Arcade
from langchain_arcade import ArcadeToolManager
from pinecone import Pinecone
import openai
from config.settings import user_id
# Configure logging with more detail
logging.basicConfig(level=logging.INFO)
# load environment variables
load_dotenv()

# Initialize Arcade client
client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect to the index
index = pc.Index("socalabg")

# search for songs using vector similarity
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the song recommendation
def search_songs(state: State) -> State:
    """Search for songs using vector similarity."""
    query = state["messages"][-1].content
    try:
        # Get embedding for the query
        print(f"Getting embedding for query: '{query}'")
        query_embedding = get_embedding(query)
        
        if query_embedding is None:
            print("Failed to get embedding for query")
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {"error": "Failed to get embedding for query"}
            }
        results = index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True
        )
        
        # Format the results properly
        if results.matches:
            song_info = results.matches[0].metadata

            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {
                    "song_recommendation": song_info
                }
            }
        else:
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {"error": "No songs found matching your request"}
            }
            
    except Exception as e:
        print(f"Error searching songs: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {"error": f"Error searching songs: {str(e)}"}
        }

    