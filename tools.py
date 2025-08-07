from langchain.agents import Tool
import openai
import os
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to the index
index = pc.Index("socalabg")

def get_embedding(text: str, model="text-embedding-ada-002"):
    """Get embedding for text using OpenAI API."""
    try:
        response = openai.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return None

def search_songs(query: str, top_k: int = 5):
    """Search for songs using vector similarity."""
    try:
        # Get embedding for the query
        print(f"Getting embedding for query: '{query}'")
        query_embedding = get_embedding(query)
        
        if query_embedding is None:
            print("Failed to get embedding for query")
            return
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results["matches"][0].metadata
            
    except Exception as e:
        print(f"Error searching songs: {str(e)}")

def recommend_song(query: str) -> str:
    return search_songs(query)

def ticketmaster_search_event(query: str) -> str:
    # Your code to call TicketMaster API and return event info
    return f"Found festival info for {query}."

# Wrap functions as LangChain Tools
song_tool = Tool(
    name="getSongRecommendation",
    func=recommend_song,
    description="Provide a song recommendation based on a user's request."
)

ticketmaster_tool = Tool(
    name="TicketMasterSearch",
    func=ticketmaster_search_event,
    description="Retrieve real-time festival dates and ticket info from TicketMaster."
)
