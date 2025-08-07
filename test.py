from dotenv import load_dotenv
import os
from pinecone import Pinecone
import openai

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
        
        print(f"\nFound {len(results.matches)} results:")
        print("=" * 50)
        
        for i, match in enumerate(results.matches, 1):
            print(f"\n{i}. Song ID: {match.id}")
            print(f"   Score: {match.score:.4f}")
            print(f"   Metadata: {match.metadata}")
            
    except Exception as e:
        print(f"Error searching songs: {str(e)}")

if __name__ == "__main__":
    # Test with a specific query
    user_query = input("Enter your search query: ").strip()
    
    if not user_query:
        user_query = "happy upbeat music"
        print(f"Using default query: '{user_query}'")
    
    search_songs(user_query, top_k=5)
