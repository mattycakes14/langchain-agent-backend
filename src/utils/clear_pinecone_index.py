from dotenv import load_dotenv
import os
from pinecone import Pinecone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to the index
index = pc.Index("socalabg")

def clear_index():
    """Clear all vectors from the Pinecone index."""
    try:
        logger.info("Clearing all vectors from Pinecone index...")
        # Delete all vectors
        index.delete(delete_all=True)
        logger.info("Successfully cleared all vectors from the index")
        
        # Verify the index is empty
        stats = index.describe_index_stats()
        logger.info(f"Index now contains {stats.total_vector_count} vectors")
        
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        raise

if __name__ == "__main__":
    clear_index()
