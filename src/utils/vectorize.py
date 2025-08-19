from dotenv import load_dotenv
import os
from pinecone import Pinecone
import pandas as pd
from supabase import create_client, Client
import openai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Supabase client
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to an existing index
index = pc.Index("socalabg")

def clear_index():
    """Clear all vectors from the Pinecone index."""
    try:
        logger.info("Clearing all vectors from Pinecone index...")
        # Delete all vectors by fetching all IDs first
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        
        if total_vectors > 0:
            # Delete all vectors (this will remove everything)
            index.delete(delete_all=True)
            logger.info(f"Successfully cleared {total_vectors} vectors from the index")
        else:
            logger.info("Index is already empty")
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        raise

def get_embedding(text: str, model="text-embedding-ada-002"):
    """Get embedding for text using OpenAI API."""
    try:
        response = openai.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding for text: {str(e)}")
        return None


def vectorize_songs():
    """Vectorize song descriptions from Supabase into Pinecone."""
    try:
        # Query songs from Supabase
        logger.info("Querying songs from Supabase...")
        response = supabase.table("Songs").select("*").execute()
        
        # Extract the data from the response
        songs = response.data
        
        if not songs:
            logger.warning("No songs found in Supabase")
            return
        
        logger.info(f"Found {len(songs)} songs to vectorize")
        
        vectors = []
        successful_embeddings = 0
        batch_size = 50
        batch_count = 0
        
        for song in songs:
            try:
                # Check if song has required fields
                if 'id' not in song or 'song_desc' not in song:
                    logger.warning(f"Skipping song with missing required fields: {song}")
                    continue
                
                # Skip if song_desc is empty or None
                if not song['song_desc'] or song['song_desc'].strip() == "":
                    logger.warning(f"Skipping song {song['id']} with empty description")
                    continue
                
                # Get embedding for song description
                embedding = get_embedding(song['song_desc'])
                
                if embedding is None:
                    logger.error(f"Failed to get embedding for song {song['id']}")
                    continue
                
                # Prepare metadata with all other fields except 'song_desc'
                metadata = {k: v for k, v in song.items() if k != 'song_desc' and v is not None}
                
                # Pinecone requires IDs as strings
                vectors.append((str(song['id']), embedding, metadata))
                successful_embeddings += 1
                
                # Process in batches of 100
                if len(vectors) >= batch_size:
                    batch_count += 1
                    logger.info(f"Upserting batch {batch_count} with {len(vectors)} vectors...")
                    index.upsert(vectors=vectors)
                    logger.info(f"Successfully upserted batch {batch_count}")
                    vectors = []  # Clear the batch
                
            except Exception as e:
                logger.error(f"Error processing song {song.get('id', 'unknown')}: {str(e)}")
                continue
        
        # Upsert any remaining vectors
        if vectors:
            batch_count += 1
            logger.info(f"Upserting final batch {batch_count} with {len(vectors)} vectors...")
            index.upsert(vectors=vectors)
            logger.info(f"Successfully upserted final batch {batch_count}")
        
        logger.info(f"Successfully processed {successful_embeddings} songs in {batch_count} batches.")
        
    except Exception as e:
        logger.error(f"Error in vectorize_songs: {str(e)}")
        raise

if __name__ == "__main__":
    # Clear the index first
    try:
        clear_index()
    except Exception as e:
        logger.error(f"Failed to clear index: {str(e)}")
        exit(1)
    
    # Vectorize songs
    vectorize_songs()
    print(index.describe_index_stats())