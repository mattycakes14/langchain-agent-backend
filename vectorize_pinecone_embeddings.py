from dotenv import load_dotenv
import os
from pinecone import Pinecone
import pandas as pd
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Supabase client
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to an existing index
index = pc.Index("socalabg")

def vectorize_songs():
    """Vectorize song descriptions from Supabase into Pinecone using Pinecone's built-in embeddings."""
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
                
                # Prepare metadata with all other fields except 'song_desc'
                metadata = {k: v for k, v in song.items() if k != 'song_desc' and v is not None}
                
                # Use Pinecone's upsert with text (Pinecone will generate embeddings automatically)
                # The text will be automatically embedded using Pinecone's default model
                vectors.append({
                    'id': str(song['id']),
                    'values': song['song_desc'],  # Pinecone will embed this text
                    'metadata': metadata
                })
                successful_embeddings += 1
                
                # Process in batches of 50
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

def vectorize_with_specific_model():
    """Alternative approach using Pinecone's specific embedding models."""
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
                
                # Prepare metadata with all other fields except 'song_desc'
                metadata = {k: v for k, v in song.items() if k != 'song_desc' and v is not None}
                
                # Use Pinecone's upsert with specific embedding model
                vectors.append({
                    'id': str(song['id']),
                    'values': song['song_desc'],
                    'metadata': metadata,
                    'sparse_values': None,  # For dense embeddings
                    'embedding_model': 'text-embedding-ada-002'  # Specify the model
                })
                successful_embeddings += 1
                
                # Process in batches of 50
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
    # Try the first approach (automatic embedding)
    try:
        vectorize_songs()
    except Exception as e:
        logger.error(f"First approach failed: {str(e)}")
        logger.info("Trying alternative approach with specific model...")
        vectorize_with_specific_model()
    
    print(index.describe_index_stats()) 