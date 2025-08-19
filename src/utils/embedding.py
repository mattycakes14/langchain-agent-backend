from langchain_openai import OpenAIEmbeddings
import openai
import os
from dotenv import load_dotenv
from config.settings import OPENROUTER_API_KEY

load_dotenv()

# Set up OpenAI API key - use OpenRouter for chat, regular OpenAI for embeddings
if OPENROUTER_API_KEY:
    # For embeddings, we need to use regular OpenAI API
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

# embedd user query
# Parameters:
# text: str - The text to embed
# model: str - The model to use for embedding
# Returns:
# list - The embedding of the text
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
