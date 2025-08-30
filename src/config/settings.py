from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from dotenv import load_dotenv
import os
import logging

# Load environment variables first
load_dotenv()

#user id for arcade
user_id = os.getenv("ARCADE_USER_ID")

# Configure logging with more detail
logging.basicConfig(level=logging.INFO)

# Set up OpenAI API key for OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if we have the required API keys
if not OPENROUTER_API_KEY:
    print("⚠️  Warning: OPENROUTER_API_KEY not found in environment variables")
    print("Please add OPENROUTER_API_KEY=your_key_here to your .env file")

if not OPENAI_API_KEY:
    print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
    print("Please add OPENAI_API_KEY=your_key_here to your .env file")

# Initialize LLMs only if we have the required API keys
llm_main = None
llm_fast = None
llm_embeddings = None

# llm_main is used for the main LLM that returns a structured output
llm_main = ChatOpenAI(
    model="openai/gpt-5",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0
)


llm_advanced = ChatOpenAI(
    model="openai/gpt-5",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0
)

# llm_fast is used for the fast LLM that returns a string
llm_fast = ChatOpenAI(
    model="openai/gpt-4.1-nano",
    base_url="https://openrouter.ai/api/v1", 
    api_key=OPENROUTER_API_KEY,
    temperature=0
)

llm_personality = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.5,
    max_tokens=1000  # Limit the response size
)

# For embeddings, we need to use the regular OpenAI API
# since OpenRouter doesn't support embeddings
llm_embeddings = ChatOpenAI(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY"),  # Use regular OpenAI API key for embeddings
    temperature=0
)