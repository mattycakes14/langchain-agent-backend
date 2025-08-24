from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import requests
from graph.graph_builder import compiled_graph
from langchain_core.messages import HumanMessage
import logging
from arcadepy import Arcade
import supabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

# initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# load environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ARCADE_API_KEY = os.getenv("ARCADE_API_KEY")

# create supabase client
supabase_client = supabase.create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Initialize Arcade client
client = Arcade(api_key=ARCADE_API_KEY)

# class that defines the request body for the /query endpoint
class PromptRequest(BaseModel):
    prompt: str
    user_id: str


# GET /health
@app.get("/health")
def health_check():
    return {"message": "OK"}

# POST /auth
@app.post("/auth/userintegrations")
def handle_auth(request: dict):
    """
    Handle authentication for various services using Arcade client before query processing.
    This replaces the auth logic that was previously handled by Arcade AI during tool execution.
    """
    user_id = request.get("user_id")
    # fetch the status of user integrations
    try:
        # fetch the user integrations
        user_integrations = supabase_client.table("user_integrations").select("*").eq("user_id", user_id).execute()
        logging.info("[AUTH] User integrations: " + str(user_integrations))

        hasPendingStatus = False
        for integration in user_integrations.data:
            if integration.get("status") == "pending":
                hasPendingStatus = True
                break
        
        if hasPendingStatus:
            return {"status": "pending"}
        else:
            return {"status": "completed"}
        
    except Exception as e:
        logging.error("[AUTH] Error fetching user integrations: " + str(e))
        return {"error": "Error fetching user integrations"}

    
# POST /query
@app.post("/query")
def handle_prompt(request: PromptRequest):
    user_query = request.prompt
    user_id = request.user_id

    logging.info(f"[QUERY] Received query: {user_query}")
    result = compiled_graph.invoke({"messages": [HumanMessage(content=user_query)]})
    logging.info("[FINAL RESULT]: " + str(result))
    return result

