from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import requests
from llm_agent import run_agent
load_dotenv()

# initialize FastAPI app
app = FastAPI()

# load environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# class that defines the request body for the /query endpoint
class PromptRequest(BaseModel):
    prompt: str
    user_id: str

# GET /health
@app.get("/health")
def health_check():
    return {"message": "OK"}

# POST /query
@app.post("/query")
def handle_prompt(request: PromptRequest):
    user_query = request.prompt
    user_id = request.user_id

    return run_agent(user_query)




