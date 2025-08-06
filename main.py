from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import requests
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
    
    print(f"User query: {user_query}")
    print(f"User ID: {user_id}")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": user_query}
        ]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    return response.json()




