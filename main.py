from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import requests
from llm_agent import compiled_graph
from langchain_core.messages import HumanMessage
import logging
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

    result = compiled_graph.invoke({"messages": [HumanMessage(content=user_query)]})
    logging.info("[FINAL RESULT]: " + str(result))
    return result




