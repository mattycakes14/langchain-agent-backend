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

# POST /auth/userintegrations
@app.post("/auth/userintegrations")
def handle_auth(request: dict):
    """
    Check the status of all user integrations
    """
    user_id = request.get("user_id")
    
    if not user_id:
        return {"error": "Missing user_id", "status": "error"}
    
    try:
        # fetch the user integrations
        user_integrations = supabase_client.table("user_integrations").select("*").eq("user_id", user_id).execute()
        logging.info(f"[AUTH] User integrations for {user_id}: {user_integrations.data}")

        pending_services = []
        hasPendingStatus = False
        for integration in user_integrations.data:
            if integration.get("status") == "pending":
                hasPendingStatus = True
                pending_services.append(integration.get("service_name"))
        
        if hasPendingStatus:
            return {"status": "pending", "pending_services": pending_services}
        else:
            return {"status": "completed"}
        
    except Exception as e:
        logging.error(f"[AUTH] Error fetching user integrations: {str(e)}")
        return {"error": "Error fetching user integrations", "status": "error"}

@app.post("/auth/userintegrations/spotify")
def handle_spotify_auth(request: dict):
    user_id = request.get("user_id")
    
    if not user_id:
        return {"error": "Missing user_id", "status": "error"}
    
    try:
        # Check if integration already exists
        existing = supabase_client.table("user_integrations").select("*").eq("user_id", user_id).eq("service_name", "spotify").execute()
        
        if existing.data and existing.data[0].get("status") == "completed":
            return {"status": "already_authenticated", "message": "Spotify already connected"}
        
        # Start auth flow
        auth_response = client.tools.authorize(
            tool_name="Spotify.PlayTrackByName",
            user_id=user_id,
        )

        integration_data = {
            "user_id": user_id,
            "service_name": "spotify",
            "status": "url_sent",
            "auth_id": auth_response.id
        }

        if existing.data:
            # Update existing record
            supabase_client.table("user_integrations").update(integration_data).eq("user_id", user_id).eq("service_name", "spotify").execute()
        else:
            # Create new record
            supabase_client.table("user_integrations").insert(integration_data).execute()
        
        return {
            "status": "url_sent",
            "auth_url": auth_response.url,
            "auth_id": auth_response.id
        }
        
    except Exception as e:
        logging.error(f"[SPOTIFY AUTH] Error: {str(e)}")
        return {"error": f"Failed to start Spotify auth: {str(e)}", "status": "error"}

@app.post("/auth/userintegrations/spotify/callback")
def handle_spotify_callback(request: dict):
    auth_id = request.get("auth_id")
    user_id = request.get("user_id")
    
    if not auth_id or not user_id:
        return {"error": "Missing auth_id or user_id", "status": "error"}
    
    try:
        # Wait for auth completion
        auth_response = client.auth.wait_for_completion(auth_id)
        
        if auth_response.status == "completed":
            # Update integration with completed status and token
            supabase_client.table("user_integrations").update({
                "status": "completed",
                "access_token": auth_response.context.token,
                "auth_id": None  # Clear auth_id since completed
            }).eq("user_id", user_id).eq("service_name", "spotify").execute()
            
            return {"status": "completed", "message": "Spotify connected successfully"}
        else:
            # Update status to failed
            supabase_client.table("user_integrations").update({
                "status": "failed",
                "auth_id": None
            }).eq("user_id", user_id).eq("service_name", "spotify").execute()
            
            return {"status": "failed", "error": "Authentication failed"}
            
    except Exception as e:
        logging.error(f"[SPOTIFY CALLBACK] Error: {str(e)}")
        return {"error": f"Failed to complete Spotify auth: {str(e)}", "status": "error"}

@app.post("/auth/userintegrations/googlecalendar")
def handle_google_calendar_auth(request: dict):
    user_id = request.get("user_id")
    
    if not user_id:
        return {"error": "Missing user_id", "status": "error"}
    
    try:
        # Check if integration already exists
        existing = supabase_client.table("user_integrations").select("*").eq("user_id", user_id).eq("service_name", "google_calendar").execute()
        
        if existing.data and existing.data[0].get("status") == "completed":
            return {"status": "already_authenticated", "message": "Google Calendar already connected"}
        
        # Start auth flow
        auth_response = client.tools.authorize(
            tool_name="GoogleCalendar.CreateEvent",
            user_id=user_id,
        )
        
        # Create or update integration record
        integration_data = {
            "user_id": user_id,
            "service_name": "google_calendar",
            "status": "url_sent",
            "auth_id": auth_response.id
        }
        
        if existing.data:
            supabase_client.table("user_integrations").update(integration_data).eq("user_id", user_id).eq("service_name", "google_calendar").execute()
        else:
            supabase_client.table("user_integrations").insert(integration_data).execute()
        
        return {
            "status": "url_sent",
            "auth_url": auth_response.url,
            "auth_id": auth_response.id
        }
        
    except Exception as e:
        logging.error(f"[GOOGLE CALENDAR AUTH] Error: {str(e)}")
        return {"error": f"Failed to start Google Calendar auth: {str(e)}", "status": "error"}

@app.post("/auth/userintegrations/googledocs")
def handle_google_docs_auth(request: dict):
    user_id = request.get("user_id")
    
    if not user_id:
        return {"error": "Missing user_id", "status": "error"}
    
    try:
        # Check if integration already exists
        existing = supabase_client.table("user_integrations").select("*").eq("user_id", user_id).eq("service_name", "google_docs").execute()
        
        if existing.data and existing.data[0].get("status") == "completed":
            return {"status": "already_authenticated", "message": "Google Docs already connected"}
        
        # Start auth flow
        auth_response = client.tools.authorize(
            tool_name="GoogleDocs.CreateDocumentFromText",
            user_id=user_id,
        )
        
        # Create or update integration record
        integration_data = {
            "user_id": user_id,
            "service_name": "google_docs",
            "status": "url_sent",
            "auth_id": auth_response.id
        }
        
        if existing.data:
            supabase_client.table("user_integrations").update(integration_data).eq("user_id", user_id).eq("service_name", "google_docs").execute()
        else:
            supabase_client.table("user_integrations").insert(integration_data).execute()
        
        return {
            "status": "url_sent",
            "auth_url": auth_response.url,
            "auth_id": auth_response.id
        }
        
    except Exception as e:
        logging.error(f"[GOOGLE DOCS AUTH] Error: {str(e)}")
        return {"error": f"Failed to start Google Docs auth: {str(e)}", "status": "error"}

@app.post("/auth/userintegrations/googlecalendar/callback")
def handle_google_calendar_callback(request: dict):
    auth_id = request.get("auth_id")
    user_id = request.get("user_id")
    
    if not auth_id or not user_id:
        return {"error": "Missing auth_id or user_id", "status": "error"}
    
    try:
        # Wait for auth completion
        auth_response = client.auth.wait_for_completion(auth_id)
        
        if auth_response.status == "completed":
            # Update integration with completed status and token
            supabase_client.table("user_integrations").update({
                "status": "completed",
                "access_token": auth_response.context.token,
                "auth_id": None
            }).eq("user_id", user_id).eq("service_name", "google_calendar").execute()
            
            return {"status": "completed", "message": "Google Calendar connected successfully"}
        else:
            # Update status to failed
            supabase_client.table("user_integrations").update({
                "status": "failed",
                "auth_id": None
            }).eq("user_id", user_id).eq("service_name", "google_calendar").execute()
            
            return {"status": "failed", "error": "Authentication failed"}
            
    except Exception as e:
        logging.error(f"[GOOGLE CALENDAR CALLBACK] Error: {str(e)}")
        return {"error": f"Failed to complete Google Calendar auth: {str(e)}", "status": "error"}

@app.post("/auth/userintegrations/googledocs/callback")
def handle_google_docs_callback(request: dict):
    auth_id = request.get("auth_id")
    user_id = request.get("user_id")
    
    if not auth_id or not user_id:
        return {"error": "Missing auth_id or user_id", "status": "error"}
    
    try:
        # Wait for auth completion
        auth_response = client.auth.wait_for_completion(auth_id)
        
        if auth_response.status == "completed":
            # Update integration with completed status and token
            supabase_client.table("user_integrations").update({
                "status": "completed",
                "access_token": auth_response.context.token,
                "auth_id": None
            }).eq("user_id", user_id).eq("service_name", "google_docs").execute()
            
            return {"status": "completed", "message": "Google Docs connected successfully"}
        else:
            # Update status to failed
            supabase_client.table("user_integrations").update({
                "status": "failed",
                "auth_id": None
            }).eq("user_id", user_id).eq("service_name", "google_docs").execute()
            
            return {"status": "failed", "error": "Authentication failed"}
            
    except Exception as e:
        logging.error(f"[GOOGLE DOCS CALLBACK] Error: {str(e)}")
        return {"error": f"Failed to complete Google Docs auth: {str(e)}", "status": "error"}

# POST /query
@app.post("/query")
def handle_prompt(request: PromptRequest):
    user_query = request.prompt
    user_id = request.user_id

    logging.info(f"[QUERY] Received query: {user_query}")
    result = compiled_graph.invoke({"messages": [HumanMessage(content=user_query)]})
    logging.info("[FINAL RESULT]: " + str(result))
    return result

