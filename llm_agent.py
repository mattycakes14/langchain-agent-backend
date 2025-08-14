from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests
from langchain_core.messages import HumanMessage, SystemMessage
import logging
from langgraph.graph import StateGraph, START, END
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
import openai
from pinecone import Pinecone
from typing import Optional, List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from descriptions_to_aliases import descriptions_to_aliases
from tavily import TavilyClient


# Configure logging with more detail
logging.basicConfig(level=logging.INFO)

# Create a logger for your specific module

load_dotenv()

# configure tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# openrouter api key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# openrouter base url
BASE = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to the index
index = pc.Index("socalabg")

# State structure for each node
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    result: dict | None
    extracted_params: dict | None  # Add this back
    search_results: dict | None

# Classify user query
class MessageClassifier(BaseModel):
    message_type: Literal["default_llm_response", "song_rec", "get_concerts", "get_weather", "yelp_search_food"] = Field(
        description="The type of message the user is sending")


llm_main = init_chat_model(
    model="gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

llm_fast = init_chat_model(
    model="gpt-4.1-nano",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Define structured output for parameter extraction (API call query parameters)
class ExtractedParams(BaseModel):
    lat: Optional[float] = Field(description="Extracted latitude from the query", default=None)
    lon: Optional[float] = Field(description="Extracted longitude from the query", default=None)
    genres: Optional[List[str]] = Field(description="Extracted genres from the query", default=None)


# Extract paramters from user query in a format to be used in API calls
# Parameters:
# query: str - The user query
# message_type: str - The type of message the user is sending
# Returns:
# dict - The extracted parameters
def extract_parameters_llm(query: str, message_type: str) -> dict:
    """Extract parameters using LLM for better accuracy."""
    try:
        # Create structured output model
        param_extractor = llm_fast.with_structured_output(ExtractedParams)
        system_prompt = ""

        # Create context-aware prompt based on message type
        if message_type == "get_weather":
            system_prompt = """Set the longitude and latitude from the user query (i.e. "weather in San Diego" -> lat: 32.7157, lon: -117.1611)"""
        elif message_type == "yelp_search_food":
            system_prompt = """Extract the longitude and latitude from the user query (i.e. "food in San Diego" -> lat: 32.7157, lon: -117.1611)"""
        
        elif message_type == "song_rec":
            system_prompt = """Extract user query song details (i.e. "Looking for melodic, upbeat rave songs" -> genres: [melodic, upbeat, rave])"""
        
        elif message_type == "get_concerts":
            system_prompt = """Set genres and lat & long from the user query. If user doesn't specify a start date and time, set it to the current date and time.
            (i.e. "EDM, House, Trance concerts in LA" -> genres: [EDM, House, Trance], lat: 34.0522, lon: -118.2437)"""
        
        elif message_type == "default_llm_response":
            system_prompt = """Extract any relevant parameters that might be useful for context."""
        
        else:
            system_prompt = """Extract any relevant parameters from the user query."""
        
        # Get structured extraction
        result = param_extractor.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract parameters from this query: {query}")
        ])
        
        return result.dict()
        
        
    except Exception as e:
        print(f"Error extracting parameters with LLM: {str(e)}")
        return {}

def classify_user_query(state: State) -> State:
    message = state["messages"][0]

    # invoke LLM that only returns a structured output
    classifier_llm = llm_fast.with_structured_output(MessageClassifier)
    
    # Get the structured output
    result = classifier_llm.invoke([
        SystemMessage(content="You are a message classifier. Analyze the user's message and classify it into one of the specified types."),
        HumanMessage(content=message.content)
    ])
    logging.info(f"[CLASSIFYING MESSAGE] Classified message: {result.message_type}")

    # Extract parameters using LLM
    extracted_params = extract_parameters_llm(message, result.message_type)
    
    logging.info(f"[EXTRACTING PARAMETERS] Extracted parameters: {extracted_params}")
    # Update the state with the classification result and extracted parameters
    return {
        "messages": state["messages"],
        "message_type": result.message_type,
        "extracted_params": extracted_params
    }


def get_LLM_response(state: State) -> State:
    query = state["messages"][-1].content
    search_results = state.get("search_results", {})

    # retrieve content from previous nodes
    song_rec = state.get("result", {}).get("song_recommendation", {})
    concerts = state.get("result", {}).get("events", {})
    weather = state.get("result", {}).get("weather_data", {})
    food = state.get("result", {}).get("food_results", {})

    output = None
    if song_rec:
        output = song_rec
    elif food:
        output = food
    elif concerts:
        output = concerts
    elif weather:
        output = weather
    else:
        output = query

    logging.info("[INTEGRATING RESULT] output: " + str(output))
    try:
        response = llm_main.invoke([
            SystemMessage(content="You're personality is a SoCal ABG. Output this result in a SoCal ABG way: " + str(output) + " and also include the search results as a list of links: " + str(search_results)),
        ])
    except Exception as e:
        print(f"Error getting LLM response: {str(e)}")
        return None
    return {
        "messages": state["messages"],
        "message_type": state.get("message_type"),
        "result": {"llm_response": response}
    }


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

# search for songs using vector similarity
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the song recommendation
def search_songs(state: State) -> State:
    """Search for songs using vector similarity."""
    query = state["messages"][0].content
    try:
        # Get embedding for the query
        print(f"Getting embedding for query: '{query}'")
        query_embedding = get_embedding(query)
        
        if query_embedding is None:
            print("Failed to get embedding for query")
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {"error": "Failed to get embedding for query"}
            }
        results = index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True
        )
        
        # Format the results properly
        if results.matches:
            song_info = results.matches[0].metadata
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {
                    "song_recommendation": song_info
                }
            }
        else:
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {"error": "No songs found matching your request"}
            }
            
    except Exception as e:
        print(f"Error searching songs: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {"error": f"Error searching songs: {str(e)}"}
        }


# search for events using TicketMaster API
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the event recommendation
def ticketmaster_search_event(state: State) -> State:
    """Find the best EDM, House, Electronic, Techno, Trance, and Dubstep events with the most well known artists"""
    # genres = state.get("extracted_params", {}).get("genres", [])
    # postal_code = state.get("extracted_params", {}).get("postal_code", "90001")
    logging.info("[SEARCHING FOR EVENTS] Searching for events")
    # Get the flexible parameters
    default_genres = ["EDM", "House", "Techno", "Trance", "Dubstep"]
    
    # Your code to call TicketMaster API and return event info
    url = "https://app.ticketmaster.com/discovery/v2/events.json"

    #Query parameters
    #apikey: TicketMaster API key (String)
    #preferredCountry: Country code (List of Strings)
    #sort: Sort by relevance and descending order (String)
    #size: Number of events to return (String)
    #classificationName: Genres to search for (List of Strings)
    optimal_edm_params = {
        'apikey': os.getenv("TICKETMASTER_API_KEY"), # REQUIRED
        "sort": "relevance,desc",
        "size": "5",
        "classificationName": state.get("extracted_params", {}).get("genres", default_genres),
        "latlong": str(state.get("extracted_params", {}).get("lat", 34.0522)) + "," + str(state.get("extracted_params", {}).get("lon", -118.2437)),
        "radius": "25"
    }

    try:
        response = requests.get(url, params=optimal_edm_params)
        
        # Parse the JSON response
        response_data = response.json()

        # extract relevant event information from response
        filtered_events = []
        for events in response_data["_embedded"]["events"]:
            name = events["name"]
            url = events["url"]
            image_url = events["images"][0]["url"]
            
            filtered_events.append({
                "name": name,
                "url": url,
                "image_url": image_url
            })

        logging.info("[SEARCHING FOR EVENTS] Found events: " + str(filtered_events))
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "extracted_params": state.get("extracted_params"),
            "result": {
                "events": filtered_events
            }
        }
    except Exception as e:
        print(f"Error searching events: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {"error": f"Error searching events: {str(e)}"}
        }

# search for food using Yelp API (TBD)
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the food recommendation
def yelp_search_food(state: State) -> State:
    """Find the best Pocha, Korean BBQ, Hotpot, Matcha, Boba, Karaoke, and other food options in users location"""
    query = state["messages"][-1].content
    
    # model to use for embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # list of descriptions
    descriptions = list(descriptions_to_aliases.keys())
    # embed descriptions
    descriptions_embeddings = model.encode(descriptions)
    # create faiss index
    index = faiss.IndexFlatL2(descriptions_embeddings.shape[1])
    index.add(descriptions_embeddings)
    
    # get embedding for query
    query_embedding = model.encode(query)
    # reshape to 2D array for FAISS search
    query_embedding = query_embedding.reshape(1, -1)
    # search for nearest neighbors
    distances, indices = index.search(query_embedding, 5)
    # get descriptions of nearest neighbors
    nearest_descriptions = [descriptions[i] for i in indices[0]]
    # get aliases of nearest descriptions
    aliases = descriptions_to_aliases[nearest_descriptions[0]]

    logging.info("[SEARCHING FOR MATCHING YELP ALIASES] MATCHING YELP ALIASES: " + str(aliases))
    
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {
        "Authorization": f"Bearer {os.getenv('YELP_API_KEY')}"
    }

    # Query parameters
    # latitude: Latitude of the location (Float) REQUIRED
    # longitude: Longitude of the location (Float) REQUIRED
    # categories: Categories to search for (List of Strings) OPTIONAL
    # radius: Radius of the search in meters (Integer) OPTIONAL
    # limit: Number of businesses to return (Integer) OPTIONAL
    # sort_by: Sort by rating (String) OPTIONAL
    params = {
        "latitude": state.get("extracted_params", {}).get("lat", 34.0522),
        "longitude": state.get("extracted_params", {}).get("lon", -118.2437),
        "categories": aliases,
        "radius": 20000,
        "limit": 5,
        "sort_by": "best_match"
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        filtered_results = []

        # extract relevant information from response
        for result in response.json()["businesses"]:
            name = result["name"]
            url = result["url"]
            image_url = result["image_url"]
            rating = result["rating"]
            phone = result["phone"]
            address = result["location"]["address1"]
            filtered_results.append({
                "name": name,
                "url": url,
                "image_url": image_url,
                "rating": rating,
                "phone": phone,
                "address": address
            })
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "food_results": filtered_results,
            }
        }
    except Exception as e:
        print(f"Error searching food: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {"error": f"Error searching food: {str(e)}"}
        }

# get weather using OpenWeatherMap API
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the weather recommendation
def get_weather(state: State) -> State:
    """Get the weather for the user using extracted location parameters"""
    query = state["messages"][-1].content
    extracted_params = state.get("extracted_params", {})
    

    #Query parameters
    #lat: Latitude of the location (Float) REQUIRED
    #lon: Longitude of the location (Float) REQUIRED
    #appid: OpenWeatherMap API key (String) REQUIRED
    #units: Units to use for temperature (String) OPTIONAL
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": extracted_params.get("lat", 34.0522),
        "lon": extracted_params.get("lon", -118.2437),
        "appid": os.getenv("OPENWEATHERMAP_API_KEY"),
        "units": "imperial"
    }
    
    try:
        response = requests.get(url, params=params)
        weather_data = response.json()
        
        # Check if the API call was successful
        if weather_data.get("cod") == 200:
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {
                    "weather_data": weather_data,
                },
                "extracted_params": extracted_params
            }
        else:
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {
                    "weather_data": "No weather data found"
                },
                "extracted_params": extracted_params
            }
            
    except Exception as e:
        print(f"Error getting weather: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {"error": f"Error getting weather: {str(e)}"},
            "extracted_params": extracted_params
        }

# search engine fallback for user query
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the search engine results
def search_web(state: State) -> State:
    """Search the web for the user query"""
    logging.info("[SEARCHING THE WEB] Searching the web for the user query")
    query = state["messages"][-1].content
    results = tavily_client.search(query, max_results=5)

    logging.info("[SEARCHING THE WEB] Search results: " + str(results))
    return {
        "search_results": results
    }

# Create graph
graph = StateGraph(State)

# create nodes
graph.add_node("classify_user_query", classify_user_query)
graph.add_node("song_rec", search_songs)
graph.add_node("get_concerts", ticketmaster_search_event)
graph.add_node("get_weather", get_weather)
graph.add_node("default_llm_response", get_LLM_response)
graph.add_node("yelp_search_food", yelp_search_food)
graph.add_node("search_web", search_web)

# add edges
graph.add_edge(START, "classify_user_query")
graph.add_conditional_edges("classify_user_query", 
    lambda state: state.get("message_type", "default_llm_response"),
    {
        "song_rec": "song_rec",
        "get_concerts": "get_concerts",
        "get_weather": "get_weather",
        "yelp_search_food": "yelp_search_food",
        "default_llm_response": "default_llm_response"
    }
)
graph.add_edge("classify_user_query", "search_web")
graph.add_edge("search_web", "default_llm_response")
graph.add_edge("song_rec", "default_llm_response")
graph.add_edge("get_concerts", "default_llm_response")
graph.add_edge("get_weather", "default_llm_response")
graph.add_edge("yelp_search_food", "default_llm_response")
graph.add_edge("default_llm_response", END)



# compile graph
compiled_graph = graph.compile()