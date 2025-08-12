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

load_dotenv()
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
    result: dict | None  # Add this new key for custom JSON results

# Classify user query
class MessageClassifier(BaseModel):
    message_type: Literal["default_llm_response", "song_rec", "get_concerts", "get_weather"] = Field(
        description="The type of message the user is sending")


llm = init_chat_model(
    model="gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def classify_user_query(state: State) -> State:
    message = state["messages"][-1]
    # invoke LLM that only returns a structured output
    classifier_llm = llm.with_structured_output(MessageClassifier)
    
    # Get the structured output
    result = classifier_llm.invoke([
        SystemMessage(content="You are a message classifier. Analyze the user's message and classify it into one of the specified types."),
        HumanMessage(content=message.content)
    ])
    
    # Update the state with the classification result
    return {
        "messages": state["messages"],
        "message_type": result.message_type,
        "result": {"classification": result.message_type}
    }


def get_LLM_response(state: State) -> State:
    query = state["messages"][-1].content

    try:
        response = llm.invoke([
            SystemMessage(content="You are a SoCal ABG. INCLUDE ALL DETAILS OF THE RESPONSE"),
            HumanMessage(content=query)
        ])
    except Exception as e:
        print(f"Error getting LLM response: {str(e)}")
        return None
    return {
        "messages": state["messages"],
        "message_type": state.get("message_type"),
        "result": {"llm_response": response}
    }


def structure_user_query(state: State) -> State:
    return {
        "messages": state["messages"],
        "message_type": None,
        "result": None
    }


# embedd user query
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
def search_songs(state: State) -> State:
    """Search for songs using vector similarity."""
    query = state["messages"][-1].content
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
        
        top_k = 5
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format the results properly
        if results.matches:
            song_info = results.matches[0].metadata
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {
                    "song_recommendation": song_info,
                    "query": query,
                    "top_k": top_k
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
def ticketmaster_search_event(state: State) -> State:
    """Find the best EDM, House, Electronic, Techno, Trance, and Dubstep events with the most well known artists"""
    query = state["messages"][-1].content
    # Get the flexible parameters
    genres = ["EDM", "House"]
    
    # Your code to call TicketMaster API and return event info
    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    edm_keywords = ['EDM', 'house',]

    optimal_edm_params = {
        'apikey': os.getenv("TICKETMASTER_API_KEY"),
        "dmaId": "324,381,382,374,385",
        "preferredCountry": ["US"],
        "locale": "*",
        "sort": "relevance,desc",
        "size": "5",
        "classificationName": genres,
    }

    try:
        response = requests.get(url, params=optimal_edm_params)
        
        # Parse the JSON response
        response_data = response.json()
        
        # Debug: Print response structure
        print(f"Response status: {response.status_code}")
        print(f"Response keys: {list(response_data.keys())}")
        
        # Check if the response contains events
        if '_embedded' not in response_data or 'events' not in response_data['_embedded']:
            print("No events found in response")
            print(f"Available keys in response: {list(response_data.keys())}")
            if '_embedded' in response_data:
                print(f"Available keys in _embedded: {list(response_data['_embedded'].keys())}")
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {"events": [], "error": "No events found"}
            }
        
        filtered_events = []
        for event in response_data['_embedded']['events']:
            # Get the first image URL if available
            image_url = None
            if event.get("images") and len(event.get("images")) > 0:
                image_url = event["images"][0].get("url")
            
            # Get the start date if available
            start_date = None
            if event.get("dates") and event["dates"].get("start"):
                start_date = event["dates"]["start"].get("localDate")
            
            # Get promoter name if available
            promoter_name = None
            if event.get("promoters") and len(event.get("promoters")) > 0:
                promoter_name = event["promoters"][0].get("name")
            
            # Get price range if available
            price_min = None
            if event.get("priceRanges") and len(event.get("priceRanges")) > 0:
                price_min = event["priceRanges"][0].get("min")
            
            filtered_event = {
                "name": event.get("name"),
                "url": event.get("url"),
                "image": image_url,
                "date": start_date,
                "promoter": promoter_name,
                "priceRange": price_min,
            }
            filtered_events.append(filtered_event)
        
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "events": filtered_events,
                "query": query,
                "genres": genres
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
def yelp_search_food(state: State) -> State:
    """Find the best Pocha, Korean BBQ, Hotpot, Matcha, Boba, Karaoke, and other food options in users location"""
    query = state["messages"][-1].content
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {
        "Authorization": f"Bearer {os.getenv('YELP_API_KEY')}"
    }
    params = {
        "term": query,
        "location": "Los Angeles, CA",
        "radius": 10000,
        "limit": 5,
        "sort_by": "rating",
        "categories": "korean,Viet,pocha,bbq,hotpot,matcha,boba,karaoke"
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "food_results": response.json(),
                "query": query,
                "location": "Los Angeles, CA"
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
def get_weather(state: State) -> State:
    """Get the weather for the user """
    query = state["messages"][-1].content
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": "37.7749",
        "long": "-122.4194",
        "q": query,
        "appid": os.getenv("OPENWEATHERMAP_API_KEY"),
        "units": "imperial"
    }
    try:
        response = requests.get(url, params=params)
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "weather_data": response.json(),
                "query": query,
                "location": {"lat": "37.7749", "long": "-122.4194"}
            }
        }
    except Exception as e:
        print(f"Error getting weather: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {"error": f"Error getting weather: {str(e)}"}
        }


# Create graph
graph = StateGraph(State)
graph.add_node("structure_user_query", structure_user_query)
graph.add_node("classify_user_query", classify_user_query)
graph.add_node("song_rec", search_songs)
graph.add_node("get_concerts", ticketmaster_search_event)
graph.add_node("get_weather", get_weather)
graph.add_node("default_llm_response", get_LLM_response)

graph.add_edge(START, "structure_user_query")
graph.add_edge("structure_user_query", "classify_user_query")
graph.add_conditional_edges("classify_user_query", 
    lambda state: state.get("message_type", "default_llm_response"),
    {
        "song_rec": "song_rec",
        "get_concerts": "get_concerts",
        "get_weather": "get_weather",
        "default_llm_response": "default_llm_response"
    }
)
graph.add_edge("song_rec", END)
graph.add_edge("get_concerts", END)
graph.add_edge("get_weather", END)
graph.add_edge("default_llm_response", END)

# compile graph
compiled_graph = graph.compile()