from langchain.agents import Tool
import openai
import os
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.router.llm_router import LLMRouterChain
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from typing import Optional

load_dotenv()
# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to the index
index = pc.Index("socalabg")

# Handle flexible parameters for ticketmaster api
# class ticketMasterSearchInput(BaseModel):
#     intent: Optional[str] = Field(description="The intent of the user's query")
#     genre: Optional[str] = Field(description="The genre of music the user is looking for")
#     location: Optional[str] = Field(description="The location of the user")
#     date: Optional[str] = Field(description="The date of the event")
#     price: Optional[str] = Field(description="The price of the event")
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
def search_songs(query: str, top_k: int = 5):
    """Search for songs using vector similarity."""
    try:
        # Get embedding for the query
        print(f"Getting embedding for query: '{query}'")
        query_embedding = get_embedding(query)
        
        if query_embedding is None:
            print("Failed to get embedding for query")
            return
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results["matches"][0].metadata
            
    except Exception as e:
        print(f"Error searching songs: {str(e)}")

# recommend a song based on the user's query
def recommend_song(query: str) -> str:
    return search_songs(query)

# search for events using TicketMaster API
def ticketmaster_search_event(query: str) -> str:

    # Get the flexible parameters
    genres = ["EDM", "House", "Techno", "Trance", "Dubstep", "Festival", "Rave", "KPop", "JPop", "Korean", "R&B"]
    
    # Your code to call TicketMaster API and return event info
    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    edm_keywords = ['EDM', 'house', 'techno', 'trance', 'dubstep', 'festival', 'rave']

    optimal_edm_params = {
        'apikey': os.getenv("TICKETMASTER_API_KEY"),
        "dmaId": "324,381,382,374,385",
        "preferredCountry": ["US"],
        "preferredState": ["CA"],
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
            return []
        
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
        return filtered_events

    except Exception as e:
        print(f"Error searching events: {str(e)}")
        return None

# search for food using Yelp API (TBD)
def yelp_search_food(query: str) -> str:
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
        return response.json()
    except Exception as e:
        print(f"Error searching food: {str(e)}")
        return None

# get weather using OpenWeatherMap API
def get_weather(query: str) -> str:
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
        return response.json()
    except Exception as e:
        print(f"Error getting weather: {str(e)}")
        return None
# final answer tool (fallback)
def final_answer(query: str) -> str:
    return "Output whatever you have as a thought or observation"
    
# Wrap functions as LangChain Tools
song_tool = Tool(
    name="getSongRecommendation",
    func=recommend_song,
    description="Provide a song recommendation"
)

ticketmaster_tool = Tool(
    name="TicketMasterSearch",
    func=ticketmaster_search_event,
    description="Find the best EDM, House, Electronic, Techno, Trance, and Dubstep events with the most well known artists",
)

yelp_tool = Tool(
    name="FoodRecommendation",
    func=yelp_search_food,
    description="Find the best Pocha, Korean BBQ, Hotpot, Matcha, Boba, Karaoke, and other food options in users location"
)

get_weather_tool = Tool(
    name="GetWeather",
    func=get_weather,
    description="Get the weather for the user (In Farenheit) "
)

final_answer_tool = Tool(
    name="FinalAnswer",
    func=final_answer,
    description="Provide a final answer to the user's query"
)
