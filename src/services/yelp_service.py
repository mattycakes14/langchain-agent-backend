from models.state import State, LocationState
import logging
import os
import requests
from dotenv import load_dotenv
from config.settings import llm_fast
from static_content.descriptions_to_aliases import descriptions_to_aliases
# from sentence_transformers import SentenceTransformer
# import faiss
from langchain_core.messages import SystemMessage

# load environment variables
load_dotenv()
# Configure logging with more detail
logging.basicConfig(level=logging.INFO)

# search for activities using Yelp API
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the yelp search results
def yelp_search_activities(state: State) -> State:
    """Search for activities using Yelp API"""
    query = state["messages"][-1].content

    llm_params = llm_fast.with_structured_output(LocationState)
    prompt = f"""
        Decide the longitude and latitude of the location the user wants to search for activities. The user's message is: {str(state["messages"][-1].content)}
        
        LONGITUDE AND LATITUDE CAN ONLY BE NUMBERS
    """

    # Simple keyword matching instead of vector search
    def simple_category_match(query_text: str) -> str:
        query_lower = query_text.lower()
        
        # ABG favorites mapping
        keyword_mappings = {
            "boba": "bubbletea",
            "bubble tea": "bubbletea", 
            "milk tea": "bubbletea",
            "taco": "tacos",
            "mexican": "mexican",
            "thai": "thai",
            "croissant": "bakeries",
            "coffee": "coffee",
            "matcha": "bubbletea",
            "food": "restaurants",
            "eat": "restaurants",
            "restaurant": "restaurants",
            "drink": "bars",
            "bar": "bars",
            "shopping": "shopping",
            "clothes": "shopping",
            "vintage": "vintage"
        }
        
        # Find best keyword match
        for keyword, category in keyword_mappings.items():
            if keyword in query_lower:
                return category
                
        # Default fallback
        return "restaurants"
    
    closest_alias = simple_category_match(query)

    logging.info(f"[YELP SEARCH] Matched category: {closest_alias}")

    # decide the longitude and latitude of the location
    result = llm_params.invoke([SystemMessage(content=prompt)])
    result = result.model_dump()
    longitude = result.get("longitude", 0)
    latitude = result.get("latitude", 0)

    # Yelp API endpoint
    url = "https://api.yelp.com/v3/businesses/search"
    
    # Headers with API key
    headers = {
        "Authorization": f"Bearer {os.getenv('YELP_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    logging.info(f"[YELP SEARCH] Latitude: {latitude}, Longitude: {longitude}")
    
    # Query parameters
    params = {
        "categories": closest_alias,
        "latitude": latitude,
        "longitude": longitude,
        "radius": 5000,  # 5km radius
        "limit": 5,
        "sort_by": "rating"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        businesses = data.get("businesses", [])
        
        # Format results
        results = []
        for business in businesses:
            results.append({
                "name": business.get("name"),
                "rating": business.get("rating"),
                "price": business.get("price"),
                "categories": [cat.get("title") for cat in business.get("categories", [])],
                "location": business.get("location", {}).get("address1"),
                "url": business.get("url")
            })
        
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "yelp_results": results
            },
        }
        
    except Exception as e:
        logging.error(f"Yelp search error: {e}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "error": f"Yelp search failed: {str(e)}"
            },
        }
