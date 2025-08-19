from models.state import State
import logging
import os
import requests
from dotenv import load_dotenv

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
    extracted_params = state.get("extracted_params", {})
    
    # Yelp API endpoint
    url = "https://api.yelp.com/v3/businesses/search"
    
    # Headers with API key
    headers = {
        "Authorization": f"Bearer {os.getenv('YELP_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    # Query parameters
    params = {
        "term": query,
        "latitude": extracted_params.get("lat", 34.0522),
        "longitude": extracted_params.get("lon", -118.2437),
        "radius": 5000,  # 5km radius
        "limit": 10,
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
            "extracted_params": extracted_params
        }
        
    except Exception as e:
        logging.error(f"Yelp search error: {e}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "error": f"Yelp search failed: {str(e)}"
            },
            "extracted_params": extracted_params
        }
