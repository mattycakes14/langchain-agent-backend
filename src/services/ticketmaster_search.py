from models.state import State
import logging
import os
import requests
# from sentence_transformers import SentenceTransformer
# import faiss
from dotenv import load_dotenv
from static_content.concert_filters import festival_to_description

# load environment variables
load_dotenv()
# Configure logging with more detail
logging.basicConfig(level=logging.INFO)
# search for events using TicketMaster API
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the event recommendation
def ticketmaster_search_event(state: State) -> State:
    """Find the best EDM, House, Electronic, Techno, Trance, and Dubstep events with the most well known artists"""

    logging.info("[SEARCHING FOR EVENTS] Searching for events")
    query = state["messages"][-1].content
    
    # Simple keyword matching for concert genres
    def simple_genre_match(query_text: str) -> str:
        query_lower = query_text.lower()
        
        # EDM/Electronic music keywords
        genre_mappings = {
            "edm": "Electronic",
            "house": "House", 
            "techno": "Techno",
            "trance": "Trance",
            "dubstep": "Dubstep",
            "electronic": "Electronic",
            "rave": "Electronic",
            "festival": "Electronic",
            "dance": "Electronic",
            "dj": "Electronic",
            "illenium": "Electronic",
            "zedd": "Electronic", 
            "blackpink": "Pop",
            "kpop": "Pop",
            "concert": "Electronic"  # Default to electronic for ABG vibe
        }
        
        # Find best keyword match
        for keyword, genre in genre_mappings.items():
            if keyword in query_lower:
                return genre
                
        # Default to Electronic for ABG persona
        return "Electronic"
    
    festival = simple_genre_match(query)
    
    logging.info("[BEST MATCH FESTIVAL] " + str(festival))

    # Your code to call TicketMaster API and return event info
    url = "https://app.ticketmaster.com/discovery/v2/events.json"

    # Query parameters
    #apikey: TicketMaster API key (String)
    #preferredCountry: Country code (List of Strings)
    #sort: Sort by relevance and descending order (String)
    #size: Number of events to return (String)
    #classificationName: Genres to search for (List of Strings)
    optimal_edm_params = {
        'apikey': os.getenv("TICKETMASTER_API_KEY"), # REQUIRED
        "sort": "relevance,desc",
        "size": "5",
        "keyword": festival,
        "radius": "25"
    }

    try:
        response = requests.get(url, params=optimal_edm_params)
        
        
        # Parse the JSON response
        response_data = response.json()

        logging.info("[RESPONSE]: " + str(response.json()))

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

