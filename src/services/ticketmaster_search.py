from models.state import State
import logging
import os
import requests
from sentence_transformers import SentenceTransformer
import faiss
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
    
    # model to use for embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # get description of genres
    descriptions = list(festival_to_description.keys())

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
    # get closest description
    closest_description = descriptions[indices[0][0]]

    # get description of closest description
    festival = festival_to_description[closest_description]
    
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

