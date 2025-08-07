from langchain.agents import Tool
import openai
import os
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests

load_dotenv()
# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to the index
index = pc.Index("socalabg")


def ticketmaster_search_event(query: str) -> str:
    # Your code to call TicketMaster API and return event info
    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    edm_keywords = ['EDM', 'house', 'techno', 'trance', 'dubstep', 'festival', 'rave']

    edm_venues =['Insomniac ']

    optimal_edm_params = {
        'apikey': os.getenv("TICKETMASTER_API_KEY"),
        "dmaId": "324,381,382,374,385",
        "preferredCountry": ["US"],
        "locale": "*",
        "sort": "relevance,desc",
        "size": "5",
        "classificationName": ["EDM", "House", "Techno", "Trance", "Dubstep", "Festival", "Rave"],
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

res = ticketmaster_search_event("EDM")
print(res)