
from langchain_core.messages import SystemMessage, HumanMessage
from models.state import ExtractedParams, State
from config.settings import llm_fast
import logging
# Configure logging with more detail
logging.basicConfig(level=logging.INFO)

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
        elif message_type == "yelp_search_activities":
            system_prompt = """Extract the longitude and latitude from the user query (i.e. "food in San Diego" -> lat: 32.7157, lon: -117.1611)"""
        
        elif message_type == "song_rec":
            system_prompt = """Extract user query song details (i.e. "Looking for melodic, upbeat rave songs" -> genres: [melodic, upbeat, rave])"""
        
        elif message_type == "get_concerts":
            system_prompt = """Set lat & long from the user query. If user doesn't specify a start date and time, set it to the current date and time.
            (i.e. "What are some EDM concerts in LA?" -> lat: 34.0522, lon: -118.2437)"""
        
        elif message_type == "default_llm_response":
            system_prompt = """Extract any relevant parameters that might be useful for context."""
        
        elif message_type == "yelp_search_activities":
            system_prompt = """Extract the longitude and latitude from the user query, (i.e. "food in San Diego" -> lat: 32.7157, lon: -117.1611)"""
        
        elif message_type == "get_google_flights":
            system_prompt = """Extract the departure and arrival airport codes from the user query, (i.e. "flights from LA to SF" -> departure_airport_code: LAX, arrival_airport_code: SFO)"""
        
        elif message_type == "get_google_hotels":
            system_prompt = """Extract the location, check-in date, and check-out date from the user query, (i.e. "hotel in San Diego" -> location: San Diego, check_in_date: 2025-08-19, check_out_date: 2025-08-20)"""
        
        elif message_type == "write_to_google_docs":
            system_prompt = """Extract the title and text content from the user query, (i.e. "write a document about my trip to San Diego" -> title: My Trip to San Diego, text_content: I had a great time in San Diego!)"""
        
        elif message_type == "search_reddit_forums":
            system_prompt = """Extract the subreddit from the user query, (i.e. "search reddit for the best restaurants in San Diego" -> subreddit: r/SanDiego)"""
        
        else:
            system_prompt = """Extract any relevant parameters from the user query."""
        
        # Get structured extraction
        result = param_extractor.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract parameters from this query: {query}")
        ])
        
        return result.model_dump()
        
        
    except Exception as e:
        print(f"Error extracting parameters with LLM: {str(e)}")
        return {}
