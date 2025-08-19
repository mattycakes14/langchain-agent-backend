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
import random
from concert_filters import festival_to_description
from arcadepy import Arcade
from langchain_arcade import ArcadeToolManager
import praw
# Configure logging with more detail
logging.basicConfig(level=logging.INFO)


load_dotenv()

# reddit client
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)
# arcade client
client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))
manager = ArcadeToolManager(api_key=os.getenv("ARCADE_API_KEY"))

# Configure SerpAPI key for Google Flights toolkit
SERP_API_KEY = os.getenv("SERP_API_KEY")
if not SERP_API_KEY:
    print("Warning: SERP_API_KEY not found in environment variables. Google Flights functionality may not work.")

# get toolkits
tools = manager.get_tools(toolkits=["GoogleCalendar"])

# user id for application
user_id = "mlau191@uw.edu"

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
    message_type: Literal["default_llm_response", "song_rec", "get_concerts", "get_weather", "yelp_search_activities", "create_calendar_event", "get_google_flights", "get_google_hotels", "write_to_google_docs", "search_reddit_forums"] = Field(
        description="The type of message the user is sending")

# Calendar state model
class CalendarState(BaseModel):
    summary: str = Field(description="The summary of the event")
    description: Optional[str] = Field(description="The description of the event")
    start_datetime: str = Field(description="The start date and time of the event in ISO 8601 format")
    end_datetime: str = Field(description="The end date and time of the event in ISO 8601 format")
    location: Optional[str] = Field(description="The location of the event")

# Flight state model
class FlightState(BaseModel):
    departure_airport_code: str = Field(description="The departure airport code (UPPERCASE 3-LETTER CODE)")
    arrival_airport_code: str = Field(description="The arrival airport code (UPPERCASE 3-LETTER CODE)")
    outbound_date: str = Field(description="The outbound date of the flight in YYYY-MM-DD format")
    num_adults: int = Field(description="The number of adults on the flight")
    sort_by: str = Field(description="The sort order of the flights (TOP_FlIGHTS, PRICE, DURATION, DEPARTURE_TIME, ARRIVAL_TIME)")

# Hotel state model
class HotelState(BaseModel):
    location: str = Field(description="The location of the hotel")
    check_in_date: str = Field(description="The check-in date of the hotel in YYYY-MM-DD format")
    check_out_date: str = Field(description="The check-out date of the hotel in YYYY-MM-DD format")
    query: str = Field(description="The user query")
    min_price: int = Field(description="The minimum price of the hotel")
    max_price: int = Field(description="The maximum price of the hotel")
    num_adults: int = Field(description="The number of adults on the hotel")
    sort_by: str = Field(description="The sort order of the hotels (RELEVANCE, LOWEST_PRICE, HIGHEST_RATING, MOST_REVIEWED)")

# Spotify state model
class SpotifyState(BaseModel):
    track_name: str = Field(description="The name of the track to play")
    artist_name: str = Field(description="The name of the artist of the track to play")

#Google Docs state model
class GoogleDocsState(BaseModel):
    title: str = Field(description="The title of the document")
    text_content: str = Field(description="The text content of the document")

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
    keyword: Optional[str] = Field(description="Extracted main keyword from the query", default=None)


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

def classify_user_query(state: State) -> State:
    message = state["messages"][0]

    # invoke LLM that only returns a structured output
    classifier_llm = llm_fast.with_structured_output(MessageClassifier)
    
    # content of message
    content = """
    You are a message classifier. Analyze the user's message and classify it into one of the specified types.
    The user's message is: {message}
    The message types are:
    - song_rec: The user is asking for a song recommendation.
    - get_concerts: The user is asking for a concert recommendation.
    - get_weather: The user is asking for the weather.
    - yelp_search_activities: The user is asking for a restaurant, cafe, or other activity recommendation.
    - create_calendar_event: The user wants to create a calendar event, schedule something, or add an event to their calendar.
    - get_google_flights: The user is asking for flight information.
    - get_google_hotels: The user is asking for hotel information.
    - write_to_google_docs: The user is asking to write to a google doc.
    - search_reddit_forums: The user is asking to search the reddit forums.
    - post_to_reddit: The user is asking to post to the reddit forums.
    - default_llm_response: The user is asking a question that doesn't fit into any of the other categories.
    """

    # Get the structured output
    result = classifier_llm.invoke([
        SystemMessage(content=content),
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
    
    # extract links from search results
    if 'results' in search_results and search_results['results']:
        links = [result["url"] for result in search_results['results']]
    else:
        links = []

    logging.info("[SEARCHING THE WEB] Search results links: " + str(links))
    # retrieve content from previous nodes
    song_rec = state.get("result", {}).get("song_recommendation", {})
    concerts = state.get("result", {}).get("events", {})
    weather = state.get("result", {}).get("weather_data", {})
    food = state.get("result", {}).get("food_results", {})
    calendar_results = state.get("result", {}).get("calendar_results", {})
    auth_complete = state.get("result", {}).get("auth_complete", {})

    output = None
    if song_rec:
        output = song_rec
    elif food:
        output = food
    elif concerts:
        output = concerts
    elif weather:
        output = weather
    elif calendar_results:
        output = calendar_results
    elif auth_complete:
        output = auth_complete
    else:
        output = query

    logging.info("[INTEGRATING RESULT] output: " + str(output))
    
    system_prompt = """ You are a 21-year-old SoCal ABG bestie. Always speak in playful, slangy, emoji-filled tone (“omg babe,” “lowkey,” “no cap”). Hype your faves, roast/judge anything off-vibe, but still give helpful answers.
        Faves: Communication Hills, Irvine Spectrum; SJSU/UCI/UCR; Pre-med/Business; Boba (Tisane, Peko, Boba Guys, 7Leaves, Izumi Matcha); Food (Cali Spartan Tacos, Green Thai Tea, Seaside ham & cheese croissant); Artists (Illenium, Zedd, IsoKnock, BlackPink, Le Sserafim, Keshi, Nuevo Lio); Games (Valorant, LoL); Raves (Countdown, Escape, Decadence, Breakaway, EDC, NorthCoast); Fashion (Essentials, Stussy, Vintage, PacSun, Brandy Melville); Loves BMWs & vaping.
        Rules: If topic fits your vibe → hype it.
        If not → playful shade + answer + steer back to faves.
        Never break character.
        
        Here is the user query: """ + query + """
        Here is the output: """ + str(output) + """
        
        List the search results in a numbered list. """ + str(links) + """
        """
    try:
        response = llm_main.invoke([
            SystemMessage(content=system_prompt),
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

def spotify_play_track(state: State) -> State:
    """Play a track on Spotify"""
    song_rec = state.get("result", {}).get("song_recommendation", {})
    artists = song_rec.get("artists", "")
    title = song_rec.get("title", "")

    logging.info("[SPOTIFY PLAY TRACK] Playing track: " + title + " by " + artists)

    tool_name = "Spotify.PlayTrackByName"

    auth_response = client.auth.start(
        user_id=user_id,
        provider="spotify",
        scopes=["user-read-playback-state", "user-modify-playback-state"]
    )

    if auth_response.status != "completed":
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "error": f"Failed to authorize Spotify tool. Please authorize the tool in the browser and try again. {auth_response.url}"
            }
        }

    auth_response = client.auth.wait_for_completion(auth_response)
    
    # get access token
    access_token = auth_response.context.token

    tool_input = {
        "track_name": title,
        "artist_name": artists
    }
    try:
        response = client.tools.execute(
        tool_name=tool_name,
            input=tool_input,
            user_id=user_id,
        )

        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "spotify_response": response
            }
        }
    except Exception as e:
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {"error": f"Error playing track: {str(e)}"}
        }
    

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

# search for food using Yelp API (TBD)
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the food recommendation
def yelp_search_activities(state: State) -> State:
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

        logging.info("[Response]: " + str(response.json()))
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
        logging.info("[SEARCHING FOR YELP ACTIVITIES] Found activities: " + str(filtered_results))
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

# query google calendar
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the calendar results
def query_google_calendar(state: State) -> State:
    """Query the user's Google Calendar for events"""
    try:
        logging.info("[GOOGLE CALENDAR] Starting calendar event creation")
        
        # Generate calendar event parameters using LLM
        generate_params = llm_fast.with_structured_output(CalendarState)
        
        system_prompt = """Generate a calendar event for the user based on their query. 
        Required fields: summary, start_datetime, end_datetime
        Optional fields: description, calendar_id, location (Don't use None for any fields, use empty string if needed)
        Use ISO 8601 format for dates (e.g., "2024-01-15T14:30:00Z")
        If no specific time is mentioned, use reasonable defaults."""
        
        result = generate_params.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["messages"][-1].content)
        ])
        # convert pydantic class to dictionary
        if isinstance(result, BaseModel):
            result_dict = result.model_dump()
        else:
            result_dict = CalendarState(**result).model_dump()
        
        logging.info(f"[GOOGLE CALENDAR] Generated event params: {result_dict}")
        
        # Get the first available tool (Google Calendar)
        if not tools:
            raise Exception("No Google Calendar tools available")
        
        tool_name = "GoogleCalendar.CreateEvent"
        logging.info(f"[GOOGLE CALENDAR] Using tool: {tool_name}")
        
        # Authorize user
        auth_response = client.tools.authorize(
            user_id=user_id,
            tool_name=tool_name
        )
        
        # Check if authorization is needed
        if hasattr(auth_response, 'url') and auth_response.url:
            logging.info(f"[GOOGLE CALENDAR] Authorization required: {auth_response.url}")
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {
                    "calendar_results": f"Please authorize Google Calendar access: {auth_response.url}"
                }
            }
        
        # Wait for authorization completion
        client.auth.wait_for_completion(auth_response)
        
        # prepare input for tool from pydantic model
        tool_input = {k: result_dict.get(k) for k in [
            "summary", "description", "start_datetime", "end_datetime", "location"
        ]}


        # Execute the calendar tool
        response = client.tools.execute(
            user_id=user_id,
            tool_name=tool_name,
            input=tool_input
        )
        
        logging.info(f"[GOOGLE CALENDAR] Tool execution response: {response}")
        
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "calendar_results": "Successfully created calendar event",
                "calendar_link": response.get('htmlLink', 'No link found')
            }
        }
        
    except Exception as e:
        logging.error(f"[GOOGLE CALENDAR] Error: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "calendar_results": f"Failed to create calendar event: {str(e)}"
            }
        }


# get google flights
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the flight results
def get_google_flights(state: State) -> State:
    """Get the best flights for the user using extracted location parameters"""
    query = state["messages"][-1].content

    generate_params = llm_fast.with_structured_output(FlightState)
    system_prompt = """Fill in the flight parameters for the user's query. 
    Required fields: departure_airport_code, arrival_airport_code, outbound_date
    Optional fields: num_adults, sort_by
    
    If user query doesn't provide enough information, use reasonable defaults.
    EVERYTHING MUST BE UPPERCASE
    """
    flight_params = generate_params.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])

    logging.info(f"[GOOGLE FLIGHTS] Generated FlightState: {flight_params.model_dump()}")
    converted_params = flight_params.model_dump()
    try:
        tool_name = "GoogleFlights.SearchOneWayFlights"
    
        tool_input = {
            "departure_airport_code": converted_params.get("departure_airport_code", "LAX"),
            "arrival_airport_code": converted_params.get("arrival_airport_code", "SFO"),
            "outbound_date": converted_params.get("outbound_date", "2025-08-19"),
            "num_adults": converted_params.get("num_adults", 1),
            "sort_by": converted_params.get("sort_by", "PRICE"),
        }

        response = client.tools.execute(
            tool_name=tool_name,
            input=tool_input,
            user_id=user_id
        )

        logging.info(f"[GOOGLE FLIGHTS] Tool execution response: {response}")

        results = response.output.value
        filtered_results = []
        if results:
            # extract relevant information from results
            for result in results['flights']:
                airline_logo = result["airline_logo"]
                extra_info = result["extensions"]
                flight_segments = result["flights"]
                price = result["price"]
                total_duration = result["total_duration"]
                filtered_results.append({
                    "airline_logo": airline_logo,
                    "extra_info": extra_info,
                    "flight_segments": flight_segments,
                    "price": price,
                    "total_duration": total_duration,
                })
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "flight_results": filtered_results
            }
        }
    except Exception as e:
        logging.error(f"[GOOGLE FLIGHTS] Error: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "flight_results": f"Failed to get flights: {str(e)}"
            }
        }

# get google hotels
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the hotel results
def get_google_hotels(state: State) -> State:
    """Get the best hotels for the user using extracted location parameters"""
    query = state["messages"][-1].content
    generate_params = llm_fast.with_structured_output(HotelState)
    system_prompt = """Fill in the hotel parameters for the user's query. 
    Required fields: location, check_in_date, check_out_date
    Optional fields: query, min_price, max_price, num_adults, sort_by
    If user query doesn't provide enough information, use reasonable defaults.

    EVERYTHING MUST BE UPPERCASE
    """
    hotel_params = generate_params.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    tool_name = "GoogleHotels.SearchHotels"
    converted_params = hotel_params.model_dump()

    logging.info(f"[GOOGLE HOTELS] Generated HotelState: {converted_params}")
    try:
        tool_input = {
            "location": converted_params.get("location", "San Diego"),
            "check_in_date": converted_params.get("check_in_date", "2025-08-19"),
            "check_out_date": converted_params.get("check_out_date", "2025-08-20"),
        }

        # add optional parameters if they are provided
        if converted_params.get("query"):
            tool_input["query"] = converted_params.get("query")
        if converted_params.get("min_price"):
            tool_input["min_price"] = converted_params.get("min_price")
        if converted_params.get("max_price"):
            tool_input["max_price"] = converted_params.get("max_price")
        if converted_params.get("num_adults"):
            tool_input["num_adults"] = converted_params.get("num_adults")
        if converted_params.get("sort_by"):
            tool_input["sort_by"] = converted_params.get("sort_by")

        response = client.tools.execute(
            tool_name=tool_name,
            input=tool_input,
            user_id=user_id
        )

        results = response.output.value
        filtered_results = []
        if results:
            for result in results['properties']:
                name = result["name"]
                description = result.get("description", "")  # Use .get() with default
                essential_info = result.get("essential_info", [])  # Use .get() with default
                nearby_places = result["nearby_places"]
                amenities = result["amenities"]
                check_in_time = result["check_in_time"]
                check_out_time = result["check_out_time"]
                link = result.get("link", "")  # Use .get() with default
                overall_rating = result["overall_rating"]
                num_reviews = result["reviews"]
                rate_per_night = result["rate_per_night"]
                rate_info = rate_per_night.get("lowest", "")
                total_rate = result.get("total_rate", {}).get("lowest", "")

                filtered_results.append({
                    "name": name,
                    "description": description,
                    "essential_info": essential_info,
                    "nearby_places": nearby_places,
                    "amenities": amenities,
                    "check_in_time": check_in_time,
                    "check_out_time": check_out_time,
                    "link": link,
                    "overall_rating": overall_rating,
                    "num_reviews": num_reviews,
                    "rate_per_night": rate_per_night,
                    "rate_info": rate_info,
                    "total_rate": total_rate
                })
        logging.info(f"[GOOGLE HOTELS] Tool execution response: {response}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "hotel_results": filtered_results
            }
        }
    except Exception as e:
        logging.error(f"[GOOGLE HOTELS] Error: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "hotel_results": f"Failed to get hotels: {str(e)}"
            }
        }

# write into google docs
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the google docs results
def write_to_google_docs(state: State) -> State:
    """Write the user query to a google doc"""
    logging.info("[GOOGLE DOCS] Writing the user query to a google doc")
    
    tool_name = "GoogleDocs.CreateDocumentFromText"
    system_prompt = """You are a 21-year-old SoCal ABG bestie who's also a writing assistant. You help users create and edit documents in Google Docs with your signature playful, slangy, emoji-filled tone while being genuinely helpful.
        Your writing specialties include:
        - Travel itineraries and trip planning
        - Meeting notes and summaries
        - Creative writing and brainstorming
        - Academic writing and research notes
        - Personal journaling and reflections
        - Work documents and presentations
        - Social media content and captions
        """
    params = llm_fast.with_structured_output(GoogleDocsState)
    result = params.invoke([SystemMessage(content=system_prompt), HumanMessage(content=state["messages"][-1].content)])
    result_dict = result.model_dump()

    tool_input = {
        "title": result_dict.get("title", "Untitled"),
        "text_content": result_dict.get("text_content", "No content provided")
    }

    auth_response = client.tools.authorize(
        tool_name=tool_name,
        user_id=user_id
    )

    if hasattr(auth_response, 'url') and auth_response.url:
        logging.info(f"[GOOGLE DOCS] Authorization required: {auth_response.url}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "document_results": f"Please authorize Google Docs access: {auth_response.url}"
            }
        }

    client.auth.wait_for_completion(auth_response)

    response = client.tools.execute(
        tool_name=tool_name,
        input=tool_input,
        user_id=user_id
    )

    logging.info(f"[GOOGLE DOCS] Tool execution response: {response}")
    return {
        "messages": state["messages"],
        "message_type": state.get("message_type"),
        "result": {
            "document_results": result
        }
    }

# search reddit forums
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the reddit forums results
def search_reddit_forums(state: State) -> State:
    """Search the reddit forums for the user query"""
    keyword_search = llm_main.invoke([SystemMessage(content="Exctract main keyword from the user query for searching posts within subreddit."), HumanMessage(content=state["messages"][-1].content)])
    search = keyword_search.content
    try:
        # Initialize Reddit client with proper error handling
        reddit_client = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        
        subreddit_name = "ravExchange"
        subreddit = reddit_client.subreddit(subreddit_name)
        results = []

        for post in subreddit.search(search, limit=5):
            results.append({
                "title": post.title,
                "url": post.url,
                "score": post.score,
                "num_comments": post.num_comments,
                "author": str(post.author) if post.author else "Unknown",
            })

        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "reddit_results": results
            }
        }
        
    except Exception as e:
        logging.error(f"Reddit search error: {e}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "error": f"Reddit search failed: {str(e)}"
            }
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
graph.add_node("yelp_search_activities", yelp_search_activities)
graph.add_node("search_web", search_web)
graph.add_node("create_calendar_event", query_google_calendar)
graph.add_node("get_google_flights", get_google_flights)
graph.add_node("get_google_hotels", get_google_hotels)
graph.add_node("spotify_play_track", spotify_play_track)
graph.add_node("write_to_google_docs", write_to_google_docs)
graph.add_node("search_reddit_forums", search_reddit_forums)
# add edges
graph.add_edge(START, "classify_user_query")
graph.add_conditional_edges("classify_user_query", 
    lambda state: state.get("message_type", "default_llm_response"),
    {
        "song_rec": "song_rec",
        "get_concerts": "get_concerts",
        "get_weather": "get_weather",
        "yelp_search_activities": "yelp_search_activities",
        "create_calendar_event": "create_calendar_event",
        "get_google_flights": "get_google_flights",
        "get_google_hotels": "get_google_hotels",
        "default_llm_response": "default_llm_response",
        "write_to_google_docs": "write_to_google_docs",
        "search_reddit_forums": "search_reddit_forums"
    }
)
# graph.add_edge("classify_user_query", "search_web")
graph.add_edge("search_reddit_forums", END)
graph.add_edge("write_to_google_docs", END)
graph.add_edge("song_rec", "spotify_play_track")
graph.add_edge("spotify_play_track", END)
graph.add_edge("search_web", "default_llm_response")
graph.add_edge("get_concerts", "default_llm_response") # temporary
graph.add_edge("get_weather", "default_llm_response")
graph.add_edge("yelp_search_activities", "default_llm_response")
graph.add_edge("create_calendar_event", END)
graph.add_edge("get_google_flights", END)
graph.add_edge("get_google_hotels", END)
graph.add_edge("default_llm_response", END)



# compile graph
compiled_graph = graph.compile()