from src.models.state import State
import os
import requests
from dotenv import load_dotenv
import logging
from src.config.settings import llm_fast
from src.models.state import LocationState
from langchain_core.messages import SystemMessage

# load environment variables
load_dotenv()
# Configure logging with more detail
logging.basicConfig(level=logging.INFO)
# get weather using OpenWeatherMap API
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the weather recommendation
def get_weather(state: State) -> State:
    """Get the weather for the user using extracted location parameters"""
    query = state["messages"][-1].content
    
    llm_params = llm_fast.with_structured_output(LocationState)
    prompt = f"""
        Fetch most accurate longitude and latitude of the location the user wants to search for weather. The user's message is: {str(state["messages"][-1].content)}
    """

    llm_result = llm_params.invoke([SystemMessage(content=prompt)])

    logging.info(f"[GET WEATHER] LLM result: {llm_result}")
    llm_result = llm_result.model_dump()
    longitude = llm_result.get("longitude", 0)
    latitude = llm_result.get("latitude", 0)

    # Query parameters
    #lat: Latitude of the location (Float) REQUIRED
    #lon: Longitude of the location (Float) REQUIRED
    #appid: OpenWeatherMap API key (String) REQUIRED
    #units: Units to use for temperature (String) OPTIONAL
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": os.getenv("OPENWEATHERMAP_API_KEY"),
        "units": "imperial"
    }
    
    try:
        response = requests.get(url, params=params)
        weather_data = response.json()
        
        logging.info(f"[GET WEATHER] Weather data: {weather_data}")
        
        # Check if the API call was successful
        if weather_data.get("cod") == 200:
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {
                    "weather_data": weather_data,
                },
            }
        else:
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {
                    "weather_data": "No weather data found"
                },
            }
            
    except Exception as e:
        print(f"Error getting weather: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {"error": f"Error getting weather: {str(e)}"},
        }
