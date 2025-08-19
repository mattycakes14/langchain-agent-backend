from models.state import State
import os
import requests
from dotenv import load_dotenv
import logging
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
