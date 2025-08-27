from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from models.state import MessageClassifier, State
from services.extract_params import extract_parameters_llm
import logging
from config.settings import llm_fast
from config.settings import llm_main
from config.settings import llm_advanced
def get_follow_up_services(state: State) -> State:
    # extract the message type
    message_type = state["message_type"]

    system_prompt = f""" 
    Given this message type: {message_type}, recommend follow up services to help the user THAT MAKES THE MOST SENSE.
    The follow up services are:
    - song_rec: get a song recommendation
    - get_concerts: get a concert recommendation
    - get_weather: get the weather
    - yelp_search_activities: get a restaurant, cafe, or other activity recommendation
    - create_calendar_event: create a calendar event, schedule something, or add an event to their calendar
    - get_google_flights: get flight information
    - get_google_hotels: get hotel information
    - write_to_google_docs: write to a google doc
    - search_reddit_forums: search the reddit forums
    - spotify_play_track: play a song on spotify
    - search_web: search the web
    - default_llm_response: default llm response

    The follow up services should be a list of services that make the most sense to help the user. (JUST RETURN THE LIST OF SERVICES, NO OTHER TEXT)
    """
    result = llm_fast.invoke([
        SystemMessage(content=system_prompt)
    ])
    logging.info(f"[GET FOLLOW UP SERVICES] Follow up services: {result.content}")
    return {
        "messages": state["messages"],
        "follow_up_services": result.content
    }
