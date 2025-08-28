from models.state import State
import logging

# Rule-based follow-up services mapping
FOLLOW_UP_RULES = {
    "song_rec": ["spotify_play_track", "get_concerts", "create_calendar_event", "search_web"],
    "get_concerts": ["create_calendar_event", "get_google_hotels", "yelp_search_activities", "song_rec"],
    "get_weather": ["yelp_search_activities", "create_calendar_event", "get_google_flights"],
    "yelp_search_activities": ["create_calendar_event", "get_weather", "get_google_hotels"],
    "create_calendar_event": ["get_weather", "yelp_search_activities", "get_google_flights"],
    "get_google_flights": ["get_google_hotels", "get_weather", "create_calendar_event"],
    "get_google_hotels": ["yelp_search_activities", "get_weather", "create_calendar_event"],
    "write_to_google_docs": ["search_web", "search_reddit_forums"],
    "search_reddit_forums": ["write_to_google_docs", "search_web"],
    "spotify_play_track": ["get_concerts", "create_calendar_event"],
    "search_web": ["write_to_google_docs", "search_reddit_forums"],
    "default_llm_response": ["search_web", "search_reddit_forums"]
}

def get_follow_up_services(state: State) -> State:
    # extract the message type
    message_type = state["message_type"]
    
    # Get follow-up services based on rules (exclude the current service)
    follow_up_services = FOLLOW_UP_RULES.get(message_type, ["search_web"])
    
    # Convert to the format expected by smartrouter
    follow_up_list = [f'"{service}"' for service in follow_up_services]
    follow_up_services_str = "[" + ",".join(follow_up_list) + "]"
    
    logging.info(f"[GET FOLLOW UP SERVICES] Follow up services: {follow_up_services_str}")
    return {
        "messages": state["messages"],
        "follow_up_services": follow_up_services_str
    }