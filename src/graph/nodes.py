from src.services.classify_user_query import classify_user_query
from src.services.search_song import search_songs
from src.services.ticketmaster_search import ticketmaster_search_event
from src.services.weather_service import get_weather
from src.services.google_service import query_google_calendar, get_google_flights, get_google_hotels, write_to_google_docs
from src.services.reddit_service import search_reddit_forums
from src.services.search_service import search_web
from src.services.spotify_service import spotify_play_track
from src.services.yelp_service import yelp_search_activities
from src.services.get_LLM_response import get_LLM_response
from src.services.get_follow_up_services import get_follow_up_services
from src.services.smartrouter import smartrouter
# Node functions for the graph
# These are imported and used in graph_builder.py