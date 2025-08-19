#!/usr/bin/env python3
"""
Comprehensive test script to verify all imports work correctly
"""

def test_all_imports():
    """Test all imports in the src directory"""
    try:
        print("Testing all imports...")
        
        # Test config imports
        from src.config.settings import llm_main, llm_fast, user_id, OPENROUTER_API_KEY, OPENAI_API_KEY
        print("✅ Config imports successful")
        
        # Test models imports
        from src.models.state import State, MessageClassifier, ExtractedParams, CalendarState, FlightState, HotelState, SpotifyState, GoogleDocsState
        print("✅ Models imports successful")
        
        # Test graph imports
        from src.graph.graph_builder import compiled_graph
        from src.graph.nodes import classify_user_query, search_songs, ticketmaster_search_event, get_weather, get_LLM_response, yelp_search_activities, search_web, query_google_calendar, get_google_flights, get_google_hotels, spotify_play_track, write_to_google_docs, search_reddit_forums
        print("✅ Graph imports successful")
        
        # Test service imports
        from src.services.classify_user_query import classify_user_query
        from src.services.extract_params import extract_parameters_llm
        from src.services.search_song import search_songs
        from src.services.ticketmaster_search import ticketmaster_search_event
        from src.services.weather_service import get_weather
        from src.services.yelp_service import yelp_search_activities
        from src.services.search_service import search_web
        from src.services.google_service import query_google_calendar, get_google_flights, get_google_hotels, write_to_google_docs
        from src.services.spotify_service import spotify_play_track
        from src.services.reddit_service import search_reddit_forums
        from src.services.get_LLM_response import get_LLM_response
        print("✅ Service imports successful")
        
        # Test utils imports
        from src.utils.embedding import get_embedding
        print("✅ Utils imports successful")
        
        # Test static content imports
        from src.static_content.concert_filters import festival_to_description
        from src.static_content.yelp_categories import yelp_categories
        from src.static_content.descriptions_to_aliases import descriptions_to_aliases
        print("✅ Static content imports successful")
        
        # Test main app import
        from src.main import app
        print("✅ Main app import successful")
        
        print("🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_all_imports()
