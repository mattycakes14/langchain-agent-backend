from langgraph.graph import StateGraph, START, END
from models.state import State
from graph.nodes import (
    classify_user_query, search_songs, ticketmaster_search_event, 
    get_weather, get_LLM_response, yelp_search_activities, search_web,
    query_google_calendar, get_google_flights, get_google_hotels,
    spotify_play_track, write_to_google_docs, search_reddit_forums, 
    get_follow_up_services, smartrouter
)
from langgraph.checkpoint.redis import RedisSaver

# Initialize Redis saver properly - need to enter the context manager
_redis_checkpointer_ctx = RedisSaver.from_conn_string("redis://localhost:6379")
_redis_checkpointer = _redis_checkpointer_ctx.__enter__()

# create graph
graph = StateGraph(State)

# add nodes
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
graph.add_node("get_follow_up_services", get_follow_up_services)
graph.add_node("smartrouter", smartrouter)
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
        "search_reddit_forums": "search_reddit_forums",
        "spotify_play_track": "spotify_play_track",
        "search_web": "search_web",
        "spotify_play_track": "spotify_play_track"
    }
)
graph.add_edge("classify_user_query", "get_follow_up_services")

# All tools go to get_follow_up_services first, then smartrouter
graph.add_edge("song_rec", "smartrouter")
graph.add_edge("get_concerts", "smartrouter")
graph.add_edge("get_weather", "smartrouter")
graph.add_edge("yelp_search_activities", "smartrouter")
graph.add_edge("create_calendar_event", "smartrouter")
graph.add_edge("get_google_flights", "smartrouter")
graph.add_edge("get_google_hotels", "smartrouter")
graph.add_edge("write_to_google_docs", "smartrouter")
graph.add_edge("search_reddit_forums", "smartrouter")
graph.add_edge("spotify_play_track", END)
graph.add_edge("search_web", "smartrouter")
graph.add_edge("get_follow_up_services", "smartrouter")
graph.add_edge("default_llm_response", "smartrouter")
graph.add_edge("smartrouter", "classify_user_query")


# Compile graph with Redis checkpointer
compiled_graph = graph.compile(checkpointer=_redis_checkpointer)