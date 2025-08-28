from typing import Annotated, TypedDict, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import add_messages

# State structure for each node
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: Annotated[str, "The type of message the user is sending"] | None
    result: dict | None
    extracted_params: dict | None  # Add this back
    search_results: dict | None

# Define structured output for parameter extraction (API call query parameters)
class ExtractedParams(BaseModel):
    lat: Optional[float] = Field(description="Extracted latitude from the query", default=None)
    lon: Optional[float] = Field(description="Extracted longitude from the query", default=None)
    keyword: Optional[str] = Field(description="Extracted main keyword from the query", default=None)

# Classify user query
class MessageClassifier(BaseModel):
    message_type: Literal["default_llm_response", "song_rec", "get_concerts", "get_weather", "yelp_search_activities", "create_calendar_event", "get_google_flights", "get_google_hotels", "write_to_google_docs", "search_reddit_forums", "spotify_play_track"] = Field(
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
