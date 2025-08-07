from langchain.agents import Tool

def spotify_play_song(query: str) -> str:
    # Your code to call Spotify API and return play confirmation or song info
    return f"Playing {query} on Spotify."

def ticketmaster_search_event(query: str) -> str:
    # Your code to call TicketMaster API and return event info
    return f"Found festival info for {query}."

# Wrap functions as LangChain Tools
spotify_tool = Tool(
    name="SpotifyPlay",
    func=spotify_play_song,
    description="Play music on Spotify based on a user's request."
)

ticketmaster_tool = Tool(
    name="TicketMasterSearch",
    func=ticketmaster_search_event,
    description="Retrieve real-time festival dates and ticket info from TicketMaster."
)
