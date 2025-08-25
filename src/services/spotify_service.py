from models.state import State
import logging
import os
from dotenv import load_dotenv
from arcadepy import Arcade
from langchain_arcade import ArcadeToolManager
from config.settings import user_id
# Configure logging with more detail
logging.basicConfig(level=logging.INFO)
# load environment variables
load_dotenv()

# Initialize Arcade client
client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))

# play a track on spotify
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the spotify response
def spotify_play_track(state: State) -> State:
    """Play a track on Spotify"""
    song_rec = state.get("result", {}).get("song_recommendation", {})
    artists = song_rec.get("artists", "")
    title = song_rec.get("title", "")

    logging.info("[SPOTIFY PLAY TRACK] Playing track: " + title + " by " + artists)

    tool_name = "Spotify.PlayTrackByName"
    
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
