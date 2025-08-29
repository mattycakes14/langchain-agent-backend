from models.state import State, SpotifyState
import logging
import os
from dotenv import load_dotenv
from arcadepy import Arcade
from langchain_arcade import ArcadeToolManager
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import llm_fast
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
    # get the conversation history
    conversation_history = state.get("conversation_history", "")

    llm_params = llm_fast.with_structured_output(SpotifyState)
    prompt = f"""
        Decide which song the user wants to play. The user's message is: {str(state["messages"][-1].content)}
        with conversation history: {conversation_history}
        
        ARTIST CAN ONLY BE ONE NAME (NO AND, NO COMMA)
        ONLY SPACE BETWEEN FIRST AND LAST NAME
    """
    # decide which song user wants to play
    result = llm_params.invoke([SystemMessage(content=prompt)])

    logging.info(f"[SPOTIFY PLAY TRACK] Result: {result}")
    result = result.model_dump()
    title = result.get("track_name", "")
    artists = result.get("artist_name", "")

    # convert the pydantic model to a dictionary
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
