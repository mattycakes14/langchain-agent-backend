from arcadepy import Arcade
from dotenv import load_dotenv
import os

load_dotenv()

client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))  # Automatically finds the `ARCADE_API_KEY` env variable
USER_ID = "mlau191@uw.edu"

TOOL_NAME = "Spotify.PlayTrackByName"

auth_response = client.tools.authorize(
    tool_name=TOOL_NAME,
    user_id=USER_ID,
)

if auth_response.status != "completed":
    print(f"Click this link to authorize: {auth_response.url}")

# Wait for the authorization to complete
client.auth.wait_for_completion(auth_response)

tool_input = {"title": "Project Plan", "text_content": "This is the project plan."}
user_tokens = {
    "access_token": "<user-access-token>",
    "refresh_token": "<user-refresh-token>",
    # any other required token fields
}
response = client.tools.execute(
    tool_name=TOOL_NAME,
    input=tool_input,
    tokens=user_tokens
)
print(response)