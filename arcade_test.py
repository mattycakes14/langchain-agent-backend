from langchain_arcade import ArcadeToolManager
from dotenv import load_dotenv
import os
from arcadepy import Arcade

load_dotenv()

client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))

manager = ArcadeToolManager(api_key=os.getenv("ARCADE_API_KEY"))
 
# Get all tools from the "Gmail" toolkit
tools = manager.get_tools(toolkits=["GoogleCalendar"])
print(manager.tools[0])

# auth_response = client.tools.authorize(
#     user_id="mlau191@uw.edu",
#     tool_name=manager.tools[3]
# )

# if auth_response.status != "completed":
#     print(f"click this link to authorize: {auth_response.url}")

# # wait for response
# client.auth.wait_for_completion(auth_response)

# tool_input = {
#     "max_results": 5,
# }

# response = client.tools.execute(
#     user_id="mlau191@uw.edu",
#     tool_name=manager.tools[3],
#     input=tool_input
# )

# print(response)