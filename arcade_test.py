from arcadepy import Arcade
import os
from dotenv import load_dotenv

load_dotenv()
# You can also set the `ARCADE_API_KEY` environment variable instead of passing it as a parameter.
client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))
 
# Arcade needs a unique identifier for your application user (this could be an email address, a UUID, etc).
# In this example, use the email you used to sign up for Arcade.dev:
user_id = "mlau191@uw.edu"
 
response = client.tools.execute(
    tool_name="GoogleFlights.SearchOneWayFlights",
    input={
        "departure_airport_code": "LAX",
        "arrival_airport_code": "SFO",
        "outbound_date": "2025-08-19",
        "num_adults": 1,
        "sort_by": "PRICE",
    },
    user_id="mlau191@uw.edu"
)
filtered_results = []
for flight in response.output.value['flights']:
    airline_logo = flight["airline_logo"]
    extra_info = flight["extensions"]
    flight_segments = flight["flights"]
    price = flight["price"]
    total_duration = flight["total_duration"]
    filtered_results.append({
        "airline_logo": airline_logo,
        "extra_info": extra_info,
        "flight_segments": flight_segments,
        "price": price,
        "total_duration": total_duration,
    })
print(filtered_results)

 