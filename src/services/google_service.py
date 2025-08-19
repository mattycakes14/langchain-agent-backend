from models.state import State
import os
import logging
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from models.state import CalendarState, FlightState, HotelState, GoogleDocsState
from arcadepy import Arcade
from langchain_arcade import ArcadeToolManager
from config.settings import llm_fast, user_id
# load environment variables
load_dotenv()

# Initialize Arcade client
client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))
# Configure logging with more detail
logging.basicConfig(level=logging.INFO)

# query google calendar
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the calendar results
def query_google_calendar(state: State) -> State:
    """Query the user's Google Calendar for events"""
    try:
        logging.info("[GOOGLE CALENDAR] Starting calendar event creation")
        
        # Generate calendar event parameters using LLM
        generate_params = llm_fast.with_structured_output(CalendarState)
        
        system_prompt = """Generate a calendar event for the user based on their query. 
        Required fields: summary, start_datetime, end_datetime
        Optional fields: description, calendar_id, location (Don't use None for any fields, use empty string if needed)
        Use ISO 8601 format for dates (e.g., "2024-01-15T14:30:00Z")
        If no specific time is mentioned, use reasonable defaults."""
        
        result = generate_params.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["messages"][-1].content)
        ])
        # convert pydantic class to dictionary
        if isinstance(result, BaseModel):
            result_dict = result.model_dump()
        else:
            result_dict = CalendarState(**result).model_dump()
        
        logging.info(f"[GOOGLE CALENDAR] Generated event params: {result_dict}")
        

        
        tool_name = "GoogleCalendar.CreateEvent"
        logging.info(f"[GOOGLE CALENDAR] Using tool: {tool_name}")
        
        # Authorize user
        auth_response = client.tools.authorize(
            user_id=user_id,
            tool_name=tool_name
        )
        
        # Check if authorization is needed
        if hasattr(auth_response, 'url') and auth_response.url:
            logging.info(f"[GOOGLE CALENDAR] Authorization required: {auth_response.url}")
            return {
                "messages": state["messages"],
                "message_type": state.get("message_type"),
                "result": {
                    "calendar_results": f"Please authorize Google Calendar access: {auth_response.url}"
                }
            }
        
        # Wait for authorization completion
        client.auth.wait_for_completion(auth_response)
        
        # prepare input for tool from pydantic model
        tool_input = {k: result_dict.get(k) for k in [
            "summary", "description", "start_datetime", "end_datetime", "location"
        ]}


        # Execute the calendar tool
        response = client.tools.execute(
            user_id=user_id,
            tool_name=tool_name,
            input=tool_input
        )
        
        logging.info(f"[GOOGLE CALENDAR] Tool execution response: {response}")
        
        # Handle the ExecuteToolResponse object properly
        calendar_link = "No link found"
        if hasattr(response, 'output') and hasattr(response.output, 'value'):
            # Try to extract the link from the response
            response_data = response.output.value
            if isinstance(response_data, dict):
                calendar_link = response_data.get('htmlLink', 'No link found')
        
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "calendar_results": "Successfully created calendar event",
                "calendar_link": calendar_link
            }
        }
        
    except Exception as e:
        logging.error(f"[GOOGLE CALENDAR] Error: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "calendar_results": f"Failed to create calendar event: {str(e)}"
            }
        }


# get google flights
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the flight results
def get_google_flights(state: State) -> State:
    """Get the best flights for the user using extracted location parameters"""
    query = state["messages"][-1].content

    generate_params = llm_fast.with_structured_output(FlightState)
    system_prompt = """Fill in the flight parameters for the user's query. 
    Required fields: departure_airport_code, arrival_airport_code, outbound_date
    Optional fields: num_adults, sort_by
    
    If user query doesn't provide enough information, use reasonable defaults.
    EVERYTHING MUST BE UPPERCASE
    """
    flight_params = generate_params.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])

    logging.info(f"[GOOGLE FLIGHTS] Generated FlightState: {flight_params.model_dump()}")
    converted_params = flight_params.model_dump()
    try:
        tool_name = "GoogleFlights.SearchOneWayFlights"
    
        tool_input = {
            "departure_airport_code": converted_params.get("departure_airport_code", "LAX"),
            "arrival_airport_code": converted_params.get("arrival_airport_code", "SFO"),
            "outbound_date": converted_params.get("outbound_date", "2025-08-19"),
            "num_adults": converted_params.get("num_adults", 1),
            "sort_by": converted_params.get("sort_by", "PRICE"),
        }

        response = client.tools.execute(
            tool_name=tool_name,
            input=tool_input,
            user_id=user_id
        )

        logging.info(f"[GOOGLE FLIGHTS] Tool execution response: {response}")

        results = response.output.value
        filtered_results = []
        if results:
            # extract relevant information from results
            for result in results['flights']:
                airline_logo = result["airline_logo"]
                extra_info = result["extensions"]
                flight_segments = result["flights"]
                price = result["price"]
                total_duration = result["total_duration"]
                filtered_results.append({
                    "airline_logo": airline_logo,
                    "extra_info": extra_info,
                    "flight_segments": flight_segments,
                    "price": price,
                    "total_duration": total_duration,
                })
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "flight_results": filtered_results
            }
        }
    except Exception as e:
        logging.error(f"[GOOGLE FLIGHTS] Error: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "flight_results": f"Failed to get flights: {str(e)}"
            }
        }

# get google hotels
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the hotel results
def get_google_hotels(state: State) -> State:
    """Get the best hotels for the user using extracted location parameters"""
    query = state["messages"][-1].content
    generate_params = llm_fast.with_structured_output(HotelState)
    system_prompt = """Fill in the hotel parameters for the user's query. 
    Required fields: location, check_in_date, check_out_date
    Optional fields: query, min_price, max_price, num_adults, sort_by
    If user query doesn't provide enough information, use reasonable defaults.

    EVERYTHING MUST BE UPPERCASE
    """
    hotel_params = generate_params.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    tool_name = "GoogleHotels.SearchHotels"
    converted_params = hotel_params.model_dump()

    logging.info(f"[GOOGLE HOTELS] Generated HotelState: {converted_params}")
    try:
        tool_input = {
            "location": converted_params.get("location", "San Diego"),
            "check_in_date": converted_params.get("check_in_date", "2025-08-19"),
            "check_out_date": converted_params.get("check_out_date", "2025-08-20"),
        }

        # add optional parameters if they are provided
        if converted_params.get("query"):
            tool_input["query"] = converted_params.get("query")
        if converted_params.get("min_price"):
            tool_input["min_price"] = converted_params.get("min_price")
        if converted_params.get("max_price"):
            tool_input["max_price"] = converted_params.get("max_price")
        if converted_params.get("num_adults"):
            tool_input["num_adults"] = converted_params.get("num_adults")
        if converted_params.get("sort_by"):
            tool_input["sort_by"] = converted_params.get("sort_by")

        response = client.tools.execute(
            tool_name=tool_name,
            input=tool_input,
            user_id=user_id
        )

        results = response.output.value
        filtered_results = []
        if results:
            for result in results['properties']:
                name = result["name"]
                description = result.get("description", "")  # Use .get() with default
                essential_info = result.get("essential_info", [])  # Use .get() with default
                nearby_places = result["nearby_places"]
                amenities = result["amenities"]
                check_in_time = result["check_in_time"]
                check_out_time = result["check_out_time"]
                link = result.get("link", "")  # Use .get() with default
                overall_rating = result["overall_rating"]
                num_reviews = result["reviews"]
                rate_per_night = result["rate_per_night"]
                rate_info = rate_per_night.get("lowest", "")
                total_rate = result.get("total_rate", {}).get("lowest", "")

                filtered_results.append({
                    "name": name,
                    "description": description,
                    "essential_info": essential_info,
                    "nearby_places": nearby_places,
                    "amenities": amenities,
                    "check_in_time": check_in_time,
                    "check_out_time": check_out_time,
                    "link": link,
                    "overall_rating": overall_rating,
                    "num_reviews": num_reviews,
                    "rate_per_night": rate_per_night,
                    "rate_info": rate_info,
                    "total_rate": total_rate
                })
        logging.info(f"[GOOGLE HOTELS] Tool execution response: {response}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "hotel_results": filtered_results
            }
        }
    except Exception as e:
        logging.error(f"[GOOGLE HOTELS] Error: {str(e)}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "hotel_results": f"Failed to get hotels: {str(e)}"
            }
        }

# write into google docs
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the google docs results
def write_to_google_docs(state: State) -> State:
    """Write the user query to a google doc"""
    logging.info("[GOOGLE DOCS] Writing the user query to a google doc")
    
    tool_name = "GoogleDocs.CreateDocumentFromText"
    system_prompt = """You are a 21-year-old SoCal ABG bestie who's also a writing assistant. You help users create and edit documents in Google Docs with your signature playful, slangy, emoji-filled tone while being genuinely helpful.
        Your writing specialties include:
        - Travel itineraries and trip planning
        - Meeting notes and summaries
        - Creative writing and brainstorming
        - Academic writing and research notes
        - Personal journaling and reflections
        - Work documents and presentations
        - Social media content and captions
        """
    params = llm_fast.with_structured_output(GoogleDocsState)
    result = params.invoke([SystemMessage(content=system_prompt), HumanMessage(content=state["messages"][-1].content)])
    result_dict = result.model_dump()

    tool_input = {
        "title": result_dict.get("title", "Untitled"),
        "text_content": result_dict.get("text_content", "No content provided")
    }

    auth_response = client.tools.authorize(
        tool_name=tool_name,
        user_id=user_id
    )

    if hasattr(auth_response, 'url') and auth_response.url:
        logging.info(f"[GOOGLE DOCS] Authorization required: {auth_response.url}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "document_results": f"Please authorize Google Docs access: {auth_response.url}"
            }
        }

    client.auth.wait_for_completion(auth_response)

    response = client.tools.execute(
        tool_name=tool_name,
        input=tool_input,
        user_id=user_id
    )

    logging.info(f"[GOOGLE DOCS] Tool execution response: {response}")
    return {
        "messages": state["messages"],
        "message_type": state.get("message_type"),
        "result": {
            "document_results": result
        }
    }
