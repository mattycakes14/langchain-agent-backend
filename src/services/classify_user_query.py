from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from models.state import MessageClassifier, State
from services.extract_params import extract_parameters_llm
import logging
from config.settings import llm_fast

# Configure logging with more detail
logging.basicConfig(level=logging.INFO)

def classify_user_query(state: State) -> State:
    message = state["messages"][0]

    # invoke LLM that only returns a structured output
    classifier_llm = llm_fast.with_structured_output(MessageClassifier)
    
    # content of message
    content = """
    You are a message classifier. Analyze the user's message and classify it into one of the specified types.
    The user's message is: {message}
    The message types are:
    - song_rec: The user is asking for a song recommendation.
    - get_concerts: The user is asking for a concert recommendation.
    - get_weather: The user is asking for the weather.
    - yelp_search_activities: The user is asking for a restaurant, cafe, or other activity recommendation.
    - create_calendar_event: The user wants to create a calendar event, schedule something, or add an event to their calendar.
    - get_google_flights: The user is asking for flight information.
    - get_google_hotels: The user is asking for hotel information.
    - write_to_google_docs: The user is asking to write to a google doc.
    - search_reddit_forums: The user is asking to search the reddit forums.
    - post_to_reddit: The user is asking to post to the reddit forums.
    - default_llm_response: The user is asking a question that doesn't fit into any of the other categories.
    """

    # Get the structured output
    result = classifier_llm.invoke([
        SystemMessage(content=content),
        HumanMessage(content=message.content)
    ])
    logging.info(f"[CLASSIFYING MESSAGE] Classified message: {result.message_type}")

    
    # Extract parameters using LLM
    extracted_params = extract_parameters_llm(message.content, result.message_type)
    
    logging.info(f"[EXTRACTING PARAMETERS] Extracted parameters: {extracted_params}")
    # Update the state with the classification result and extracted parameters
    return {
        "messages": state["messages"],
        "message_type": result.message_type,
        "extracted_params": extracted_params
    }