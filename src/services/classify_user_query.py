from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from models.state import MessageClassifier, State
from services.extract_params import extract_parameters_llm
from services.get_conversation_history import get_conversation_window
import logging
from config.settings import llm_fast, llm_main
import redis
import json

# Configure logging with more detail
logging.basicConfig(level=logging.INFO)

def classify_user_query(state: State) -> State:
    # get the last message
    message = state["messages"][-1]
    logging.info(f"[CLASSIFYING MESSAGE] latest message: {message}")

    # get the conversation history
    conversation_history = get_conversation_window("matt1234")    
    user_messages = conversation_history.get("user_messages", [])
    agent_result = conversation_history.get("agent_result", "")

    logging.info(f"[Fetching conversation history] User messages: {user_messages}")
    logging.info(f"[Fetching conversation history] Agent result: {agent_result}")
    logging.info(f"[Fetching conversation history] Agent result length: {len(str(agent_result))} characters")
    prompt = f"""
        User messages: {str(user_messages)}
        Agent result: {str(agent_result)}

        Is this conversation history relevant to the new user query? 
        Answer with ONLY "RELEVANT" or "NOT RELEVANT". 
        If relevant, provide a 1-sentence summary (max 20 words).
    """

    related_messages = llm_main.invoke([SystemMessage(content=prompt)])
    # invoke LLM that only returns a structured output
    
    logging.info(f"[LLM DECISION FOR CONVERSATION HISTORY] {related_messages.content}")

    classifier_llm = llm_fast.with_structured_output(MessageClassifier)
    
    # content of message
    content = f"""
    You are a message classifier. Analyze the user's message and classify it into one of the specified types.
    The user's message is: {str(message)}
    With conversation history: {str(related_messages.content)}

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
    - spotify_play_track: The user is asking to play a song on spotify.
    - search_web: The user is asking to search the web.
    - get_follow_up_services: The user is asking for follow up services.
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
        "extracted_params": extracted_params,
        "conversation_history": related_messages.content
    }