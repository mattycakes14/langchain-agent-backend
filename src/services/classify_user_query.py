from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from models.state import MessageClassifier, State
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
    
    user_messages = conversation_history.get("user_messages", []) if conversation_history else []
    agent_result = conversation_history.get("agent_result", "") if conversation_history else ""

    logging.info(f"[Fetching conversation history] User messages: {user_messages}")
    logging.info(f"[Fetching conversation history] Agent result: {agent_result}")
    logging.info(f"[Fetching conversation history] Agent result length: {len(str(agent_result))} characters")

    
    # content of message
    content = f"""
        Analyze this user message with conversation context:
        Message: {message.content}
        Recent context: {agent_result}

        Determine the message type based on the user's intent and the conversation context.
        Append RELEVANT CONVERSATION HISTORY to conversation_history. If NOT RELEVANT, RETURN EMPTY STRING.
    """

    llm_params = llm_fast.with_structured_output(MessageClassifier)

    # Get the structured output
    result = llm_params.invoke([
        SystemMessage(content=content),
        HumanMessage(content=message.content)
    ])

    result = result.model_dump()
    logging.info(f"[CLASSIFYING MESSAGE] Classified message: {result}")

    
    # Update the state with the classification result and extracted parameters
    return {
        "messages": state["messages"],
        "message_type": result.get("message_type", "default_llm_response"),
        "conversation_history": result.get("conversation_history", ""),
    }
