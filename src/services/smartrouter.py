from models.state import State
from config.settings import llm_fast
from models.state import MessageClassifier
from langchain_core.messages import SystemMessage
import logging

logging.basicConfig(level=logging.INFO)

def smartrouter(state: State) -> State:

    logging.info(f"[SMARTROUTER] Initializing smartrouter")
    # tooling result
    tooling_result = state.get("result", {})
    # get the follow up services (handle if it doesn't exist)
    follow_up_services = state.get("follow_up_services", "No follow-up services available")

    # get user query
    user_query = state["messages"][0].content

    # content of message
    system_prompt = f"""
        Here is the previous tool response: {tooling_result}
        Output the tooling response then ask the user which service they want to use: {follow_up_services} (connect the follow up services to the previous tool response)
        The user's message is: {user_query}
    """

    result = llm_fast.invoke([
        SystemMessage(content=system_prompt),
    ])

    logging.info(f"[SMARTROUTER] Result: {result}")
    return {
        "messages": state["messages"],
        "message_type": state.get("message_type"),
        "result": {
            "smartrouter_result": result.content
        }
    }