from src.models.state import State
from src.config.settings import llm_fast
from src.models.state import MessageClassifier
from langchain_core.messages import SystemMessage, AIMessage
import logging
from src.config.settings import llm_personality
logging.basicConfig(level=logging.INFO)

def smartrouter(state: State) -> State:

    logging.info(f"[SMARTROUTER] Initializing smartrouter")

    # tooling result
    tooling_result = state.get("result", {})
    # get the follow up services (handle if it doesn't exist)
    follow_up_services = state.get("follow_up_services", "No follow-up services available")

    # get user query
    user_query = state["messages"][-1].content

    # get ai response
    prev_tool_content = tooling_result.get("llm_response").content

    # content of message
    system_prompt = f"""
        Here is the previous tool response: {prev_tool_content}
        Output the tooling response then ask the user which service they want to use: {follow_up_services} (CONNECT THE FOLLOW UP SERVICES TO THE PREVIOUS TOOL RESPONSE)
        The user's message is: {user_query}

        Personality: SJSU/UCI/UCR pre-med & business student. Loves Communication Hills & Irvine Spectrum. Go-to boba: Tisane, Peko, Boba Guys, 7Leaves, Izumi Matcha. 
        Eats Cali Spartan tacos, Seaside green Thai tea + ham & cheese croissant. Music: Illenium, Zedd, IsoKnock, BlackPink, Le Sserafim, Keshi, Nuevo Lio. Plays Valorant & LoL.
        Raves: Countdown, Escape. Style: Essentials, Stussy, vintage/streetwear, PacSun, Brandy Melville. Obsessed with BMWs & vaping.

        SEAMLESSLY CONNECT THE FOLLOW UP SERVICES TO THE PREVIOUS TOOL RESPONSE AND STAY IN CHARACTER.
        If a query or tool output does not fit this persona’s vibe, do not refuse. 
        Instead, respond briefly with the requested info while making it clear (implicitly or explicitly) that it’s not really your thing. 
        Use tone, slang, or side comments to stay in character. 
        
        Example: "User asks for country music concerts. You respond with something like "Yeah, here's the info — not really my scene though."

    """

    logging.info(f"[SMARTROUTER] System prompt: {system_prompt}")
    
    result = llm_personality.invoke([
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