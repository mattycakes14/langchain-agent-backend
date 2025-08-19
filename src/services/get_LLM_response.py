from models.state import State
import logging
from config.settings import llm_main
from langchain_core.messages import SystemMessage
# Configure logging with more detail
logging.basicConfig(level=logging.INFO)
def get_LLM_response(state: State) -> State:
    query = state["messages"][-1].content
    search_results = state.get("search_results", {})
    
    # extract links from search results
    if 'results' in search_results and search_results['results']:
        links = [result["url"] for result in search_results['results']]
    else:
        links = []

    logging.info("[SEARCHING THE WEB] Search results links: " + str(links))
    # retrieve content from previous nodes
    song_rec = state.get("result", {}).get("song_recommendation", {})
    concerts = state.get("result", {}).get("events", {})
    weather = state.get("result", {}).get("weather_data", {})
    food = state.get("result", {}).get("food_results", {})
    calendar_results = state.get("result", {}).get("calendar_results", {})
    auth_complete = state.get("result", {}).get("auth_complete", {})

    output = None
    if song_rec:
        output = song_rec
    elif food:
        output = food
    elif concerts:
        output = concerts
    elif weather:
        output = weather
    elif calendar_results:
        output = calendar_results
    elif auth_complete:
        output = auth_complete
    else:
        output = query

    logging.info("[INTEGRATING RESULT] output: " + str(output))
    
    system_prompt = """ You are a 21-year-old SoCal ABG bestie. Always speak in playful, slangy, emoji-filled tone (“omg babe,” “lowkey,” “no cap”). Hype your faves, roast/judge anything off-vibe, but still give helpful answers.
        Faves: Communication Hills, Irvine Spectrum; SJSU/UCI/UCR; Pre-med/Business; Boba (Tisane, Peko, Boba Guys, 7Leaves, Izumi Matcha); Food (Cali Spartan Tacos, Green Thai Tea, Seaside ham & cheese croissant); Artists (Illenium, Zedd, IsoKnock, BlackPink, Le Sserafim, Keshi, Nuevo Lio); Games (Valorant, LoL); Raves (Countdown, Escape, Decadence, Breakaway, EDC, NorthCoast); Fashion (Essentials, Stussy, Vintage, PacSun, Brandy Melville); Loves BMWs & vaping.
        Rules: If topic fits your vibe → hype it.
        If not → playful shade + answer + steer back to faves.
        Never break character.
        
        Here is the user query: """ + query + """
        Here is the output: """ + str(output) + """
        
        List the search results in a numbered list. """ + str(links) + """
        """
    try:
        response = llm_main.invoke([
            SystemMessage(content=system_prompt),
        ])
    except Exception as e:
        print(f"Error getting LLM response: {str(e)}")
        return None
    return {
        "messages": state["messages"],
        "message_type": state.get("message_type"),
        "result": {"llm_response": response}
    }
