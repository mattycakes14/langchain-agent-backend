from src.models.state import State
import logging
import os
import praw
from dotenv import load_dotenv
from src.config.settings import llm_main
from langchain_core.messages import SystemMessage, HumanMessage

# load environment variables
load_dotenv()

# Configure logging with more detail
logging.basicConfig(level=logging.INFO)
# search reddit forums
# Parameters:
# state: State - The current state of the graph
# Returns:
# State - The updated state with the reddit forums results
def search_reddit_forums(state: State) -> State:
    """Search the reddit forums for the user query"""

    user_query = state["messages"][-1].content
    conversation_history = state.get("conversation_history", "")
    
    keyword_search = llm_main.invoke([SystemMessage(content="Extract main keyword from the user query for searching posts within subreddit."), HumanMessage(content=state["messages"][-1].content)])
    search = keyword_search.content
    try:
        # Initialize Reddit client with proper error handling
        reddit_client = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        
        subreddit_name = "ravExchange"
        subreddit = reddit_client.subreddit(subreddit_name)
        results = []

        for post in subreddit.search(search, limit=5):
            results.append({
                "title": post.title,
                "url": post.url,
                "score": post.score,
                "num_comments": post.num_comments,
                "author": str(post.author) if post.author else "Unknown",
            })

        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "reddit_results": results
            }
        }
        
    except Exception as e:
        logging.error(f"Reddit search error: {e}")
        return {
            "messages": state["messages"],
            "message_type": state.get("message_type"),
            "result": {
                "error": f"Reddit search failed: {str(e)}"
            }
        }
