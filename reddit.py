import praw
from dotenv import load_dotenv
import os
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

subreddit = reddit.subreddit("ravExchange")
for post in subreddit.search("disco lines", limit=5):
    print(post.title)
    print(post.url)
    print(post.score)
    print(post.num_comments)
    print(post.author)
    print(post.created_utc)
    print(post.selftext)
    print(post.permalink)