import requests
import logging

def search_reddit_public(query: str, limit: int = 5):
    """Search Reddit using the public JSON API (no authentication required)"""
    try:
        url = "https://www.reddit.com/search.json"
        params = {
            'q': query,
            'limit': limit,
            'sort': 'relevance',
            't': 'all'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        posts = []
        
        for post in data['data']['children']:
            post_data = post['data']
            posts.append({
                'title': post_data['title'],
                'url': f"https://reddit.com{post_data['permalink']}",
                'subreddit': post_data['subreddit'],
                'score': post_data['score'],
                'num_comments': post_data['num_comments'],
                'author': post_data['author'],
                'created_utc': post_data['created_utc']
            })
        
        return posts
        
    except Exception as e:
        logging.error(f"Error searching Reddit: {e}")
        return []

if __name__ == "__main__":
    # Test the public Reddit search
    results = search_reddit_public("python programming", 3)
    for i, post in enumerate(results, 1):
        print(f"{i}. {post['title']}")
        print(f"   Subreddit: r/{post['subreddit']}")
        print(f"   Score: {post['score']} | Comments: {post['num_comments']}")
        print(f"   URL: {post['url']}")
        print()
