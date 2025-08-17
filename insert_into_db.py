from supabase import create_client, Client
import csv
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# CSV for songs
csv_file_path = 'refined_songs.csv'

with open(csv_file_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    # headers are title,genre,artists,song_desc,youtube_url
    for row in reader:
        title = row['title']
        genre = row['genre']
        artists = row['artists']
        song_desc = row['song_desc']
        youtube_url = row['youtube_url']

        # Print or process
        song_data = {
            "title": title,
            "genre": genre,
            "artists": artists,
            "song_desc": song_desc,
            "youtube_url": youtube_url
        }
        try: 
            supabase.table("Songs").insert(song_data).execute()
            print(f"Inserted song: {title}")
        except Exception as e:
            print(f"Error inserting song: {e}")
