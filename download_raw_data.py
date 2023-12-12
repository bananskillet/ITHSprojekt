import settings
import re
import os
import csv
from googleapiclient.discovery import build

# Clean youtube title from special characters 
def clean_title(title_input, video_data):
    pattern = r'[^\w\s\.\!\?\,\&\-]'

    title = str(title_input).replace("‘", "'").replace("’", "'")
    cleaned_title = re.sub(pattern, '', title)

    return cleaned_title

def get_channel_videos(channel_id, api_key, filename):
    # Build a youtube client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Get the playlist of youtube channel
    channel_response = youtube.channels().list(id=channel_id, part='contentDetails').execute()
    playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # Fetch videos 
    videos = []
    next_page_token = None
    while True:
        pl_response = youtube.playlistItems().list(playlistId=playlist_id, 
                                                   part='snippet', 
                                                   maxResults=10000,
                                                   pageToken=next_page_token).execute()

        videos += pl_response['items']
        next_page_token = pl_response.get('nextPageToken')

        if not next_page_token:
            break

    # Write video details to csv-file
    with open(filename, mode='a', newline='', encoding='utf8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Title', 'Author', 'Channel_ID', 'URL', 'Video_ID','Language','Textfile_Original', 'Sentiment', 'Phrases', 'Keywords','Timestamp'])
        
        for video in videos:
            video_data = video['snippet']
            title_temp = video_data['title']
            title = clean_title(title_temp, video_data)
            author = video_data['channelTitle']
            channel = channel_id
            video_id = video_data['resourceId']['videoId']
            url = f'https://www.youtube.com/watch?v={video_id}'

            writer.writerow([title, author, channel, url, video_id])

    print(f"Data for {len(videos)} videos written to {filename}")


os.chdir('data_raw')
youtube_api_key = settings.YOUTUBE_API_KEY
filename = input('Enter filename: ')
filename = filename + '.csv'
youtube_channel_id = input('Enter youtube channel id: ') 
get_channel_videos(youtube_channel_id, youtube_api_key, filename)