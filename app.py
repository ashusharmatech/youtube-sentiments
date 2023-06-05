from flask import Flask, request, jsonify, render_template
import googleapiclient.discovery
from textblob import TextBlob
import nltk
import re
from flask_cors import CORS
import argparse
import os
import logging


# Parse command-line arguments
parser = argparse.ArgumentParser(description='YouTube Comment Sentiment Analysis')
parser.add_argument('--api_key', help='Your YouTube Data API key')
args = parser.parse_args()


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
CORS(app)


# Set up the YouTube API client
api_service_name = "youtube"
api_version = "v3"
api_key = args.api_key

youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=api_key)


def extract_video_id(url):
    # Extract the video ID from the YouTube URL
    video_id = None
    if 'youtube.com' in url:
        video_id = re.findall(r'v=([a-zA-Z0-9_-]+)', url)
    elif 'youtu.be' in url:
        video_id = re.findall(r'youtu.be/([a-zA-Z0-9_-]+)', url)
    if video_id:
        return video_id[0]
    else:
        return None


def get_video_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100
    )

    try:
        while request:
            response = request.execute()
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            if "nextPageToken" in response:
                request = youtube.commentThreads().list_next(request, response)
            else:
                request = None

    except Exception as e:
        logging.error("Error while fetching comments: %s", str(e))
        raise

    return comments




def analyze_comment_sentiment(comment):
    blob = TextBlob(comment)
    sentiment = blob.sentiment.polarity
    return sentiment


def get_comments_sentiment(video_id):
    try:
        comments = get_video_comments(video_id)
        sentiments = [analyze_comment_sentiment(comment) for comment in comments]
        return sentiments

    except Exception as e:
        logging.error("Error while analyzing sentiment: %s", str(e))
        raise

def calculate_sentiment_score(sentiments):
    total_sentiment = sum(sentiments)
    score = total_sentiment / len(sentiments)
    return score

# Define the route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        video_url = request.json['video_url']

        # Extract the YouTube video ID from the URL
        video_id = extract_video_id(video_url)

        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'})

        # Fetch comments and perform sentiment analysis
        sentiments = get_comments_sentiment(video_id)
        score = calculate_sentiment_score(sentiments)

        # Prepare response
        response = {
            'success': True,
            'message': 'Sentiment analysis completed successfully.',
            'score': score,
            'sentiments': sentiments
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Get the port number from the environment variable (or use a default value)
    port = int(os.environ.get('PORT', 5000))

    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=port)