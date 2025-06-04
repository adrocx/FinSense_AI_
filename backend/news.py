from flask import Blueprint, jsonify
import os
import requests
from urllib.parse import quote
import logging
import time
from functools import lru_cache
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Cache configuration
NEWS_CACHE_DURATION = 300  # 5 minutes
NEWS_RATE_LIMIT_DELAY = 1  # 1 second between requests
last_request_time = 0

news_bp = Blueprint('news', __name__)

def validate_api_key():
    """Validate that the NewsAPI key is present and properly formatted"""
    if not NEWS_API_KEY:
        logger.error("NewsAPI key is missing")
        return False
    if not isinstance(NEWS_API_KEY, str) or len(NEWS_API_KEY) < 10:
        logger.error("NewsAPI key appears to be invalid")
        return False
    return True

@lru_cache(maxsize=100)
def get_cached_news(query: str, timestamp: int) -> list:
    """Get cached news articles for a query"""
    return []

def get_news(query: str, limit: int = 5):
    """Fetch news articles for a given query using News API"""
    global last_request_time
    
    # Check cache first
    current_time = int(time.time())
    cache_key = f"{query}_{current_time // NEWS_CACHE_DURATION}"
    cached_news = get_cached_news(cache_key, current_time)
    if cached_news:
        return cached_news

    if not validate_api_key():
        logger.error("Cannot fetch news: Invalid API key")
        return get_demo_news()

    try:
        # Rate limiting
        time_since_last = time.time() - last_request_time
        if time_since_last < NEWS_RATE_LIMIT_DELAY:
            time.sleep(NEWS_RATE_LIMIT_DELAY - time_since_last)

        # URL encode the query parameter
        encoded_query = quote(query)
        url = f"{NEWS_API_URL}?q={encoded_query}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&language=en"
        
        logger.info(f"Fetching news for query: {query}")
        response = requests.get(url, timeout=10)
        last_request_time = time.time()
        
        if response.status_code == 401:
            logger.error("NewsAPI authentication failed - check API key")
            return get_demo_news()
        elif response.status_code == 429:
            logger.error("NewsAPI rate limit exceeded")
            return get_demo_news()
        elif response.status_code != 200:
            logger.error(f"NewsAPI request failed with status {response.status_code}")
            return get_demo_news()

        try:
            data = response.json()
            if "status" in data and data["status"] == "error":
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return get_demo_news()
                
            articles = data.get("articles", [])[:limit]
            news_list = [
                {
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "timestamp": a.get("publishedAt", ""),
                    "content": a.get("description") or a.get("content") or "",
                    "url": a.get("url", ""),
                    "sentiment": 0,  # Placeholder for future sentiment analysis
                    "credibility": 3  # Default credibility
                }
                for a in articles
            ]
            
            # Cache the results
            get_cached_news.cache_clear()  # Clear old cache entries
            get_cached_news(cache_key, current_time)
            
            return news_list
        except ValueError as e:
            logger.error(f"Failed to parse NewsAPI response: {str(e)}")
            return get_demo_news()
            
    except requests.RequestException as e:
        logger.error(f"Network error while fetching news for {query}: {str(e)}")
        return get_demo_news()
    except Exception as e:
        logger.error(f"Unexpected error fetching news for {query}: {str(e)}")
        return get_demo_news()

def get_demo_news():
    """Return demo news articles when API fails"""
    return [
        {
            "title": "Demo Market Rally",
            "source": "DemoSource",
            "timestamp": datetime.now().isoformat(),
            "content": "Stocks rallied today as investors cheered strong earnings.",
            "url": "https://example.com",
            "sentiment": 0.5,
            "credibility": 4
        },
        {
            "title": "Demo Fed Rate Decision",
            "source": "DemoSource",
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "content": "The Federal Reserve held rates steady, citing stable inflation.",
            "url": "https://example.com",
            "sentiment": 0.1,
            "credibility": 3
        },
        {
            "title": "Demo Tech Stocks Surge",
            "source": "DemoSource",
            "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            "content": "Tech stocks led the market higher on new AI breakthroughs.",
            "url": "https://example.com",
            "sentiment": 0.7,
            "credibility": 5
        }
    ]

@news_bp.route('/news', methods=['GET'])
def news():
    """Get latest general financial news (not filtered by ticker)"""
    try:
        news_list = get_news('finance', limit=10)
        return jsonify(news_list)
    except Exception as e:
        logger.error(f"Error in news endpoint: {str(e)}")
        return jsonify(get_demo_news()) 