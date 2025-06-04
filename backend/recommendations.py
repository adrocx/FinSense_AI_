from flask import Blueprint, jsonify
import os
import requests
from news import get_news
import openai
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

recommendations_bp = Blueprint('recommendations', __name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_BASE_URL = "https://api.polygon.io"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# For demo, use a static list of tickers
TOP_TICKERS = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"]

# Simple in-memory cache
_recommendations_cache = {"data": None, "timestamp": 0}
CACHE_TTL = 60  # seconds

def get_polygon_data(ticker):
    try:
        # Get real-time price
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={POLYGON_API_KEY}"
        resp = requests.get(url)
        price = None
        if resp.status_code == 200 and resp.json().get("results"):
            price = resp.json()["results"][0]["c"]
        # Get company name
        url2 = f"{POLYGON_BASE_URL}/v3/reference/tickers/{ticker}?apiKey={POLYGON_API_KEY}"
        resp2 = requests.get(url2)
        name = ticker
        if resp2.status_code == 200 and resp2.json().get("results"):
            name = resp2.json()["results"].get("name", ticker)
        return {"ticker": ticker, "company_name": name, "price": price}
    except Exception as e:
        print(f"Polygon error for {ticker}: {e}")
        return {"ticker": ticker, "company_name": ticker, "price": None}

def get_groq_analysis(prompt):
    try:
        client = openai.OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "system", "content": "You are a financial research assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return "No AI analysis available."

def prompt_groq_for_recommendations(stocks_data):
    prompt = (
        "You are a world-class financial AI. Given the following real-time stock data and news, "
        "analyze and recommend the top 3 stocks to buy right now. For each, provide:\n"
        "- Ticker\n- Company Name\n- Sentiment Score (-1 to 1)\n- Short summary (1-2 sentences)\n"
        "Respond in JSON as an array of objects with keys: ticker, company_name, sentiment, summary.\n"
        "Here is the data:\n"
    )
    for stock in stocks_data:
        prompt += (
            f"\nTicker: {stock['ticker']}\n"
            f"Company: {stock['company_name']}\n"
            f"Price: {stock['price']}\n"
            f"News:\n"
        )
        for article in stock["news"]:
            prompt += f"- {article['title']} ({article['source']}): {article['content']}\n"
    # Call Groq (or OpenAI) API
    recs_str = get_groq_analysis(prompt)
    # Log the raw response for debugging
    print("Raw Groq response:\n", recs_str)
    # Clean the response: remove control characters except for newlines and tabs
    cleaned = ''.join(ch for ch in recs_str if ord(ch) >= 32 or ch in '\n\t')
    # Extract JSON array from the cleaned response
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except Exception as e:
            print(f"JSON parse error: {e}")
            print("Raw JSON string:", json_str)
            return []
    else:
        print("No JSON array found in Groq response.")
        print("Cleaned response:\n", cleaned)
        return []

def get_recommendations_cached():
    now = time.time()
    if _recommendations_cache["data"] and now - _recommendations_cache["timestamp"] < CACHE_TTL:
        return _recommendations_cache["data"]
    # Otherwise, recompute
    data = compute_recommendations()
    _recommendations_cache["data"] = data
    _recommendations_cache["timestamp"] = now
    return data

def compute_recommendations():
    with ThreadPoolExecutor() as executor:
        stock_futures = {executor.submit(get_polygon_data, ticker): ticker for ticker in TOP_TICKERS}
        news_futures = {executor.submit(get_news, ticker, 3): ticker for ticker in TOP_TICKERS}
        stocks_data = []
        news_data = {}
        for future in as_completed(stock_futures):
            ticker = stock_futures[future]
            try:
                stocks_data.append(future.result())
            except Exception:
                stocks_data.append({"ticker": ticker, "company_name": ticker, "price": None})
        for future in as_completed(news_futures):
            ticker = news_futures[future]
            try:
                news_data[ticker] = future.result()
            except Exception:
                news_data[ticker] = []
        for stock in stocks_data:
            stock["news"] = news_data.get(stock["ticker"], [])
    try:
        recs = prompt_groq_for_recommendations(stocks_data)
        return recs[:3]
    except Exception:
        fallback = []
        for s in stocks_data[:3]:
            fallback.append({
                "ticker": s["ticker"],
                "company_name": s.get("company_name", s["ticker"]),
                "sentiment": 0,
                "summary": "No AI analysis available."
            })
        return fallback

@recommendations_bp.route('/recommendations', methods=['GET'])
def recommendations():
    try:
        recs = get_recommendations_cached()
        return jsonify(recs)
    except Exception as e:
        return jsonify([]), 500 