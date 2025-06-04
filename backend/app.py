from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any
import time
import requests
import openai
from functools import wraps, lru_cache
from dotenv import load_dotenv
from groq import Groq
import json
from dateutil import parser as date_parser
from portfolio import calculate_portfolio_metrics
from quarterly import get_quarterly_recommendations
from pypfopt import EfficientFrontier, risk_models, expected_returns
from polygon import RESTClient
from news import news_bp  # Import the news blueprint
from recommendations import recommendations_bp  # Import the recommendations blueprint
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

app.register_blueprint(news_bp)  # Register the news blueprint
app.register_blueprint(recommendations_bp)  # Register the recommendations blueprint

# Add Groq API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key=GROQ_API_KEY)

# News API configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Cache for fundamental data (ticker: {data: ..., timestamp: ...})
fundamental_cache = {}
CACHE_DURATION = 600 # Cache for 10 minutes (in seconds)

# Add rate limiting for yfinance
RATE_LIMIT_DELAY = 2  # Reduced delay between requests to 2 seconds
last_request_time = 0

# Add a per-ticker cache for sentiment
sentiment_cache = {}  # {ticker: {"result": ..., "timestamp": ...}}
CACHE_DURATION = 300  # 5 minutes in seconds

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
polygon_client = RESTClient(POLYGON_API_KEY)

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# Add rate limiting cache
polygon_cache = {}
POLYGON_CACHE_DURATION = 300  # 5 minutes
POLYGON_RATE_LIMIT_DELAY = 12  # 12 seconds between requests

# Polygon API configuration
POLYGON_BASE_URL = "https://api.polygon.io/v2"

def rate_limited_request(delay=5):
    def decorator(func):
        last_request_time = 0
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_request_time
            current_time = time.time()
            time_since_last_request = current_time - last_request_time
            if time_since_last_request < delay:
                time.sleep(delay - time_since_last_request)
            try:
                result = func(*args, **kwargs)
                last_request_time = time.time()
                return result
            except Exception as e:
                raise e
        return wrapper
    return decorator

@app.route('/', methods=['GET'])
def root():
    """Root endpoint to verify API is running"""
    return jsonify({
        "status": "ok",
        "message": "API is running",
        "endpoints": {
            "dashboard": "/dashboard",
            "sentiment": "/sentiment",
            "fundamental": "/fundamental",
            "portfolio": "/portfolio",
            "quarterly": "/quarterly"
        }
    })

@rate_limited_request
def get_stock_data(ticker: str) -> Dict[str, Any]:
    """Get real-time stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1mo")
        
        if hist.empty:
            print(f"No price data found for {ticker}, using fallback data")
            # Generate fallback data for SPY
            if ticker == "SPY":
                return {
                    "company_name": "SPDR S&P 500 ETF Trust",
                    "sector": "ETF",
                    "market_cap": "$400,000,000,000",
                    "eps": "$20.00",
                    "pe_ratio": "25.00",
                    "revenue": "$0.00",
                    "price": 500.00,
                    "change": 0.5,
                    "volume": 1000000
                }
            return {}
        
        return {
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "market_cap": f"${info.get('marketCap', 0):,.2f}",
            "eps": f"${info.get('trailingEps', 0):.2f}",
            "pe_ratio": f"{info.get('trailingPE', 0):.2f}",
            "revenue": f"${info.get('totalRevenue', 0):,.2f}",
            "price": info.get("currentPrice", 0),
            "change": info.get("regularMarketChangePercent", 0),
            "volume": info.get("regularMarketVolume", 0)
        }
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {str(e)}")
        return {}

# Helper for Groq completions
def groq_generate_content(prompt, max_tokens=300, temperature=0.3, retries=3, delay=1):
    for attempt in range(retries):
        try:
            print(f"[DEBUG] Attempting Groq API call (attempt {attempt + 1}/{retries})")
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=temperature,
                max_tokens=max_tokens
            )
            text = response.choices[0].message.content
            print(f"[DEBUG] Groq API call successful, response length: {len(text)}")
            return text
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Groq API call failed (attempt {attempt + 1}/{retries}): {error_msg}")
            if '429' in error_msg or 'rate limit' in error_msg.lower():
                print(f"Groq rate limit hit, retrying in {delay} seconds...")
                import time
                time.sleep(delay)
                delay *= 2
                continue
            if attempt == retries - 1:  # Last attempt
                print(f"[ERROR] Groq API call failed after {retries} attempts. Last error: {error_msg}")
                return f"Unable to generate AI analysis due to: {error_msg}"
            import time
            time.sleep(delay)
            delay *= 2
    print("[ERROR] Groq retries exhausted")
    return "Unable to generate AI analysis after multiple attempts"

def get_news(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch news articles for a given ticker using News API"""
    try:
        url = f"{NEWS_API_URL}?q={ticker}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&language=en"
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])[:limit]
            return [
                {
                    "title": a["title"],
                    "source": a["source"]["name"],
                    "timestamp": a["publishedAt"],
                    "content": a.get("description") or a.get("content") or "",
                    "url": a["url"],
                    "sentiment": 0,  # Will be updated by AI
                    "credibility": 3  # Default credibility
                }
                for a in articles
            ]
        return []
    except Exception as e:
        print(f"Error fetching news for {ticker}: {str(e)}")
        return []

def get_sentiment_timeline(ticker: str) -> List[Dict[str, Any]]:
    """Get 10-year sentiment timeline based on historical price data, with fallback demo data if unavailable."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="10y", interval="1mo")
        
        if hist.empty:
            print(f"No historical data found for {ticker}, using fallback demo data.")
            # Fallback: generate 10 years (120 months) of demo data
            from datetime import datetime, timedelta
            base_date = datetime.now() - timedelta(days=365*10)
            timeline = []
            price = 100.0
            for i in range(120):
                date = (base_date + timedelta(days=30*i)).strftime("%Y-%m")
                # Demo: sine wave sentiment, price up and down
                import math
                sentiment = math.sin(i/12*2*math.pi)  # yearly cycle
                label = "Bullish" if sentiment > 0.2 else "Bearish" if sentiment < -0.2 else "Neutral"
                price += sentiment * 2  # small price movement
                timeline.append({
                    "timestamp": date,
                    "price": round(price, 2),
                    "score": round(sentiment, 2),
                    "label": label
                })
            return timeline
        # Calculate monthly returns and sentiment scores
        hist['Returns'] = hist['Close'].pct_change()
        hist['Sentiment'] = hist['Returns'].apply(lambda x: 
            1 if x > 0.05 else  # Strong positive
            -1 if x < -0.05 else  # Strong negative
            0.5 if x > 0 else  # Slightly positive
            -0.5 if x < 0 else  # Slightly negative
            0  # Neutral
        )
        # Convert to list of monthly data points
        timeline = []
        for date, row in hist.iterrows():
            if pd.notna(row['Sentiment']):  # Skip first row (NaN due to pct_change)
                timeline.append({
                    "timestamp": date.strftime("%Y-%m"),
                    "price": float(row['Close']),
                    "score": float(row['Sentiment']),
                    "label": "Bullish" if row['Sentiment'] > 0 else "Bearish" if row['Sentiment'] < 0 else "Neutral"
                })
        return timeline
    except Exception as e:
        print(f"Error getting sentiment timeline for {ticker}: {str(e)}. Using fallback demo data.")
        # Fallback: same as above
        from datetime import datetime, timedelta
        base_date = datetime.now() - timedelta(days=365*10)
        timeline = []
        price = 100.0
        for i in range(120):
            date = (base_date + timedelta(days=30*i)).strftime("%Y-%m")
            import math
            sentiment = math.sin(i/12*2*math.pi)
            label = "Bullish" if sentiment > 0.2 else "Bearish" if sentiment < -0.2 else "Neutral"
            price += sentiment * 2
            timeline.append({
                "timestamp": date,
                "price": round(price, 2),
                "score": round(sentiment, 2),
                "label": label
            })
        return timeline

@app.route('/sentiment', methods=['POST'])
def sentiment():
    """Analyze sentiment for a given ticker"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({"error": "No ticker provided"}), 400

        # 1. Get news articles
        news = get_news(ticker, limit=5)
        # Patch: If no news, use demo news for popular tickers
        if not news and ticker in ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA"]:
            news = [
                {"title": f"{ticker} demo headline 1", "source": "DemoSource", "timestamp": "2024-05-23", "content": f"This is a demo news article for {ticker}.", "url": "https://example.com", "sentiment": 0.2, "credibility": 4},
                {"title": f"{ticker} demo headline 2", "source": "DemoSource", "timestamp": "2024-05-22", "content": f"Another demo news article for {ticker}.", "url": "https://example.com", "sentiment": -0.1, "credibility": 3},
                {"title": f"{ticker} demo headline 3", "source": "DemoSource", "timestamp": "2024-05-21", "content": f"Yet another demo news article for {ticker}.", "url": "https://example.com", "sentiment": 0.5, "credibility": 5}
            ]
        if not news:
            return jsonify({
                "error": "No news articles found for this ticker",
                "overall_sentiment": {"score": 0, "label": "Neutral"},
                "sentiment_timeline": [],
                "key_findings": ["No recent news available for analysis"],
                "recommendations": ["Unable to provide recommendations without recent news"],
                "top_news": []
            })

        # 2. Get sentiment timeline
        sentiment_timeline = get_sentiment_timeline(ticker)

        # 2b. Get 5 years of annual revenue and net income for PnL Growth
        pnl_growth = []
        try:
            stock = yf.Ticker(ticker)
            financials = stock.financials
            print(f"[DEBUG] yfinance financials for {ticker}:")
            print(financials)
            if not financials.empty:
                years = list(financials.columns)[-5:][::-1]
                for year in years:
                    revenue_keys = ['Total Revenue', 'TotalRevenue', 'Revenue']
                    net_income_keys = ['Net Income', 'NetIncome']
                    revenue = None
                    net_income = None
                    for rk in revenue_keys:
                        if rk in financials.index:
                            revenue = financials.at[rk, year]
                            break
                    for nk in net_income_keys:
                        if nk in financials.index:
                            net_income = financials.at[nk, year]
                            break
                    pnl_growth.append({
                        'year': str(year.year) if hasattr(year, 'year') else str(year),
                        'revenue': float(revenue) if revenue is not None and revenue == revenue else None,
                        'net_income': float(net_income) if net_income is not None and net_income == net_income else None
                    })
                pnl_growth = pnl_growth[::-1]
            else:
                # Fallback: try income_stmt
                income_stmt = stock.income_stmt
                print(f"[DEBUG] yfinance income_stmt for {ticker}:")
                print(income_stmt)
                if not income_stmt.empty:
                    years = list(income_stmt.columns)[-5:][::-1]
                    for year in years:
                        revenue = income_stmt.at['Total Revenue', year] if 'Total Revenue' in income_stmt.index else None
                        net_income = income_stmt.at['Net Income', year] if 'Net Income' in income_stmt.index else None
                        pnl_growth.append({
                            'year': str(year.year) if hasattr(year, 'year') else str(year),
                            'revenue': float(revenue) if revenue is not None and revenue == revenue else None,
                            'net_income': float(net_income) if net_income is not None and net_income == net_income else None
                        })
                    pnl_growth = pnl_growth[::-1]
            print(f"[DEBUG] pnl_growth for {ticker}: {pnl_growth}")
        except Exception as e:
            print(f"Error fetching PnL growth for {ticker}: {str(e)}")
            pnl_growth = []

        # 2c. Get price history using Polygon
        price_history_10y = []
        try:
            from datetime import date, timedelta
            today = date.today()
            ten_years_ago = today - timedelta(days=365*10)
            price_history = get_polygon_history_batch([ticker], ten_years_ago, today, POLYGON_API_KEY)
            
            if not price_history.empty and ticker in price_history:
                for date, price in price_history[ticker].items():
                    # Format date as "MMM YYYY" (e.g., "Jan 2020")
                    price_history_10y.append({
                        "timestamp": date.strftime("%b %Y"),
                        "price": float(price)
                    })
            else:
                print(f"No price history found for {ticker} in Polygon, using demo data")
                # Generate demo data
                base_date = today - timedelta(days=365*10)
                price = 100.0
                import math
                for i in range(120):
                    d = (base_date + timedelta(days=30*i))
                    sentiment = math.sin(i/12*2*math.pi)
                    price += sentiment * 2
                    price_history_10y.append({
                        "timestamp": d.strftime("%b %Y"),  # Format as "MMM YYYY"
                        "price": round(price, 2)
                    })
        except Exception as e:
            print(f"Error fetching price history for {ticker}: {str(e)}")
            # Generate demo data on error
            from datetime import date, timedelta
            base_date = date.today() - timedelta(days=365*10)
            price = 100.0
            import math
            for i in range(120):
                d = (base_date + timedelta(days=30*i))
                sentiment = math.sin(i/12*2*math.pi)
                price += sentiment * 2
                price_history_10y.append({
                    "timestamp": d.strftime("%b %Y"),  # Format as "MMM YYYY"
                    "price": round(price, 2)
                })

        # 3. Prepare Groq prompt with top 3 high-impact news
        news_for_groq = news[:3]
        news_text = "\n\n".join([
            f"Title: {a.get('title','')}. Source: {a.get('source','')}. Content: {a.get('content','')}" for a in news_for_groq
        ])
        prompt = f"""
You are a financial AI assistant. Given the following recent news about {ticker}, assess the overall sentiment for investors (score -1 to 1, label), and provide:
1. Sentiment score (number between -1 and 1)
2. Sentiment label (Bullish, Bearish, Neutral, etc.)
3. Three key points investors should consider (as a JSON array of strings)
4. Three actionable recommendations for investors (as a JSON array of strings)

News:
{news_text}

Respond in JSON with keys: sentiment_score, sentiment_label, key_points, recommendations.
"""
        groq_response = groq_generate_content(prompt, max_tokens=300, temperature=0.3)
        import re, json
        match = re.search(r'\{[\s\S]*\}', groq_response)
        if not match:
            # Groq failed or rate-limited
            return jsonify({
                "overall_sentiment": {"score": 0, "label": "Neutral"},
                "sentiment_timeline": sentiment_timeline,
                "pnl_growth": pnl_growth,
                "price_history_10y": price_history_10y,
                "key_findings": ["AI service is currently rate-limited or failed, please try again later."],
                "recommendations": ["Unable to provide recommendations at this time."],
                "top_news": news
            })
        try:
            parsed = json.loads(match.group(0))
            score = float(parsed.get("sentiment_score", 0))
            label = parsed.get("sentiment_label", "Neutral")
            key_findings = parsed.get("key_points", [])
            recommendations = parsed.get("recommendations", [])
            
            # Update news articles with sentiment
            for article in news:
                article["sentiment"] = score  # Use overall sentiment for all articles
            
            return jsonify({
                "overall_sentiment": {"score": score, "label": label},
                "sentiment_timeline": sentiment_timeline,
                "pnl_growth": pnl_growth,
                "price_history_10y": price_history_10y,
                "key_findings": key_findings,
                "recommendations": recommendations,
                "top_news": news
            })
        except Exception as e:
            print(f"Error parsing Groq response: {str(e)}")
            return jsonify({
                "overall_sentiment": {"score": 0, "label": "Neutral"},
                "sentiment_timeline": sentiment_timeline,
                "pnl_growth": pnl_growth,
                "price_history_10y": price_history_10y,
                "key_findings": ["Error analyzing sentiment"],
                "recommendations": ["Unable to provide recommendations at this time"],
                "top_news": news
            })
    except Exception as e:
        print(f"Error in AI-driven sentiment analysis: {str(e)}")
        return jsonify({
            "error": "Error analyzing sentiment",
            "details": str(e)
        }), 500

@app.route('/fundamental', methods=['POST'])
def fundamental():
    """Get fundamental analysis data"""
    data = request.get_json()
    ticker = data.get('ticker')

    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    # Check cache first
    if ticker in fundamental_cache and (time.time() - fundamental_cache[ticker]['timestamp']) < CACHE_DURATION:
        print(f"Returning cached fundamental data for {ticker}")
        return jsonify(fundamental_cache[ticker]['data'])

    # Define rich fallback data at the start
    fallback = {
        "company_info": {
            "name": f"{ticker} Corporation",
            "symbol": ticker,
            "exchange": "NASDAQ",
            "sector": "Technology",
            "industry": "Software",
            "country": "USA",
            "website": f"https://www.{ticker.lower()}.com"
        },
        "key_metrics": {
            "current_price": 250.0,
            "pe_ratio": 28.0,
            "market_cap": 1.5e12,
            "eps_ttm": 8.5,
            "dividend_yield": 0.9,
            "beta": 1.1,
            "52_week_high": 300.0,
            "52_week_low": 200.0,
            "volume": 20000000,
            "avg_volume": 25000000
        },
        "ai_insights": {
            "revenue_growth": f"{ticker} has shown steady revenue growth over the past 5 years, with a CAGR of 10.5%.",
            "profitability": f"Gross margin is 65.2%, with operating margin at 35.1%.",
            "balance_sheet_strength": f"Debt-to-equity ratio is 0.45. Cash reserves are strong at $120B.",
            "valuation_assessment": f"{ticker} trades at a P/E of 28, in line with sector averages."
        },
        "financial_ratios": [
            {"metric": "Liquidity", "score": 60},
            {"metric": "Leverage", "score": 70},
            {"metric": "Efficiency", "score": 65},
            {"metric": "Profitability", "score": 80},
            {"metric": "Market Value", "score": 55}
        ],
        "competitive_analysis": [
            {"symbol": ticker, "pe_ratio": 28.0, "revenue_growth": 10.5, "gross_margin": 65.2, "roe": 40.0, "dividend_yield": 0.9},
            {"symbol": "AAPL", "pe_ratio": 29.5, "revenue_growth": 8.1, "gross_margin": 43.8, "roe": 147.9, "dividend_yield": 0.52},
            {"symbol": "GOOGL", "pe_ratio": 25.1, "revenue_growth": 15.2, "gross_margin": 56.9, "roe": 28.6, "dividend_yield": 0.0}
        ],
        "financial_statements": {
            "income_statement": [
                {"period": "2025 Q1", "Revenue": 50000, "Gross Profit": 32500, "Operating Income": 17500, "Net Income": 14000, "EPS (Diluted)": 2.10},
                {"period": "2024 Q4", "Revenue": 48000, "Gross Profit": 31000, "Operating Income": 16000, "Net Income": 13000, "EPS (Diluted)": 1.95},
                {"period": "2024 Q3", "Revenue": 47000, "Gross Profit": 30500, "Operating Income": 15500, "Net Income": 12500, "EPS (Diluted)": 1.85}
            ],
            "balance_sheet": [
                {"period": "2025 Q1", "Total Assets": 350000, "Total Liabilities": 210000, "Shareholder Equity": 140000},
                {"period": "2024 Q4", "Total Assets": 340000, "Total Liabilities": 205000, "Shareholder Equity": 135000}
            ],
            "cash_flow": [
                {"period": "2025 Q1", "Net Cash from Operations": 18000, "Net Cash from Investing": -4000, "Net Cash from Financing": -6000},
                {"period": "2024 Q4", "Net Cash from Operations": 17000, "Net Cash from Investing": -3500, "Net Cash from Financing": -5500}
            ]
        }
    }

    try:
        # Add delay before making request
        time.sleep(RATE_LIMIT_DELAY)
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
            if info:
                fallback["company_info"].update({
                    "name": info.get("longName", fallback["company_info"]["name"]),
                    "symbol": ticker,
                    "exchange": info.get("exchange", fallback["company_info"]["exchange"]),
                    "sector": info.get("sector", fallback["company_info"]["sector"]),
                    "industry": info.get("industry", fallback["company_info"]["industry"]),
                    "country": info.get("country", fallback["company_info"]["country"]),
                    "website": info.get("website", fallback["company_info"]["website"]),
                })
                fallback["key_metrics"].update({
                    "current_price": info.get("currentPrice", fallback["key_metrics"]["current_price"]),
                    "pe_ratio": info.get("trailingPE", fallback["key_metrics"]["pe_ratio"]),
                    "market_cap": info.get("marketCap", fallback["key_metrics"]["market_cap"]),
                    "eps_ttm": info.get("trailingEps", fallback["key_metrics"]["eps_ttm"]),
                    "dividend_yield": info.get("dividendYield", fallback["key_metrics"]["dividend_yield"]),
                    "beta": info.get("beta", fallback["key_metrics"]["beta"]),
                    "52_week_high": info.get("fiftyTwoWeekHigh", fallback["key_metrics"]["52_week_high"]),
                    "52_week_low": info.get("fiftyTwoWeekLow", fallback["key_metrics"]["52_week_low"]),
                    "volume": info.get("volume", fallback["key_metrics"]["volume"]),
                    "avg_volume": info.get("averageVolume", fallback["key_metrics"]["avg_volume"])
                })
        except Exception as e:
            print(f"Error getting stock info for {ticker}: {str(e)}")
            # Continue with fallback data
        # --- Financial Ratios Calculation ---
        def safe_score(val, min_val, max_val):
            if val is None or val == 'N/A':
                return 50
            try:
                val = float(val)
                return max(0, min(100, int(100 * (val - min_val) / (max_val - min_val))))
            except Exception:
                return 50
        ratios = [
            {"metric": "Liquidity", "score": safe_score(info.get('currentRatio', 1.5), 0.5, 3)},
            {"metric": "Leverage", "score": safe_score(info.get('debtToEquity', 0.5), 2, 0)},  # Lower is better
            {"metric": "Efficiency", "score": safe_score(info.get('assetTurnover', 1), 0.2, 2)},
            {"metric": "Profitability", "score": safe_score(info.get('grossMargins', 0.3), 0, 1)},
            {"metric": "Market Value", "score": safe_score(info.get('trailingPE', 20), 10, 40)}
        ]
        # --- Competitive Analysis ---
        comp_tickers = [ticker] + [t for t in ['AAPL', 'MSFT', 'GOOGL'] if t != ticker]
        comp_data = []
        for comp in comp_tickers:
            try:
                cinfo = yf.Ticker(comp).info
                comp_data.append({
                    "symbol": comp,
                    "pe_ratio": cinfo.get('trailingPE', 'N/A'),
                    "revenue_growth": round(100 * cinfo.get('revenueGrowth', 0), 1) if cinfo.get('revenueGrowth') is not None else 'N/A',
                    "gross_margin": round(100 * cinfo.get('grossMargins', 0), 1) if cinfo.get('grossMargins') is not None else 'N/A',
                    "roe": round(100 * cinfo.get('returnOnEquity', 0), 1) if cinfo.get('returnOnEquity') is not None else 'N/A',
                    "dividend_yield": round(100 * cinfo.get('dividendYield', 0), 3) if cinfo.get('dividendYield') is not None else 'N/A',
                })
            except Exception:
                comp_data.append({
                    "symbol": comp,
                    "pe_ratio": 'N/A',
                    "revenue_growth": 'N/A',
                    "gross_margin": 'N/A',
                    "roe": 'N/A',
                    "dividend_yield": 'N/A',
                })
        fallback["financial_ratios"] = ratios
        fallback["competitive_analysis"] = comp_data
    except Exception as e:
        print(f"Error fetching fundamental data for {ticker}: {str(e)}")
        # Continue with fallback data

    # --- AI Financial Analysis with Groq ---
    news = get_news(ticker, limit=5)
    prompt = f"""
You are a financial analyst AI. Here is the company's fundamental data and recent news:

FUNDAMENTALS:
{json.dumps(fallback, indent=2)}

NEWS:
{json.dumps(news, indent=2)}

Analyze the company's financial health, growth, profitability, and any risks or opportunities mentioned in the news. Provide a deep, insightful summary of the company's current position. Conclude with a clear recommendation: is this stock a Buy, Sell, or Hold? Justify your answer with specifics from the data and news.
"""
    ai_analysis = groq_generate_content(prompt, max_tokens=512, temperature=0.2)
    print("Groq AI Analysis:", ai_analysis)
    # After the Groq call, set only ai_insights['financial_analysis'] if Groq returns a result
    if ai_analysis.strip():
        fallback["ai_insights"]["financial_analysis"] = ai_analysis.strip()
    # Do NOT update any other ai_insights fields with Groq output. Leave them as fallback/static text.

    fundamental_cache[ticker] = {'data': fallback, 'timestamp': time.time()}
    return jsonify(fallback)

def get_sector_performance():
    """Get current sector performance data"""
    sectors = {
        "XLK": "Technology",
        "XLF": "Financial",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLI": "Industrial",
        "XLP": "Consumer Staples",
        "XLY": "Consumer Discretionary",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate"
    }
    
    performance = []
    for etf, name in sectors.items():
        try:
            stock = yf.Ticker(etf)
            hist = stock.history(period="1mo")
            if not hist.empty:
                performance.append({
                    "name": name,
                    "performance": float((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100)
                })
        except Exception as e:
            print(f"Error fetching data for {etf}: {str(e)}")
            continue
    
    # If no sector data available, provide fallback data
    if not performance:
        performance = [
            {"name": "Technology", "performance": 0.0},
            {"name": "Financial", "performance": 0.0},
            {"name": "Healthcare", "performance": 0.0},
            {"name": "Energy", "performance": 0.0},
            {"name": "Industrial", "performance": 0.0}
        ]
    
    return performance

@app.route('/portfolio', methods=['POST'])
def portfolio():
    """Optimize portfolio based on holdings"""
    try:
        risk_tolerance = request.form.get('risk_tolerance', 'Moderate')
        if 'portfolio' not in request.files:
            return jsonify({"error": "No portfolio file provided"}), 400
        portfolio_file = request.files['portfolio']
        if portfolio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        try:
            df = pd.read_csv(portfolio_file)
            required_columns = ['ticker', 'amount', 'price']
            if not all(col in df.columns for col in required_columns):
                return jsonify({"error": "CSV must contain 'ticker', 'amount', and 'price' columns"}), 400
            total_worth = None
            if (df['ticker'].str.upper() == 'TOTAL_WORTH').any():
                total_row = df[df['ticker'].str.upper() == 'TOTAL_WORTH']
                try:
                    total_worth = float(total_row['price'].values[0])
                except Exception:
                    total_worth = None
                df = df[df['ticker'].str.upper() != 'TOTAL_WORTH']
            holdings = df[required_columns].to_dict('records')
        except Exception as e:
            return jsonify({"error": f"Error reading portfolio file: {str(e)}"}), 400

        # --- Current Portfolio Metrics ---
        metrics = calculate_portfolio_metrics(holdings, total_worth_from_csv=total_worth)
        if total_worth is not None:
            metrics['total_worth_from_csv'] = total_worth

        # --- Batch Fetch Historical Data for All Tickers ---
        import datetime
        today = datetime.date.today()
        one_year_ago = today - datetime.timedelta(days=365)
        data = get_alpha_vantage_history_batch([h['ticker'] for h in holdings], one_year_ago, today, ALPHA_VANTAGE_API_KEY)
        
        valid_holdings = []
        failed_tickers = []
        for h in holdings:
            ticker = h['ticker']
            if ticker in data.columns and not data[ticker].dropna().empty:
                valid_holdings.append(h)
            else:
                failed_tickers.append(ticker)

        if not valid_holdings:
            return jsonify({"error": "No valid tickers found for optimization"}), 400

        # Use only valid holdings for optimization
        tickers = [h['ticker'] for h in valid_holdings]
        amounts = [float(h['amount']) for h in valid_holdings]
        prices = [float(h['price']) for h in valid_holdings]
        
        # Get the data for valid tickers only
        data = data[tickers]
        
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # Create EfficientFrontier instance
        ef = EfficientFrontier(mu, S)
        
        # --- Portfolio Optimization Logic ---
        if risk_tolerance == 'Conservative':
            target_volatility = 0.10
        elif risk_tolerance == 'Aggressive':
            weights = ef.max_sharpe()
            target_volatility = None
        else:  # Moderate
            target_volatility = 0.15

        if target_volatility is not None:
            try:
                weights = ef.efficient_risk(target_volatility)
                used_volatility = target_volatility
                warning_vol = None
            except Exception as e:
                min_vol = ef.min_volatility()
                if isinstance(min_vol, dict) or hasattr(min_vol, 'get'):
                    min_vol_value = min_vol.get('fun', list(min_vol.values())[0])
                else:
                    min_vol_value = min_vol
                ef = EfficientFrontier(mu, S)
                try:
                    weights = ef.efficient_risk(min_vol_value)
                    used_volatility = min_vol_value
                    warning_vol = f"Requested target volatility {target_volatility} was too low. Used minimum achievable volatility {min_vol_value:.3f} instead."
                except Exception as e2:
                    weights = ef.max_sharpe()
                    used_volatility = None
                    warning_vol = f"Could not achieve target volatility. Using maximum Sharpe ratio portfolio instead."
        else:
            weights = ef.max_sharpe()
            used_volatility = None
            warning_vol = None

        cleaned_weights = ef.clean_weights()
        opt_perf = ef.portfolio_performance()
        total_value = sum([a * p for a, p in zip(amounts, prices)])

        # --- Build allocations ---
        current_allocation = []
        optimized_allocation = []
        for h in valid_holdings:
            ticker = h['ticker']
            amount = float(h['amount'])
            price = float(h['price'])
            value = amount * price
            pct = round(100 * value / total_value, 2) if total_value else 0
            try:
                info = yf.Ticker(ticker).info
                company_name = info.get('longName', ticker)
            except Exception:
                company_name = ticker
            current_allocation.append({
                'ticker': ticker,
                'company_name': company_name,
                'amount': amount,
                'value': value,
                'percentage': pct
            })

        for ticker, weight in cleaned_weights.items():
            try:
                info = yf.Ticker(ticker).info
                company_name = info.get('longName', ticker)
            except Exception:
                company_name = ticker
            optimized_allocation.append({
                'ticker': ticker,
                'company_name': company_name,
                'optimal_percentage': round(100 * weight, 2)
            })

        # --- Build actionable recommendations ---
        actions = []
        curr_pct_map = {c['ticker']: c['percentage'] for c in current_allocation}
        for opt in optimized_allocation:
            ticker = opt['ticker']
            curr_pct = curr_pct_map.get(ticker, 0)
            diff = opt['optimal_percentage'] - curr_pct
            if abs(diff) < 1:
                continue
            elif diff > 0:
                actions.append({
                    'action': 'Increase',
                    'ticker': ticker,
                    'company_name': opt['company_name'],
                    'details': f"Increase {ticker} by {round(diff,2)}% of portfolio"
                })
            else:
                actions.append({
                    'action': 'Reduce',
                    'ticker': ticker,
                    'company_name': opt['company_name'],
                    'details': f"Reduce {ticker} by {abs(round(diff,2))}% of portfolio"
                })

        # --- Enhanced AI Analysis ---
        import json
        portfolio_json = json.dumps({
            'holdings': holdings,
            'metrics': metrics,
            'total_worth': total_worth,
            'risk_tolerance': risk_tolerance,
            'optimized_allocation': optimized_allocation,
            'current_allocation': current_allocation
        }, indent=2)

        # Get market news and sentiment for portfolio stocks
        portfolio_news = []
        for ticker in tickers:
            try:
                news = get_news(ticker)
                if news:
                    portfolio_news.extend(news[:3])  # Get top 3 news items per stock
            except Exception:
                continue

        # Get market sector performance
        sector_performance = get_sector_performance()

        # Generate AI analysis with enhanced context
        groq_prompt = f"""
You are a world-class financial AI. Analyze the following portfolio and provide comprehensive insights:

PORTFOLIO DATA:
{portfolio_json}

MARKET CONTEXT:
- Risk Tolerance: {risk_tolerance}
- Current Market Conditions: {sector_performance}
- Recent News: {json.dumps(portfolio_news, indent=2)}

Please provide:
1. Portfolio Analysis:
   - Current risk profile and alignment with selected risk tolerance
   - Diversification assessment
   - Sector exposure analysis
   - Performance metrics evaluation

2. Market Analysis:
   - Current market conditions and trends
   - Sector performance insights
   - Market sentiment analysis

3. Stock-Specific Analysis:
   - Individual stock performance and outlook
   - News impact analysis
   - Risk factors and opportunities

4. Recommendations:
   - Portfolio optimization suggestions
   - New stock suggestions based on:
     * Current market trends
     * Sector opportunities
     * Risk tolerance alignment
     * Diversification needs
   - Specific actions to take

5. Risk Assessment:
   - Overall portfolio risk score (0-100)
   - Risk factors and mitigation strategies
   - Alignment with risk tolerance

Format the response in clear sections with actionable insights and specific recommendations.
"""
        ai_analysis = groq_generate_content(groq_prompt, max_tokens=1000, temperature=0.2)

        # --- Add AI-suggested tickers to optimized allocation ---
        import re
        # Extract tickers: uppercase words of 2-5 letters not already in optimized_allocation
        existing_opt_tickers = set([o['ticker'] for o in optimized_allocation])
        ticker_regex = re.compile(r'\b([A-Z]{2,5})\b')
        ai_suggested = set()
        for match in ticker_regex.finditer(ai_analysis):
            ticker = match.group(1)
            if ticker not in existing_opt_tickers and ticker not in ['ETF', 'AI']:
                ai_suggested.add(ticker)
        # Assign default allocation (e.g., 5% each, or split 15% among all new tickers)
        default_pct = 5
        if len(ai_suggested) > 0:
            total_default = min(15, len(ai_suggested) * default_pct)
            per_ticker_pct = round(total_default / len(ai_suggested), 2)
        else:
            per_ticker_pct = 0
        # Fetch price and add to optimized_allocation
        for ticker in ai_suggested:
            try:
                info = yf.Ticker(ticker).info
                price = info.get('regularMarketPrice')
                company_name = info.get('longName', ticker)
                shares = 0
                if price and total_value:
                    shares = int((per_ticker_pct / 100 * total_value) // price)
                optimized_allocation.append({
                    'ticker': ticker,
                    'company_name': company_name,
                    'optimal_percentage': per_ticker_pct,
                    'price': price,
                    'shares': shares
                })
            except Exception:
                optimized_allocation.append({
                    'ticker': ticker,
                    'company_name': ticker,
                    'optimal_percentage': per_ticker_pct,
                    'price': None,
                    'shares': 0
                })
        # Also add price and shares for existing optimized tickers
        for o in optimized_allocation:
            if 'price' not in o or o['price'] is None:
                try:
                    info = yf.Ticker(o['ticker']).info
                    price = info.get('regularMarketPrice')
                    o['price'] = price
                except Exception:
                    o['price'] = None
            if 'shares' not in o or o['shares'] is None:
                try:
                    if o['price'] and total_value:
                        o['shares'] = int((o['optimal_percentage'] / 100 * total_value) // o['price'])
                    else:
                        o['shares'] = 0
                except Exception:
                    o['shares'] = 0

        # --- Response ---
        response = {
            'current_portfolio': {
                'total_value': metrics['total_value'],
                'annualized_return': metrics['expected_return'],
                'risk_level': metrics['risk_level'],
                'asset_allocation': [
                    {'category': sector, 'percentage': percentage}
                    for sector, percentage in metrics['sector_exposure'].items()
                ],
                'top_holdings': current_allocation[:5]
            },
            'recommended_optimization': {
                'projected_value': total_value * (1 + opt_perf[0]),
                'projected_return': round(opt_perf[0] * 100, 2),
                'optimized_risk_level': risk_tolerance,
                'recommended_asset_allocation': [
                    {'category': sector, 'percentage': percentage}
                    for sector, percentage in metrics['sector_exposure'].items()
                ],
                'recommended_actions': actions,
                'optimized_allocation': optimized_allocation,
                'tax_loss_harvesting': []
            },
            'warning': warning_vol,
            'ai_analysis': ai_analysis,
            'market_context': {
                'sector_performance': sector_performance,
                'portfolio_news': portfolio_news
            }
        }

        if total_worth is not None:
            response['total_worth_from_csv'] = total_worth

        if failed_tickers:
            response['warning'] = f"Warning: Could not fetch data for: {', '.join(failed_tickers)}. These tickers were excluded from optimization."
        if warning_vol:
            response['volatility_warning'] = warning_vol

        return jsonify(response)
    except Exception as e:
        print(f"Error in portfolio optimization: {str(e)}")
        return jsonify({
            'error': 'Error optimizing portfolio',
            'details': str(e)
        }), 500

@app.route('/quarterly', methods=['GET'])
def quarterly():
    """Get quarterly market analysis"""
    try:
        # Get market data
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="3mo")
        
        if spy_hist.empty:
            print("No SPY data available, using fallback data")
            spy_return = 0.0  # Fallback value
        else:
            spy_return = (spy_hist["Close"].iloc[-1] / spy_hist["Close"].iloc[0] - 1) * 100
        
        # Get sector performance
        sectors = ["XLK", "XLF", "XLV", "XLE", "XLI"]  # Tech, Financial, Healthcare, Energy, Industrial
        sector_performance = []
        
        for sector in sectors:
            try:
                etf = yf.Ticker(sector)
                hist = etf.history(period="3mo")
                if not hist.empty:
                    performance = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                    sector_performance.append({
                        "name": sector,
                        "performance": performance
                    })
            except Exception as e:
                print(f"Error fetching data for {sector}: {str(e)}")
                continue
        
        # If no sector data available, provide fallback data
        if not sector_performance:
            sector_performance = [
                {"name": "XLK", "performance": 0.0},
                {"name": "XLF", "performance": 0.0},
                {"name": "XLV", "performance": 0.0},
                {"name": "XLE", "performance": 0.0},
                {"name": "XLI", "performance": 0.0}
            ]
        
        return jsonify({
            "market_trend": "Bullish" if spy_return > 0 else "Bearish",
            "spy_performance": spy_return,
            "market_outlook": "Market showing positive momentum" if spy_return > 0 else "Market showing weakness",
            "top_sectors": sorted(sector_performance, key=lambda x: x["performance"], reverse=True),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Quarterly analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Test yfinance connection
        spy = yf.Ticker("SPY")
        spy.info
        
        return jsonify({
            "status": "healthy",
            "services": {
                "yfinance": True,
                "groq": GROQ_API_KEY is not None,
                "news_api": bool(NEWS_API_KEY)
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/status')
def api_status():
    """Check API status and availability"""
    status = {
        "status": "operational",
        "services": {
            "yfinance": True,
            "groq": GROQ_API_KEY is not None,
            "news_api": bool(NEWS_API_KEY)
        },
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(status)

def analyze_sentiment_with_cache(ticker, text, retries=3, delay=1):
    now = time.time()
    # Check cache
    if ticker in sentiment_cache and now - sentiment_cache[ticker]["timestamp"] < CACHE_DURATION:
        return sentiment_cache[ticker]["result"], True
    # Call Groq
    result = groq_generate_content(text, max_tokens=300, temperature=0.3)
    # Cache result
    sentiment_cache[ticker] = {"result": result, "timestamp": now}
    return result, False

def get_polygon_history_batch(tickers, from_date, to_date, api_key):
    import requests
    import pandas as pd
    import time
    from datetime import date, timedelta
    all_data = {}
    for ticker in tickers:
        # Try Polygon only
        try:
            url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/month/{from_date}/{to_date}?adjusted=true&sort=asc&apiKey={api_key}'
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                if results:
                    dates = [pd.to_datetime(r['t'], unit='ms') for r in results]
                    closes = [r['c'] for r in results]
                    all_data[ticker] = pd.Series(closes, index=dates)
                    continue
            # If 10 years fails or returns no data, try 5 years
            five_years_ago = to_date - timedelta(days=365*5)
            url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/month/{five_years_ago}/{to_date}?adjusted=true&sort=asc&apiKey={api_key}'
            for attempt in range(3):
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        results = resp.json().get('results', [])
                        if results:
                            dates = [pd.to_datetime(r['t'], unit='ms') for r in results]
                            closes = [r['c'] for r in results]
                            all_data[ticker] = pd.Series(closes, index=dates)
                            break
                    elif resp.status_code == 429:  # Rate limit
                        print(f"Rate limit hit for {ticker}, waiting before retry...")
                        time.sleep(5)  # Wait longer for rate limit
                    else:
                        print(f"Polygon API error for {ticker}: {resp.status_code}")
                except Exception as e:
                    print(f"Exception for {ticker} (attempt {attempt+1}): {e}")
                    time.sleep(2)
            else:
                print(f"Failed to fetch data for {ticker} after all Polygon attempts.")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    df = pd.DataFrame(all_data)
    return df

def get_alpha_vantage_history_batch(tickers, from_date, to_date, api_key):
    all_data = {}
    for ticker in tickers:
        try:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={ticker}&apikey={api_key}'
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get('Monthly Adjusted Time Series', {})
                if data:
                    # Parse and filter by date
                    series = {datetime.strptime(date, '%Y-%m-%d'): float(vals['5. adjusted close']) for date, vals in data.items()}
                    filtered = {d: v for d, v in series.items() if from_date <= d.date() <= to_date}
                    if filtered:
                        all_data[ticker] = pd.Series(filtered)
            else:
                print(f"Alpha Vantage API error for {ticker}: {resp.status_code}")
        except Exception as e:
            print(f"Error fetching Alpha Vantage data for {ticker}: {e}")
    df = pd.DataFrame(all_data)
    return df

@app.route('/api/polygon_history', methods=['GET'])
def get_polygon_history():
    global last_request_time
    
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker parameter is required"}), 400

    # Check cache first
    current_time = int(time.time())
    cache_key = f"{ticker}_{current_time // CACHE_DURATION}"
    cached_data = get_cached_market_data(cache_key, current_time)
    if cached_data:
        return jsonify(cached_data)

    if not POLYGON_API_KEY:
        logger.error("Polygon API key is missing")
        return jsonify({"error": "API key not configured"}), 500

    try:
        # Rate limiting
        time_since_last = time.time() - last_request_time
        if time_since_last < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - time_since_last)

        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        url = f"{POLYGON_BASE_URL}/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {
            "apiKey": POLYGON_API_KEY,
            "limit": 30
        }
        
        logger.info(f"Fetching data for {ticker}")
        response = requests.get(url, params=params)
        last_request_time = time.time()
        
        if response.status_code == 429:
            logger.warning(f"Rate limit exceeded for {ticker}")
            return jsonify({"error": "Rate limit exceeded"}), 429
        elif response.status_code != 200:
            logger.error(f"Error fetching data for {ticker}: {response.status_code}")
            return jsonify({"error": f"Failed to fetch data: {response.status_code}"}), response.status_code

        data = response.json()
        if "results" not in data:
            logger.error(f"No results found for {ticker}")
            return jsonify({"error": "No data available"}), 404

        # Format the data
        history = []
        for bar in data["results"]:
            history.append({
                "date": datetime.fromtimestamp(bar["t"] / 1000).strftime("%Y-%m-%d"),
                "open": bar["o"],
                "high": bar["h"],
                "low": bar["l"],
                "close": bar["c"],
                "volume": bar["v"]
            })

        result = {"history": history}
        
        # Cache the results
        get_cached_market_data.cache_clear()  # Clear old cache entries
        get_cached_market_data(cache_key, current_time)
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing request for {ticker}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/dashboard', methods=['GET'])
def dashboard():
    demo_data = [
        {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "price": 190.12,
            "change": 1.23,
            "percent_change": 0.65
        },
        {
            "ticker": "MSFT",
            "company_name": "Microsoft Corp.",
            "price": 320.45,
            "change": -2.34,
            "percent_change": -0.73
        },
        {
            "ticker": "TSLA",
            "company_name": "Tesla Inc.",
            "price": 780.56,
            "change": 5.67,
            "percent_change": 0.73
        }
    ]
    return jsonify(demo_data)

if __name__ == '__main__':
    port = 5001
    while port < 5010:  # Try ports 5001-5009
        try:
            app.run(host='0.0.0.0', port=port, debug=True)
            break
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"Port {port} is in use, trying {port + 1}")
                port += 1
            else:
                raise e
    port = 5001
    while port < 5010:  # Try ports 5001-5009
        try:
            app.run(host='0.0.0.0', port=port, debug=True)
            break
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"Port {port} is in use, trying {port + 1}")
                port += 1
            else:
                raise e