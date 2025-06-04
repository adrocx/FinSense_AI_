import yfinance as yf
from transformers import pipeline

def get_sentiment(ticker):
    try:
        # Get stock info
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return {
                'error': 'No news articles found for this ticker'
            }
        
        # Initialize sentiment analyzer
        sentiment_analyzer = pipeline('sentiment-analysis')
        
        # Analyze sentiment for each article
        sentiments = []
        for article in news[:20]:  # Analyze up to 5 articles
            title = article.get('title', '')
            if not title:
                continue
            sentiment = sentiment_analyzer(title)[0]
            sentiments.append({
                'title': title,
                'sentiment': sentiment['label'],
                'score': round(sentiment['score'], 3)
            })
        
        if not sentiments:
            return {
                'error': 'Could not analyze sentiment for this ticker'
            }
        
        # Calculate overall sentiment score (average of positive scores)
        sentiment_score = sum(s['score'] if s['sentiment'] == 'POSITIVE' else 1-s['score'] for s in sentiments) / len(sentiments)
        
        # Key findings: summarize each article's sentiment
        key_findings = [f"{s['title']} ({s['sentiment']}, {s['score']})" for s in sentiments]
        
        # Recommendations (simple logic)
        recommendations = []
        if sentiment_score > 0.6:
            recommendations.append("Market sentiment is strongly positive. Consider buying or holding.")
        elif sentiment_score > 0.4:
            recommendations.append("Market sentiment is mixed. Exercise caution.")
        else:
            recommendations.append("Market sentiment is negative. Consider reducing exposure.")
        
        return {
            'sentiment_score': sentiment_score,
            'key_findings': key_findings,
            'recommendations': recommendations
        }
        
    except Exception as e:
        return {
            'error': f'Error analyzing sentiment: {str(e)}'
        }