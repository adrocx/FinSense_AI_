import yfinance as yf

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        
        if not info or 'shortName' not in info:
            return {'error': 'No data found for this ticker'}
        
        metrics = {
            'ticker': ticker,
            'company_name': info.get('shortName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'eps': info.get('trailingEps', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'revenue': financials.loc['Total Revenue'][0] if 'Total Revenue' in financials.index else 'N/A',
            'error': None
        }
        return metrics
    except Exception as e:
        return {'error': f'Error fetching fundamentals: {str(e)}'}