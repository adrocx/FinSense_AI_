import axios, { AxiosError } from 'axios';
import { NewsArticle } from '../types';

// API endpoint from environment variables
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001';

// Dashboard (market) data
export const fetchMarketData = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/dashboard`);
    return response.data;
  } catch (error) {
    console.error('Error fetching dashboard data:', error);
    throw error;
  }
};

// News data (from news endpoint)
export const fetchNewsData = async (): Promise<NewsArticle[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/news`);
    if (!Array.isArray(response.data)) {
      console.error('Invalid news data format received');
      return [];
    }
    return response.data.map((article: any) => ({
      title: article.title || '',
      source: article.source || 'Unknown',
      timestamp: article.timestamp || '',
      content: article.content || '',
      url: article.url || '',
      sentiment: article.sentiment || 0,
      credibility: article.credibility || 3
    }));
  } catch (error) {
    if (error instanceof AxiosError) {
      console.error('Error fetching news data:', error.response?.data || error.message);
    } else {
      console.error('Error fetching news data:', error);
    }
    return [];
  }
};

// Sentiment analysis (expects { ticker })
export const analyzeSentiment = async (ticker: string): Promise<SentimentAnalysis> => {
  try {
    const response = await axios.post(`${API_BASE_URL}/sentiment`, { ticker });
    // Assuming the backend now returns the new structure
    return response.data as SentimentAnalysis; 
  } catch (error) {
    console.error('Error analyzing sentiment:', error);
    // Return a default structure matching the new type in case of error
    return {
      overall_sentiment: { score: 0, label: 'Neutral' },
      sentiment_timeline: [],
      key_findings: ['Error analyzing sentiment'],
      recommendations: ['Unable to provide recommendations at this time'],
      top_news: []
    };
  }
};

// Fundamental analysis (expects { ticker })
export const fetchFundamentalData = async (ticker: string) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/fundamental`, { ticker });
    return {
      company_info: response.data.company_info || {},
      key_metrics: response.data.key_metrics || {},
      ai_insights: response.data.ai_insights || {},
      financial_ratios: response.data.financial_ratios || [],
      competitive_analysis: response.data.competitive_analysis || [],
      financial_statements: response.data.financial_statements || {}
    };
  } catch (error) {
    console.error('Error fetching fundamental data:', error);
    return {
      company_info: {},
      key_metrics: {},
      ai_insights: {},
      financial_ratios: [],
      competitive_analysis: [],
      financial_statements: {}
    };
  }
};

// Portfolio data (file upload, expects FormData)
export const fetchPortfolioData = async (file: File) => {
  try {
    const formData = new FormData();
    formData.append('portfolio', file);
    formData.append('risk_tolerance', 'Moderate'); // Default risk tolerance

    const response = await axios.post(`${API_BASE_URL}/portfolio`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return {
      current_portfolio: response.data.current_portfolio || {},
      recommended_optimization: response.data.recommended_optimization || {},
      metrics: response.data.metrics || {}
    };
  } catch (error) {
    console.error('Error fetching portfolio data:', error);
    throw error;
  }
};

// Quarterly market analysis (includes sector data)
export const fetchSectorAnalysis = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/quarterly`);
    return {
      market_trend: response.data.market_trend || 'Neutral',
      spy_performance: response.data.spy_performance || 0,
      market_outlook: response.data.market_outlook || 'Market data unavailable',
      top_sectors: response.data.top_sectors || [],
      timestamp: response.data.timestamp
    };
  } catch (error) {
    console.error('Error fetching sector analysis:', error);
    return {
      market_trend: 'Neutral',
      spy_performance: 0,
      market_outlook: 'Market data unavailable',
      top_sectors: [],
      timestamp: new Date().toISOString()
    };
  }
};

// Stock data (using quarterly endpoint)
export const fetchStockData = async (symbol: string) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/quarterly`);
    const news = response.data.news || [];
    return {
      metrics: response.data.metrics || {},
      news: Array.isArray(news) ? news.map((article: any) => ({
        title: article.title || '',
        source: article.source || 'Unknown',
        timestamp: article.timestamp || '',
        content: article.content || '',
        url: article.url || '',
        sentiment: article.sentiment || 0,
        credibility: article.credibility || 3
      })) : []
    };
  } catch (error) {
    if (error instanceof AxiosError) {
      console.error('Error fetching stock data:', error.response?.data || error.message);
    } else {
      console.error('Error fetching stock data:', error);
    }
    return {
      metrics: {},
      news: []
    };
  }
};

// General financial news (not filtered by ticker)
export const fetchGeneralNews = async (): Promise<NewsArticle[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/news`);
    if (!Array.isArray(response.data)) {
      console.error('Invalid news data format received');
      return [];
    }
    return response.data.map((article: any) => ({
      title: article.title || '',
      source: article.source || 'Unknown',
      timestamp: article.timestamp || '',
      content: article.content || '',
      url: article.url || '',
      sentiment: article.sentiment || 0,
      credibility: article.credibility || 3
    }));
  } catch (error) {
    if (error instanceof AxiosError) {
      console.error('Error fetching general news:', error.response?.data || error.message);
    } else {
      console.error('Error fetching general news:', error);
    }
    return [];
  }
};

// Fetch top 3 AI-driven stock recommendations
export const fetchRecommendations = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/recommendations`);
    return response.data || [];
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    return [];
  }
}; 