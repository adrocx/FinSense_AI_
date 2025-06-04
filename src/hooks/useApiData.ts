import { useState, useEffect, useCallback } from 'react';
import {
  fetchStockData,
  analyzeSentiment,
  fetchPortfolioData,
  fetchMarketData,
  fetchSectorAnalysis,
  fetchGeneralNews,
  fetchRecommendations
} from '../services/api';

export const useApiData = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [portfolioData, setPortfolioData] = useState<any>(null);
  const [newsData, setNewsData] = useState<any[]>([]);
  const [marketData, setMarketData] = useState<any>(null);
  const [sectorData, setSectorData] = useState<any>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch all data in parallel
      const [news, market, sectors] = await Promise.all([
        fetchGeneralNews(),
        fetchMarketData(),
        fetchSectorAnalysis()
      ]);

      setNewsData(news);
      setMarketData(market);
      setSectorData(sectors);
    } catch (err) {
      setError('Failed to fetch data');
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const uploadPortfolio = useCallback(async (file: File) => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchPortfolioData(file);
      setPortfolioData(data);
      return data;
    } catch (err) {
      setError('Failed to upload portfolio');
      console.error('Error uploading portfolio:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const getStockData = useCallback(async (symbol: string) => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchStockData(symbol);
      return data;
    } catch (err) {
      setError('Failed to fetch stock data');
      console.error('Error fetching stock data:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const getSentimentAnalysis = useCallback(async (text: string) => {
    try {
      setLoading(true);
      setError(null);
      const data = await analyzeSentiment(text);
      return data;
    } catch (err) {
      setError('Failed to analyze sentiment');
      console.error('Error analyzing sentiment:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Add function to fetch general news
  const getGeneralNews = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchGeneralNews();
      return data;
    } catch (err) {
      setError('Failed to fetch general news');
      console.error('Error fetching general news:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Add function to fetch recommendations
  const getRecommendations = useCallback(async () => {
    try {
      const data = await fetchRecommendations();
      return data;
    } catch (e) {
      return [];
    }
  }, []);

  // Initial data fetch
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Set up real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      fetchData();
    }, 60000); // Update every minute

    return () => clearInterval(interval);
  }, [fetchData]);

  return {
    loading,
    error,
    portfolioData,
    newsData,
    marketData,
    sectorData,
    getStockData,
    getSentimentAnalysis,
    uploadPortfolio,
    refreshData: fetchData,
    getGeneralNews,
    getRecommendations
  };
}; 