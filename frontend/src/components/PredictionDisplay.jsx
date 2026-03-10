import { useEffect, useState } from "react";
import axios from "axios";
import stockSymbols from "../utils/stockSymbols";
import "./styles/PredictionDisplay.css";

export default function PredictionDisplay({ symbol }) {
  const [price, setPrice] = useState(null);
  const [sentiment, setSentiment] = useState(null);
  const [articleCount, setArticleCount] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPrice = async (symbol) => {
      try {
        let actualSymbol = symbol;
        const symbolKey = symbol.toUpperCase().replace(/\s+/g, '');
        if (stockSymbols[symbolKey]) {
          actualSymbol = stockSymbols[symbolKey].NSE;
        }

        console.log(`Fetching prediction for: ${symbol} -> ${actualSymbol}`);
        // Changed API URL to relative path for better environment compatibility
        const response = await axios.post(`/predict/${actualSymbol}/`);

        if (response.data.error) {
          setError(response.data.error);
          setPrice(null);
          setSentiment(null);
          setArticleCount(null);
          return;
        }

        setPrice(response.data.predicted_close);
        setSentiment(response.data.average_sentiment);
        setArticleCount(response.data.article_count);
        setError(null);
      } catch (err) {
        console.error("Error fetching prediction:", err);
        setError(`Failed to get prediction: ${err.message}`);
        setPrice(null);
        setSentiment(null);
        setArticleCount(null);
      }
    };

    if (symbol) {
      setLoading(true);
      fetchPrice(symbol).finally(() => setLoading(false));
    }
  }, [symbol]);

  return (
    <div className="prediction-container">
      <h2 className="prediction-title">AI Prediction</h2>

      {loading ? (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Analyzing market data with AI...</p>
        </div>
      ) : error ? (
        <div className="error-message">
          <i className="fas fa-exclamation-circle"></i>
          <p>{error}</p>
        </div>
      ) : price ? (
        <div className="prediction-content">
          <div className="prediction-value">
            <span className="label">Predicted Next Price:</span>
            <span className="value success">
              ₹{price.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
            </span>
          </div>

          {sentiment !== null && (
            <div className="sentiment-value">
              <span className="label">News Sentiment Score:</span>
              <span className={`value ${sentiment >= 0 ? 'up' : 'down'}`}>
                {sentiment.toFixed(3)}
              </span>
            </div>
          )}
          {typeof articleCount === 'number' && (
            <div className={`article-count ${articleCount > 0 ? 'up' : 'down'}`}>
              <span className="label">News Articles Fetched:</span>
              <span className={`value ${articleCount > 0 ? 'up' : 'down'}`}>{articleCount}</span>
            </div>
          )}

          <div className="prediction-indicator success">
            <i className="fas fa-brain"></i>
            <span>AI Prediction Generated Successfully</span>
          </div>

          <div className="prediction-note">
            <i className="fas fa-info-circle"></i>
            <p>This prediction is based on the Transformer model analyzing historical price data, technical indicators, and news sentiment analysis.</p>
          </div>
        </div>
      ) : (
        <div className="error-message">
          <i className="fas fa-exclamation-circle"></i>
          <p>No prediction available</p>
        </div>
      )}
    </div>
  );
}
