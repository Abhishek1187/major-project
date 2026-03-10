import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import PredictionDisplay from '../components/PredictionDisplay';
import OHLCVViewer from '../components/StockChart';
import '../components/styles/StockPage.css';
import Header from '../components/Header.jsx';

const StockPage = () => {
  const { symbol } = useParams();
  const [latestPrice, setLatestPrice] = useState(null);
  const [chartLoaded, setChartLoaded] = useState(false);
  const [showPrediction, setShowPrediction] = useState(false);

  // Reset state when symbol changes to prevent auto-prediction on new stock
  React.useEffect(() => {
    setShowPrediction(false);
    setChartLoaded(false);
    setLatestPrice(null);
  }, [symbol]);

  const handleChartLoad = () => {
    setChartLoaded(true);
  };

  const handlePredictClick = () => {
    if (chartLoaded) {
      setShowPrediction(true);
    } else {
      alert('Please wait until the chart finishes loading.');
    }
  };

  return (
    <div className="stock-page">
      <Header />

      <div className="stock-header">
        <div className="stock-info">
          <h1 className="stock-name">
            <i className="fas fa-chart-line"></i>
            {symbol}
          </h1>
          <div className="price-container">
            <span className="price-label">Current Price</span>
            <p className="latest-price">
              ₹ {latestPrice?.toLocaleString('en-IN', { maximumFractionDigits: 2 }) ?? 'Loading...'}
            </p>
          </div>
        </div>
      </div>

      <div className="content-container">
        <div className="chart-section">
          <div className="chart-wrapper">
            <h3 className="chart-title">
              <i className="fas fa-chart-area"></i>
              Price Chart
            </h3>
            <OHLCVViewer
              symbol={symbol}
              onChartLoad={handleChartLoad}
              onLatestPrice={setLatestPrice}
              setLoadingParent={(loading) => setChartLoaded(!loading)}
            />
          </div>

          <div className="prediction-section">
            <button 
              onClick={handlePredictClick} 
              className={`predict-button ${chartLoaded ? 'enabled' : 'disabled'}`}
              disabled={!chartLoaded}
            >
              <i className="fas fa-brain"></i>
              {chartLoaded ? 'Predict Future Price' : 'Loading Chart...'}
            </button>

            {showPrediction && <PredictionDisplay symbol={symbol} />}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StockPage;
