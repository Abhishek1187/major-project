import React from 'react';
import Header from '../components/Header';
import { Link } from 'react-router-dom';
import '../components/styles/Homepage.css';

const HomePage = () => {
  return (
    <div className="home-page">
      <Header />
      <main className="main-content">
        <div className="hero-section">
          <h1 className="tagline">
            <span className="highlight">Smart</span> Stock Predictions
          </h1>
          <p className="description">
            Track live stock prices, analyze market trends, and get AI-powered predictions using advanced LSTM and Transformer models.
          </p>
          
          <div className="features-grid">
            <div className="feature-card">
              <i className="fas fa-chart-line"></i>
              <h3>Real-Time Charts</h3>
              <p>Interactive charts with technical indicators</p>
            </div>
            
            <div className="feature-card">
              <i className="fas fa-brain"></i>
              <h3>AI Predictions</h3>
              <p>Advanced machine learning models for accurate forecasting</p>
            </div>
            
            <div className="feature-card">
              <i className="fas fa-bolt"></i>
              <h3>Live Updates</h3>
              <p>Real-time stock and index price updates</p>
            </div>
          </div>

          <div className="instruction-card">
            <i className="fas fa-info-circle"></i>
            <p>Select a stock or index from the menu above to begin analysis</p>
          </div>

          <div style={{ marginTop: '20px' }}>
            <Link to="/model-comparison">
              <button className="homepage-model-comparison-button">Model Comparison</button>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
};

export default HomePage;
