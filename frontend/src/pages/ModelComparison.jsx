import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import './ModelComparison.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const ModelComparison = () => {
  const [stocks, setStocks] = useState([]);
  const [selectedStock, setSelectedStock] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch list of stocks for selection
    const fetchStocks = async () => {
      try {
        const response = await axios.get('/api/stocks'); // Adjust API endpoint as needed
        console.log('Stocks API response:', response.data);
        if (Array.isArray(response.data)) {
          setStocks(response.data);
        } else {
          console.error('Stocks API response is not an array:', response.data);
          setStocks([]);
        }
      } catch (error) {
        console.error('Error fetching stocks:', error);
        setStocks([]);
      }
    };
    fetchStocks();
  }, []);

  const handleStockChange = (e) => {
    setSelectedStock(e.target.value);
    setPredictions(null);
  };

  const fetchPredictions = async () => {
    if (!selectedStock) return;
    setLoading(true);
    try {
      const response = await axios.get(`/api/model_comparison?symbol=${selectedStock}`);
      console.log("Model comparison API response:", response.data);
      setPredictions(response.data);
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
    setLoading(false);
  };

  const getChartData = (model) => {
    if (!predictions) return {};
    return {
      labels: predictions[model].time_series,
      datasets: [
        {
          label: 'Predicted',
          data: predictions[model].predictions,
          borderColor: 'rgb(53, 162, 235)',
          backgroundColor: 'rgba(53, 162, 235, 0.5)',
        },
        {
          label: 'Actual',
          data: predictions[model].actuals,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
        },
      ],
    };
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Prediction vs Actual',
      },
    },
  };

  return (
    <div className="model-comparison-page">
      <h1>
        <i className="fas fa-chart-bar"></i> Model Comparison
      </h1>

      <div>
        <label htmlFor="stock-select">Select Stock Symbol: </label>
        <select
          id="stock-select"
          value={selectedStock}
          onChange={handleStockChange}
        >
          <option value="">-- Choose a stock --</option>
          {stocks.map((stock) => (
            <option key={stock.symbol} value={stock.symbol}>
              {stock.name} ({stock.symbol})
            </option>
          ))}
        </select>

        <button
          onClick={fetchPredictions}
          disabled={!selectedStock || loading}
        >
          {loading ? 'Analyzing...' : 'Compare Models'}
        </button>
      </div>

      {predictions && (
        <div className="predictions-results-vertical">
          <h2>Analysis Results for {selectedStock}</h2>

          <div className="model-sections">
            {['LSTM', 'Transformer'].map((model) => (
              <div key={model} className="model-section">
                <h3>{model}</h3>
                <div className="predicted-price">
                  Predicted: ₹
                  {predictions[model]?.predicted_close?.toFixed(2) || 'N/A'}
                </div>
                <div className="predicted-price">
                  Actual: ₹{predictions[model]?.actual_close?.toFixed(2) || 'N/A'}
                </div>

                <Line options={options} data={getChartData(model)} />

                <div className="metrics-table-container">
                  <table className="metrics-table">
                    <tbody>
                      <tr>
                        <th>MSE</th>
                        <td>
                          {predictions[model]?.mse?.toFixed(4) ?? 'N/A'}
                        </td>
                      </tr>
                      <tr>
                        <th>RMSE</th>
                        <td>
                          {predictions[model]?.rmse?.toFixed(4) ?? 'N/A'}
                        </td>
                      </tr>
                      <tr>
                        <th>MAE</th>
                        <td>
                          {predictions[model]?.mae?.toFixed(4) ?? 'N/A'}
                        </td>
                      </tr>
                      <tr>
                        <th>R² Score</th>
                        <td>
                          {predictions[model]?.r2_score?.toFixed(4) ?? 'N/A'}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelComparison;
