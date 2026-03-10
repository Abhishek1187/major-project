import React from 'react';
import DropdownMenu from './DropDown.jsx';
import SearchBar from './SearchBar';
import './styles/Header.css';
import { useNavigate, Link } from 'react-router-dom';

// Map display names to stockSymbols keys
const stockDisplayMap = {
  'Reliance': 'RELIANCE',
  'Airtel': 'AIRTEL',
  'ONGC': 'ONGC',
  'Adani Enterprises': 'ADANIENTERPRISES',
  'Adani Port': 'ADANIPORT',
  'Infosys': 'INFOSYS',
  'HDFC Bank': 'HDFCBANK',
  'Axis Bank': 'AXISBANK',
  'Bajaj Finance': 'BAJAJFINANCE',
  'SBI Bank': 'SBIBANK'
};

const indexDisplayMap = {
  'Nifty 50': 'NIFTY50',
  'Bank Nifty': 'BANKNIFTY',
  'Nifty Midcap': 'NIFTYMIDCAP',
  'Nifty Auto': 'NIFTYAUTO'
};

const Header = () => {
  const navigate = useNavigate();

  const handleStockSelect = (item) => {
    const symbol = stockDisplayMap[item] || item.replace(/\s+/g, '').toUpperCase();
    navigate(`/stock/${symbol}`);
  };

  const handleIndexSelect = (item) => {
    const symbol = indexDisplayMap[item] || item.replace(/\s+/g, '').toUpperCase();
    navigate(`/stock/${symbol}`);
  };

  return (
    <header className="header">
      <Link to="/" className="logo-link">
        <div className="logo">
          <i className="fas fa-chart-bar"></i>
          <h1>Smart Stock Predictor</h1>
        </div>
      </Link>

      <nav className="nav">
        <div className="nav-group">
          <DropdownMenu 
            title={<><i className="fas fa-chart-line"></i>Indices</>}
            items={Object.keys(indexDisplayMap)}
            onSelect={handleIndexSelect}
          />
          <DropdownMenu 
            title={<><i className="fas fa-building"></i>Stocks</>}
            items={Object.keys(stockDisplayMap)}
            onSelect={handleStockSelect}
          />
        </div>
        <div className="search-container">
          <SearchBar />
        </div>
      </nav>
    </header>
  );
};

export default Header;
