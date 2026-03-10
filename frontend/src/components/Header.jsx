import React from 'react';
import DropdownMenu from './DropDown.jsx';
import SearchBar from './SearchBar';
import './styles/Header.css';
import { useNavigate, Link } from 'react-router-dom';

const Header = () => {
  const navigate = useNavigate();

  const handleSelect = (item) => {
    const symbol = item.replace(/\s+/g, '').toUpperCase();
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
            items={['Nifty 50', 'Bank Nifty', 'Nifty MIDCAP', 'Nifty Auto', 'Nifty Finance']}
            onSelect={handleSelect}
          />
          <DropdownMenu 
            title={<><i className="fas fa-building"></i>Stocks</>}
            items={['Reliance', 'Airtel', 'ONGC', 'Adani Enterprises', 'Adani Port', 'Infosys', 'HDFC Bank', 'Axis Bank', 'Bajaj Finance', 'SBI Bank']}
            onSelect={handleSelect}
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
