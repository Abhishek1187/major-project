import React, { useState } from 'react';
import './styles/SearchBar.css';

const SearchBar = () => {
  const [query, setQuery] = useState('');

  const handleSearch = () => {
    alert(`Searching for: ${query}`);
  };

  return (
    <div className="search-bar">
      <input
        type="text"
        placeholder="Search stock/index"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button onClick={handleSearch}>Search</button>
    </div>
  );
};

export default SearchBar;
