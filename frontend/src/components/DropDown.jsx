import React, { useState } from 'react';
import './styles/Header.css';

const DropdownMenu = ({ title, items, onSelect }) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="dropdown" onMouseLeave={() => setOpen(false)}>
      <button onClick={() => setOpen(!open)} className="dropdown-button">
        {title} ▾
      </button>
      {open && (
        <ul className="dropdown-menu">
          {items.map(item => (
            <li 
                key={item} 
                className="dropdown-item"
                onClick={() => {
                    onSelect(item);
                    setOpen(false);
                }}
            >
                {item}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default DropdownMenu;
