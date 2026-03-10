import React from 'react';
import { render, screen } from '@testing-library/react';
import StockPage from '../pages/StockPage';

describe('StockPage Component', () => {
  test('renders StockPage heading', () => {
    render(<StockPage />);
    const headingElement = screen.getByRole('heading', { name: /Stock Page/i });
    expect(headingElement).toBeInTheDocument();
  });

  // Add more tests as needed for StockPage functionality
});
