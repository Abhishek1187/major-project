import React from 'react';
import { render, screen } from '@testing-library/react';
import HomePage from '../pages/HomePage';

describe('HomePage Component', () => {
  test('renders HomePage heading', () => {
    render(<HomePage />);
    const headingElement = screen.getByRole('heading', { name: /Home Page/i });
    expect(headingElement).toBeInTheDocument();
  });

  // Add more tests as needed for HomePage functionality
});
