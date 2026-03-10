import React from 'react';
import { render, screen } from '@testing-library/react';
import PredictionDisplay from '../PredictionDisplay';

describe('PredictionDisplay Component', () => {
  const mockProps = {
    predictedClose: 150.25,
    averageSentiment: 0.75,
    articleCount: 20,
  };

  test('renders predicted close price', () => {
    render(<PredictionDisplay {...mockProps} />);
    const priceElement = screen.getByText(/Predicted Close Price:/i);
    expect(priceElement).toBeInTheDocument();
    expect(priceElement).toHaveTextContent('150.25');
  });

  test('renders average sentiment', () => {
    render(<PredictionDisplay {...mockProps} />);
    const sentimentElement = screen.getByText(/Average Sentiment:/i);
    expect(sentimentElement).toBeInTheDocument();
    expect(sentimentElement).toHaveTextContent('0.75');
  });

  test('renders article count', () => {
    render(<PredictionDisplay {...mockProps} />);
    const articleCountElement = screen.getByText(/Articles Analyzed:/i);
    expect(articleCountElement).toBeInTheDocument();
    expect(articleCountElement).toHaveTextContent('20');
  });
});
