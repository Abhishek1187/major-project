from django.shortcuts import render

import yfinance as yf
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime
from dateutil import parser

@api_view(['GET'])
def get_ohlcv(request, symbol):
    try:
        end_time = request.query_params.get('end_time')
        limit = int(request.query_params.get('limit', 375))

        data = yf.Ticker(symbol)

        hist = data.history(period="7d", interval='1m')
        hist.reset_index(inplace=True)

        if end_time:
            # Parse the provided end_time and filter data
            end_dt = parser.parse(end_time)
            hist = hist[hist['Datetime'] < end_dt]

        # Limit the number of rows returned
        hist = hist.tail(limit)

        ohlcv_data = hist[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].to_dict(orient='records')
        return Response(ohlcv_data, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
