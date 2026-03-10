import React, { useEffect, useState, useMemo } from "react";
import ReactApexChart from "react-apexcharts";
import axios from "axios";
import stockSymbols from "../utils/stockSymbols.js";

const OHLCVViewer = ({ symbol, onLatestPrice, setLoadingParent }) => {
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchOHLCV = async (nseSymbol) => {
    const res = await axios.get(`/api/ohlcv/${nseSymbol}/`, {
      params: { limit: 200 }, // Reduced for faster load
    });
    return res.data.filter(entry =>
      entry.Open != null && entry.High != null && entry.Low != null && entry.Close != null && entry.Datetime
    );
  };

  // Memoize chart configuration for performance
  const { chartOptions, chartSeries } = useMemo(() => {
    if (chartData.length === 0) {
      return { chartOptions: {}, chartSeries: [{ data: [] }] };
    }

    const formattedData = chartData.map(entry => ({
      x: new Date(entry.Datetime),
      y: [
        parseFloat(entry.Open),
        parseFloat(entry.High),
        parseFloat(entry.Low),
        parseFloat(entry.Close),
      ],
    }));

    const allPrices = chartData.flatMap(entry => [
      parseFloat(entry.Open),
      parseFloat(entry.High),
      parseFloat(entry.Low),
      parseFloat(entry.Close)
    ]);
    const priceRange = Math.max(...allPrices) - Math.min(...allPrices);
    const padding = priceRange * 0.08;
    const minPrice = Math.min(...allPrices) - padding;
    const maxPrice = Math.max(...allPrices) + padding;

    return {
      chartSeries: [{ name: 'Price', data: formattedData }],
      chartOptions: {
        chart: {
          type: "candlestick",
          background: 'transparent',
          animations: { enabled: false }, // Disabled for faster render
          toolbar: {
            show: true,
            tools: {
              download: false,
              selection: true,
              zoom: true,
              zoomin: true,
              zoomout: true,
              pan: true,
              reset: true
            },
            autoSelected: 'zoom'
          },
          zoom: { enabled: true, type: 'x', autoScaleYaxis: true },
        },
        grid: {
          borderColor: 'rgba(255,255,255,0.1)',
          strokeDashArray: 3,
        },
        plotOptions: {
          candlestick: {
            colors: {
              upward: '#22c55e',
              downward: '#ef4444'
            },
            wick: { useFillColor: true }
          }
        },
        xaxis: {
          type: "datetime",
          labels: {
            style: { colors: '#94a3b8', fontSize: '11px' },
            datetimeUTC: false,
            datetimeFormatter: {
              hour: 'HH:mm',
              day: 'dd MMM',
            }
          },
          axisBorder: { color: 'rgba(255,255,255,0.1)' },
          axisTicks: { color: 'rgba(255,255,255,0.1)' },
        },
        yaxis: {
          tooltip: { enabled: true },
          labels: {
            formatter: val => '₹' + val.toFixed(2),
            style: { colors: '#94a3b8', fontSize: '11px' },
          },
          min: minPrice,
          max: maxPrice,
        },
        tooltip: {
          theme: 'dark',
          custom: function({ seriesIndex, dataPointIndex, w }) {
            const o = w.globals.seriesCandleO[seriesIndex][dataPointIndex];
            const h = w.globals.seriesCandleH[seriesIndex][dataPointIndex];
            const l = w.globals.seriesCandleL[seriesIndex][dataPointIndex];
            const c = w.globals.seriesCandleC[seriesIndex][dataPointIndex];
            const change = ((c - o) / o * 100).toFixed(2);
            const color = c >= o ? '#22c55e' : '#ef4444';
            return `<div style="padding:12px;background:#1e293b;border-radius:8px;border:1px solid #334155;">
              <div style="margin-bottom:8px;font-weight:600;color:#f1f5f9;">OHLC Data</div>
              <div style="display:grid;gap:4px;font-size:13px;">
                <div style="color:#94a3b8;">Open: <span style="color:#f1f5f9;font-weight:500;">₹${o.toFixed(2)}</span></div>
                <div style="color:#94a3b8;">High: <span style="color:#22c55e;font-weight:500;">₹${h.toFixed(2)}</span></div>
                <div style="color:#94a3b8;">Low: <span style="color:#ef4444;font-weight:500;">₹${l.toFixed(2)}</span></div>
                <div style="color:#94a3b8;">Close: <span style="color:#f1f5f9;font-weight:500;">₹${c.toFixed(2)}</span></div>
                <div style="color:#94a3b8;margin-top:4px;padding-top:4px;border-top:1px solid #334155;">Change: <span style="color:${color};font-weight:600;">${change}%</span></div>
              </div>
            </div>`;
          }
        },
        dataLabels: { enabled: false },
      }
    };
  }, [chartData]);

  useEffect(() => {
    const loadChart = async () => {
      const symbolObj = stockSymbols[symbol];
      const nseSymbol = symbolObj?.NSE;
      
      if (!nseSymbol) {
        setError(`Symbol "${symbol}" not found`);
        setLoading(false);
        setLoadingParent?.(false);
        return;
      }

      setError(null);
      setChartData([]);
      setLoading(true);
      setLoadingParent?.(true);

      try {
        const data = await fetchOHLCV(nseSymbol);
        if (data.length > 0) {
          setChartData(data);
          const latestClose = parseFloat(data[data.length - 1].Close);
          onLatestPrice?.(latestClose.toFixed(2));
        } else {
          setError('No data available');
        }
      } catch (err) {
        console.error("Chart error:", err);
        setError(`Failed to load: ${err.message}`);
      } finally {
        setLoading(false);
        setLoadingParent?.(false);
      }
    };

    loadChart();
  }, [symbol]);

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '350px', color: '#94a3b8' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ width: '40px', height: '40px', border: '3px solid rgba(6,182,212,0.2)', borderTop: '3px solid #06b6d4', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 12px' }}></div>
          <p>Loading chart data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '350px', color: '#ef4444' }}>
        <div style={{ textAlign: 'center' }}>
          <i className="fas fa-exclamation-circle" style={{ fontSize: '2rem', marginBottom: '8px' }}></i>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (chartData.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '350px', color: '#94a3b8' }}>
        <p>No chart data available</p>
      </div>
    );
  }

  return (
    <div style={{ width: '100%' }}>
      <ReactApexChart
        options={chartOptions}
        series={chartSeries}
        type="candlestick"
        height={350}
      />
    </div>
  );
};

export default OHLCVViewer;
