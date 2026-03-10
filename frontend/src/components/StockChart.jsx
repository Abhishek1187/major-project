import React, { useEffect, useState } from "react";
import ReactApexChart from "react-apexcharts";
import axios from "axios";
import stockSymbols from "../utils/stockSymbols.js";

const OHLCVViewer = ({ symbol, onLatestPrice, setLoadingParent }) => {
  const [chartOptions, setChartOptions] = useState({});
  const [chartSeries, setChartSeries] = useState([{ data: [] }]);
  const [earliestTime, setEarliestTime] = useState(null);
  const [latestTime, setLatestTime] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchOHLCV = async (nseSymbol, limit = 375) => {
    const res = await axios.get(`http://127.0.0.1:8000/api/ohlcv/${nseSymbol}/`, {
      params: { limit },
    });
    return res.data.filter(entry =>
      entry.Open != null && entry.High != null && entry.Low != null && entry.Close != null && entry.Datetime
    );
  };

  const prepareChartData = (data) => {
    return data.map(entry => ({
      x: new Date(entry.Datetime),
      y: [
        parseFloat(entry.Open),
        parseFloat(entry.High),
        parseFloat(entry.Low),
        parseFloat(entry.Close),
      ],
    }));
  };

  const initializeChart = async () => {
    const symbolObj = stockSymbols[symbol];
    console.log(symbol);
    const nseSymbol = symbolObj?.NSE;
    if (!nseSymbol) return;

    setLoading(true);
    setLoadingParent?.(true);

    try {
      const data = await fetchOHLCV(nseSymbol);
      if (data.length > 0) {
        const formattedData = prepareChartData(data);
        const earliest = new Date(data[0].Datetime);
        const latest = new Date(data[data.length - 1].Datetime);
        setEarliestTime(earliest);
        setLatestTime(latest);

        const closePrices = data.map(entry => parseFloat(entry.Close));
        const minPrice = Math.floor(Math.min(...closePrices));
        const maxPrice = Math.ceil(Math.max(...closePrices));

        setChartSeries([{ data: formattedData }]);
        setChartOptions({
          chart: {
          type: "candlestick",
          height: 700,
          zoom: { enabled: true, type: "x" },
          toolbar: { autoSelected: "zoom" },
          animations: {
            enabled: true,
            easing: 'easeinout',
            speed: 300,
            animateGradually: {
              enabled: true,
              delay: 150
            },
            dynamicAnimation: {
              enabled: true,
              speed: 350
            }
          }
        },
          title: { text: `${symbol}`, align: "left" },
          xaxis: {
            type: "datetime",
            labels: {
              rotate: -45,
              style: { fontSize: "10px" },
              datetimeUTC: false,
            },
            min: earliest.getTime(),
            max: latest.getTime(),
          },
          yaxis: {
            tooltip: {
              enabled: true
            },
            tickAmount: 6,
            labels: {
              formatter: val => val.toFixed(2),
              style: { fontSize: "10px" },
            },
            min: minPrice,
            max: maxPrice,
          },
          tooltip: {
            enabled: true,
            shared: true,
            custom: function({ series, seriesIndex, dataPointIndex, w }) {
              const o = w.globals.seriesCandleO[seriesIndex][dataPointIndex];
              const h = w.globals.seriesCandleH[seriesIndex][dataPointIndex];
              const l = w.globals.seriesCandleL[seriesIndex][dataPointIndex];
              const c = w.globals.seriesCandleC[seriesIndex][dataPointIndex];
              return (
                '<div class="apexcharts-tooltip-candlestick">' +
                '<div>Open: ' + o.toFixed(2) + '</div>' +
                '<div>High: ' + h.toFixed(2) + '</div>' +
                '<div>Low: ' + l.toFixed(2) + '</div>' +
                '<div>Close: ' + c.toFixed(2) + '</div>' +
                '</div>'
              );
            }
          },
          plotOptions: {
            candlestick: { wick: { useFillColor: true } },
          },
          dataLabels: { enabled: false },
        });

        const latestClose = parseFloat(data[data.length - 1].Close).toFixed(2);
        onLatestPrice?.(latestClose);
      }
    } catch (error) {
      console.error("Error loading chart data:", error);
    } finally {
      setLoading(false);
      setLoadingParent?.(false);
    }
  };

  useEffect(() => {
    initializeChart();
  }, [symbol]);

  return (
    <div style={{ overflowX: "auto", paddingBottom: "10px", width: "100%", maxWidth: "1200px", margin: "0 auto" }}>
      {loading ? (
        <p>Loading chart...</p>
      ) : (
        <ReactApexChart
          options={chartOptions}
          series={chartSeries}
          type="candlestick"
          height={400}
          width="100%"
        />
      )}
    </div>
  );
};

export default OHLCVViewer;
