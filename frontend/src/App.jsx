import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Homepage from "./pages/HomePage.jsx";
import StockPage from "./pages/StockPage.jsx";
import ModelComparison from "./pages/ModelComparison.jsx";

// Proxy URL to avoid CORS issues for external API calls
const proxyUrl = "https://cors-anywhere.herokuapp.com/";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route path="/stock/:symbol" element={<StockPage proxyUrl={proxyUrl} />} />
        <Route path="/model-comparison" element={<ModelComparison />} />
      </Routes>
    </Router>
  );
};

export default App;
