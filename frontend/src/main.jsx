import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import 'bootstrap/dist/css/bootstrap.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';

// Import common style
import '../src/styles/main.css';
import '../src/styles/component/ScatterPlot.css';

// Import pages
import ErrorPage from './components/ErrorPage';
import HomePage from './pages/HomePage';
import PredictPricePage from './pages/PredictPricePage';
import PredictHistory from './pages/PredictHistory';
import MarketCharts from './pages/MarketCharts';
import Calculator from './pages/Calculator';

// Import components
import ScatterPlotPage from './components/ScatterPlotComponent';

const router = createBrowserRouter([
  {
    path: '/',
    element: <HomePage />,
    errorElement: <ErrorPage />
  },
  {
    path: '/predict-price',
    element: <PredictPricePage />
  },
  {
    path: '/predict-history',
    element: <PredictHistory />
  },
  {
    path: '/market-charts',
    element: <MarketCharts />
  },
  {
    path: '/calculator',
    element: <Calculator />
  }
]);

const container = document.getElementById('root');
const root = createRoot(container);

root.render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
