// frontend/src/pages/MarketCharts.jsx
import React, { useState } from 'react';
import NavBar from '../components/NavBar';
import Footer from '../components/Footer';
import '../styles/pages/charts.css';
import ScatterPlotPage from '../components/ScatterPlotComponent';
import BarChart from '../components/BarChartComponent';
import GaugeChartComponent from '../components/GaugeChartComponent'; // Import the GaugeChartComponent

export default function MarketCharts() {
  const [currentChartIndex, setCurrentChartIndex] = useState(0);
  const [animate, setAnimate] = useState(false);

  const handleArrowClick = () => {
    setAnimate(true);
    setTimeout(() => {
      setCurrentChartIndex((prevIndex) => (prevIndex + 1) % 3); // Assume 3 charts now
      setAnimate(false);
    }, 500);
  };

  const renderChart = () => {
    if (currentChartIndex === 0) {
      return <ScatterPlotPage />;
    } else if (currentChartIndex === 1) {
      return <BarChart />;
    } else if (currentChartIndex === 2) {
      return <GaugeChartComponent />;
    }
  };

  const renderChartText = () => {
    if (currentChartIndex === 0) {
      return (
        <p>
          Scatter Plot: This chart displays the distribution of houses by comparing various attributes with their prices. The x-axis represents the CBD distance and Landsize while the y-axis represents the Price. Users can zoom in and out to see the distribution of houses in different areas.Each dot represents a house's price and its distance from the CBD or the landsize.The scatter plot also shows the clusters of houses to help users identify the different groups of houses. 
        </p>
      );
    } else if (currentChartIndex === 1) {
      return (
        <p>
          Bar Chart: This chart allows users to compare the median prices of different suburbs over the years 2020 to 2023. Users can select specific attempts to see how the prices vary across different suburbs.
        </p>
      );
    } else if (currentChartIndex === 2) {
      return (
        <p>
          Gauge Chart: This chart shows the percentage distribution of S and NS properties. Users can enter a property score to see where it falls within the distribution, providing a visual representation of its position in the histogram.
        </p>
      );
    }
  };

  return (
    <>
      <NavBar />
      <div className="container mt-5">
        <div className="d-flex flex-column justify-content-center align-items-center text-center">
          <div className="row">
            <div className="col-11 image-container">
              <h1 className="chart-title">Market Chart</h1>
              <div className={`chart-content ${animate ? 'blur-slide' : ''}`}>
                {renderChart()}
              </div>
            </div>
            <div className="col-1 text-end arrow-icon" onClick={handleArrowClick}>
              <button className="btn btn-danger rounded-circle">
                <i className="fa-solid fa-arrow-right"></i>
              </button>
            </div>
          </div>
          <div className="row mt-5 mb-5">
            <div className="col-12">
              {renderChartText()}
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </>
  );
}