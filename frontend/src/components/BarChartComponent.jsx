// Data_Visualization/Bar_chart/BarChartComponent.jsx
import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import medianPriceSuburbData from '../data/Median_price_suburb'; // Import the data

const BarChartComponent = () => {
  const chartRef = useRef(null);
  const [currentData, setCurrentData] = useState(medianPriceSuburbData[0]); // Initialize with the first suburb
  const [historyData, setHistoryData] = useState([]);
  const [selectedAttempt, setSelectedAttempt] = useState(null);
  const [predictions, setPredictions] = useState([]); // Ensure predictions is initialized as an array

  const fetchPredictions = () => {
    fetch('http://127.0.0.1:8000/prediction-history/')
      .then((response) => {
        if(!response.ok){
          throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        return response.json();
      })
      .then((data) => {
        setPredictions(data.predictions);
      })
      .catch((error) => {
        alert('Error fetching history: ' + error.message);
      });
  };
  
  useEffect(() => {
    fetchPredictions();
    populateDropdown(medianPriceSuburbData);
  }, []);
  
  useEffect(() => {
    updateChart(currentData, selectedAttempt ? selectedAttempt.predicted_price : undefined);
  }, [currentData, selectedAttempt]);

  const populateDropdown = (data) => {
    const select = d3.select("#suburb-select");
    select.selectAll("option").remove(); // Clear existing options
    data.forEach(function (d) {
      select.append("option")
        .attr("value", d.suburb)
        .text(d.suburb);
    });

    select.on("change", function () {
      const selectedSuburb = d3.select(this).property("value");
      const selectedData = data.find(d => d.suburb === selectedSuburb);
      setCurrentData(selectedData);
    });
  };

  const handleAttemptChange = (event) => {
    const selectedAttemptId = parseInt(event.target.value, 10);
    const selectedAttempt = predictions.find(attempt => attempt.id === selectedAttemptId);
    setSelectedAttempt(selectedAttempt);
  };

  const updateChart = (data, predictionPrice) => {
    d3.select(chartRef.current).selectAll("*").remove(); // Clear previous chart

    const padding = 100;
    const w = 1000;
    const h = 400;

    // Define the color scale with higher contrast
    const colorScale = d3.scaleLinear()
      .domain([d3.min([data.Price_2020, data.Price_2021, data.Price_2022, data.Price_2023, predictionPrice || 0]), 
               d3.max([data.Price_2020, data.Price_2021, data.Price_2022, data.Price_2023, predictionPrice || 0])])
      .range(["#ffb3b3", "#ff0000"]); // Light pink to bright red

    const xScale = d3.scaleBand()
      .domain(["2020", "2021", "2022", "2023", "Prediction"])
      .range([padding, w - padding])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max([data.Price_2020, data.Price_2021, data.Price_2022, data.Price_2023, predictionPrice || 0])])
      .range([h - padding, padding]);

    const svg = d3.select(chartRef.current)
      .append("svg")
      .attr("width", w)
      .attr("height", h);

    const prices = [
      { year: "2020", value: data.Price_2020 },
      { year: "2021", value: data.Price_2021 },
      { year: "2022", value: data.Price_2022 },
      { year: "2023", value: data.Price_2023 }
    ];

    if (predictionPrice !== undefined) {
      prices.push({ year: "Prediction", value: predictionPrice });
    }

    svg.selectAll("rect")
      .data(prices)
      .enter()
      .append("rect")
      .attr("x", d => xScale(d.year))
      .attr("y", d => yScale(d.value))
      .attr("width", xScale.bandwidth())
      .attr("height", d => h - padding - yScale(d.value))
      .attr("fill", d => colorScale(d.value)); // Apply color scale

    // Add text labels
    svg.selectAll("text")
      .data(prices)
      .enter()
      .append("text")
      .attr("x", d => xScale(d.year) + xScale.bandwidth() / 2)
      .attr("y", d => yScale(d.value) - 5)
      .attr("text-anchor", "middle")
      .attr("fill", "black")
      .text(d => d.value);

    // Add x-axis
    svg.append("g")
      .attr("transform", `translate(0,${h - padding})`)
      .call(d3.axisBottom(xScale));

    // Add y-axis
    svg.append("g")
      .attr("transform", `translate(${padding},0)`)
      .call(d3.axisLeft(yScale));
  };

  return (
    <div>
      <select id="suburb-select"></select>
      <div id="chart" ref={chartRef}></div>
      <select onChange={handleAttemptChange}>
        <option value="">Select an attempt</option>
        {predictions.map(attempt => (
          <option key={attempt.id} value={attempt.id}>
            Attempt {attempt.id} - Predicted Price: {attempt.predicted_price}
          </option>
        ))}
      </select>
    </div>
  );
};

export default BarChartComponent;