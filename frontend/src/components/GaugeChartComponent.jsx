import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import Property_label from '../data/label_data'; // Import the data

const GaugeChartComponent = () => {
  const gaugeRef = useRef(null);
  const histogramRef = useRef(null);
  const [salesPredictions, setSalesPredictions] = useState([]);
  const [selectedAttempt, setSelectedAttempt] = useState(null);

  const width = 600, height = 150, innerRadius = 60, outerRadius = 100;
  const margin = { top: 40, right: 50, bottom: 40, left: 60 }; // Increased left margin
  const histWidth = 900 - margin.left - margin.right;
  const histHeight = 400 - margin.top - margin.bottom;

  const gaugeFillColor = "#FF69B4"; // Pink color for S
  const gaugeBackgroundColor = "#FF1493"; // Deep pink color for NS
  const histogramFillColor = "#FF6347"; // Tomato color for histogram
  const histogramLineColor = "#FF4500"; // Orange red color for selected line

  const fetchSalesPredictions = () => {
    fetch('http://127.0.0.1:8000/sale-prediction-history/')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        return response.json();
      })
      .then((data) => {
        setSalesPredictions(data.predictions);
      })
      .catch((error) => {
        alert('Error fetching sales predictions: ' + error.message);
      });
  };

  useEffect(() => {
    fetchSalesPredictions();
  }, []);

  useEffect(() => {
    updateCharts();
  }, [selectedAttempt, salesPredictions]);

  const drawGauge = (sPercentage, nsPercentage, selectedStatus) => {
    d3.select(gaugeRef.current).selectAll("*").remove(); // Clear previous SVG

    const svg = d3.select(gaugeRef.current).append("svg")
      .attr("width", width)
      .attr("height", height);

    const arcGenerator = d3.arc()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius)
      .startAngle(-Math.PI / 2);

    svg.append("path")
      .attr("d", arcGenerator.endAngle(-Math.PI / 2 + (sPercentage / 100) * Math.PI))
      .attr("transform", `translate(${width / 2},${height})`)
      .attr("fill", gaugeFillColor);

    svg.append("path")
      .attr("d", arcGenerator
        .startAngle(-Math.PI / 2 + (sPercentage / 100) * Math.PI)
        .endAngle(Math.PI / 2)
      )
      .attr("transform", `translate(${width / 2},${height})`)
      .attr("fill", gaugeBackgroundColor);

    svg.append("text")
      .attr("x", width / 2 - 200) // Adjusted x position
      .attr("y", height) // Adjusted y position
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .text(`Sold Percentage: ${sPercentage.toFixed(1)}%`);

    svg.append("text")
      .attr("x", width / 2 + 200) // Adjusted x position
      .attr("y", height) // Adjusted y position
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .text(`On Sale Percentage: ${nsPercentage.toFixed(1)}%`);

  };

  const drawHistogram = (propertyValues, selectedPropertyValue) => {
    const x = d3.scaleLinear()
      .domain([d3.min(propertyValues), d3.max(propertyValues)])
      .range([0, histWidth]);

    const histogram = d3.histogram()
      .domain(x.domain())
      .thresholds(x.ticks(20));

    const bins = histogram(propertyValues);

    const y = d3.scaleLinear()
      .domain([0, d3.max(bins, d => d.length)])
      .range([histHeight, 0]);

    d3.select(histogramRef.current).selectAll("*").remove(); // Clear previous SVG

    const histSvg = d3.select(histogramRef.current).append("svg")
      .attr("width", histWidth + margin.left + margin.right)
      .attr("height", histHeight + margin.top + margin.bottom + 70) // Increased height to accommodate text
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    histSvg.append("g")
      .attr("transform", "translate(0," + histHeight + ")")
      .call(d3.axisBottom(x))
      .append("text")
      .attr("y", 30)
      .attr("x", histWidth / 2)
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .text("Property Score"); // Updated x-axis label

    // Add Property Value text below the center of x-axis
    histSvg.append("text")
      .attr("y", histHeight + 50) // Adjusted y position to be below the x-axis
      .attr("x", histWidth / 2)
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .text("Property Value");

    const yAxis = histSvg.append("g")
      .call(d3.axisLeft(y));

    yAxis.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -margin.left + 20) // Adjusted y position to be on the left side
      .attr("x", -histHeight / 2)
      .attr("dy", "1em") // Adjusted dy to move the label away from the axis
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .text("Frequency"); // Updated y-axis label

    const area = d3.area()
      .x(d => x(d.x0))
      .y0(histHeight)
      .y1(d => y(d.length))
      .curve(d3.curveBasis);

    histSvg.append("path")
      .datum(bins)
      .attr("fill", histogramFillColor)
      .attr("opacity", 0.5)
      .attr("stroke", "none")
      .attr("d", area);

    if (selectedPropertyValue !== null) {
      histSvg.append("line")
        .attr("x1", x(selectedPropertyValue))
        .attr("x2", x(selectedPropertyValue))
        .attr("y1", histHeight)
        .attr("y2", 0)
        .attr("stroke", histogramLineColor)
        .attr("stroke-width", 2);

      histSvg.append("text")
        .attr("x", x(selectedPropertyValue))
        .attr("y", -10)
        .attr("text-anchor", "middle")
        .style("fill", histogramLineColor)
        .text(`Property Score: ${selectedPropertyValue}`);
    }
  };

  const updateCharts = () => {
    const data = Property_label;

    const totalCount = data.length;
    const sCount = data.filter(d => d.Status === "S").length;
    const nsCount = data.filter(d => d.Status === "NS").length;

    const sPercentage = (sCount / totalCount) * 100;
    const nsPercentage = (nsCount / totalCount) * 100;

    let selectedStatus = null;
    let selectedPropertyValue = null;

    if (selectedAttempt) {
      const attemptData = salesPredictions.find(
        (attempt) => attempt.id === parseInt(selectedAttempt)
      );
      if (attemptData) {
        selectedStatus = attemptData.predicted_status;
        selectedPropertyValue = attemptData.predicted_result;
      }
    }

    drawGauge(sPercentage, nsPercentage, selectedStatus);
    const propertyValues = data.map(d => +d.Property_value);
    drawHistogram(propertyValues, selectedPropertyValue);
  };

  return (
    <div>
      <h1>Sold, On Sale and Property Score Distribution</h1>
      <select onChange={(e) => setSelectedAttempt(e.target.value)}>
        <option value="">Select an attempt</option>
        {salesPredictions.map((attempt) => (
          <option key={attempt.id} value={attempt.id}>
            Attempt {attempt.id}
          </option>
        ))}
      </select>
      <div id="gauge-container" ref={gaugeRef}></div>
      <div id="histogram-container" ref={histogramRef}></div>
    </div>
  );
};

export default GaugeChartComponent;