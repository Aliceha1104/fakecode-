import React, { useEffect, useState } from 'react';
import * as d3 from 'd3';
import dataCBD from '../data/denormalized_price_cbd.json'; // Adjust the path as needed
import dataLandsize from '../data/denormalized_price_landsize.json'; // Adjust the path as needed

const ScatterPlotComponent = () => {
    const [currentData, setCurrentData] = useState(dataCBD);
    const [xField, setXField] = useState('CBD Distance');
    const [yField, setYField] = useState('Price');
    const [isCBD, setIsCBD] = useState(true); // State to track which dataset is currently displayed

    const margin = { top: 20, right: 30, bottom: 70, left: 150 }; // Increased bottom and right margins
    const width = 800 - margin.left - margin.right;
    const height = 600 - margin.top - margin.bottom;

    useEffect(() => {
        // Set up SVG dimensions and groups
        const svg = d3.select("#scatterPlot")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const xScale = d3.scaleLinear().range([0, width]);
        const yScale = d3.scaleLinear().range([height, 0]);
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

        const xAxis = d3.axisBottom(xScale);
        const yAxis = d3.axisLeft(yScale);

        svg.append("g").attr("class", "x-axis").attr("transform", `translate(0,${height})`);
        svg.append("g").attr("class", "y-axis");

        svg.append("text")
            .attr("class", "x-axis-label axis-label")
            .attr("x", width / 2)
            .attr("y", height + 50)
            .attr("text-anchor", "middle");

        svg.append("text")
            .attr("class", "y-axis-label axis-label")
            .attr("x", -height / 2)
            .attr("y", -120)
            .attr("text-anchor", "middle")
            .attr("transform", "rotate(-90)");

        // Create tooltip and set initial styles
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("display", "none") // Initially hidden
            .style("position", "absolute")
            .style("background", "white")
            .style("border", "1px solid black")
            .style("padding", "10px")
            .style("border-radius", "5px")
            .style("pointer-events", "none"); // Prevent mouse events on the tooltip

        // Function to update chart for a given dataset
        const updateChart = (data) => {
            xScale.domain(d3.extent(data, d => d[xField]));
            yScale.domain(d3.extent(data, d => d[yField]));

            svg.select(".x-axis").transition().duration(1000).call(xAxis);
            svg.select(".y-axis").transition().duration(1000).call(yAxis);

            svg.select(".x-axis-label").text(xField);
            svg.select(".y-axis-label").text(yField);

            const circles = svg.selectAll("circle").data(data);

            circles.enter().append("circle")
                .attr("data-cluster", d => d.kmeans_3) // Add data attribute for cluster
                .attr("fill", d => colorScale(d.kmeans_3))
                .attr("r", 5) // Reduced radius to prevent overlap
                .merge(circles)
                .transition()
                .duration(1000)
                .attr("cx", d => xScale(d[xField]))
                .attr("cy", d => yScale(d[yField]));

            circles.exit().remove();

            // Add interactivity with detailed tooltip
            svg.selectAll("circle")
                .on("mouseover", function (event, d) {
                    d3.select(this)
                        .transition()
                        .duration(100)
                        .attr("r", 8) // Increase on hover for emphasis
                        .attr("stroke", "black")
                        .attr("stroke-width", 2);
                    tooltip.style("display", "block")
                        .html(`
                            <strong>Cluster:</strong> ${d.kmeans_3}<br>
                            <strong>${xField}:</strong> ${d[xField]}<br>
                            <strong>${yField}:</strong> ${d[yField].toLocaleString('en-US', { style: 'currency', currency: 'USD' })}
                        `);
                })
                .on("mousemove", function (event) {
                    tooltip.style("left", (event.pageX + 15) + "px")
                        .style("top", (event.pageY - 28) + "px");
                })
                .on("mouseout", function () {
                    d3.select(this)
                        .transition()
                        .duration(100)
                        .attr("r", 5) // Reset size on mouse out
                        .attr("stroke", "none");
                    tooltip.style("display", "none"); // Hide tooltip
                });

            // Cluster selection legend
            const clusters = [...new Set(data.map(d => d.kmeans_3))];

            const legend = svg.append("g")
                .attr("transform", `translate(${width - 150}, 10)`);

            clusters.forEach((cluster, index) => {
                legend.append("rect")
                    .attr("x", 0)
                    .attr("y", index * 25)
                    .attr("width", 20)
                    .attr("height", 20)
                    .attr("fill", colorScale(cluster))
                    .on("click", () => handleClusterClick(cluster));

                legend.append("text")
                    .attr("x", 30)
                    .attr("y", index * 25 + 15)
                    .text(`Group ${cluster}`)
                    .style("cursor", "pointer");
            });
        };

        const handleClusterClick = (cluster) => {
            const isActive = svg.selectAll("circle").filter(d => d.kmeans_3 === cluster).classed("highlighted");

            svg.selectAll("circle")
                .transition()
                .duration(500)
                .attr("opacity", isActive ? 1 : 0.2); // Fade out others if this is active

            svg.selectAll("circle.highlighted")
                .classed("highlighted", false);

            if (!isActive) {
                svg.selectAll(`circle[data-cluster='${cluster}']`)
                    .classed("highlighted", true)
                    .transition()
                    .duration(500)
                    .attr("opacity", 1); // Bring back highlighted ones
            }
        };

        // Initial chart load with the current data
        updateChart(currentData);

        // Zoom functionality
        const zoomHandler = d3.zoom()
            .scaleExtent([1, 10]) // Set the zoom limits
            .on("zoom", (event) => {
                const new_xScale = event.transform.rescaleX(xScale);
                const new_yScale = event.transform.rescaleY(yScale);

                svg.selectAll("circle")
                    .attr("cx", d => new_xScale(d[xField]))
                    .attr("cy", d => new_yScale(d[yField]));

                svg.select(".x-axis").call(xAxis.scale(new_xScale));
                svg.select(".y-axis").call(yAxis.scale(new_yScale));
            });

        svg.call(zoomHandler);

        // Cleanup on component unmount
        return () => {
            d3.select("#scatterPlot").select("svg").remove();
        };
    }, [currentData, xField, yField]); // Depend on currentData, xField, and yField

    // Function to handle dataset switch
    const handleSwitchData = () => {
        if (isCBD) {
            setCurrentData(dataLandsize);
            setXField('Landsize'); // Update xField for Landsize
        } else {
            setCurrentData(dataCBD);
            setXField('CBD Distance'); // Update xField for CBD
        }
        setIsCBD(!isCBD); // Toggle the dataset state
    };

    // Reset function to show all circles again
    const handleReset = () => {
        d3.selectAll("circle")
            .transition()
            .duration(500)
            .attr("opacity", 1); // Show all points
    };

    return (
        <div>
            <h1>Scatter plot for housing market distributions</h1>
            <button onClick={handleSwitchData}>
                Switch to {isCBD ? "Landsize" : "CBD"} Data
            </button>
            <button onClick={handleReset}>
                Reset Filter
            </button>
            <div id="scatterPlot"></div>
        </div>
    );
};

export default ScatterPlotComponent;
