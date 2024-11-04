// Set dimensions and constants for the gauge chart
var width = 300, height = 150, innerRadius = 60, outerRadius = 100;

// Set dimensions and margins for the histogram
var margin = {top: 20, right: 30, bottom: 40, left: 40},
    histWidth = 800 - margin.left - margin.right,
    histHeight = 400 - margin.top - margin.bottom;


// Append the SVG object for the histogram to the body of the page
var histSvg = d3.select("body").append("svg")
    .attr("width", histWidth + margin.left + margin.right)
    .attr("height", histHeight + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Load CSV data and initialize the chart
d3.csv("label_data.csv").then(data => {
    // Calculate the counts for "S" and "NS"
    var totalCount = data.length;
    var sCount = data.filter(d => d.Status === "S").length;
    var nsCount = data.filter(d => d.Status === "NS").length;

    // Calculate the percentages for each
    var sPercentage = (sCount / totalCount) * 100;
    var nsPercentage = (nsCount / totalCount) * 100;

    console.log("S Percentage:", sPercentage, "NS Percentage:", nsPercentage);

    // Draw the gauge with the calculated percentages
    drawGauge(sPercentage, nsPercentage);

    // Extract Property_value and convert to numbers
    var propertyValues = data.map(d => +d.Property_value);

    // Set the scales for the histogram
    var x = d3.scaleLinear()
    .domain([d3.min(propertyValues), d3.max(propertyValues)])
    .range([0, histWidth]);

    var histogram = d3.histogram()
    .domain(x.domain())
    .thresholds(x.ticks(20));

    var bins = histogram(propertyValues);

    var y = d3.scaleLinear()
    .domain([0, d3.max(bins, d => d.length)])
    .range([histHeight, 0]);

    // Add the X Axis for the histogram
    histSvg.append("g")
    .attr("transform", "translate(0," + histHeight + ")")
    .call(d3.axisBottom(x))
    .append("text")
    .attr("y", 30)
    .attr("x", histWidth / 2)
    .attr("text-anchor", "middle")
    .style("font-size", "14px")
    .text("Property Value");

    // Add the Y Axis for the histogram
    histSvg.append("g")
    .call(d3.axisLeft(y))
    .append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", -40)
    .attr("x", -histHeight / 2)
    .attr("text-anchor", "middle")
    .style("font-size", "14px")
    .text("Frequency");

    // Add the bars for the histogram
    histSvg.selectAll(".bar")
    .data(bins)
    .enter().append("rect")
    .attr("class", "bar")
    .attr("x", d => x(d.x0) + 1)
    .attr("y", d => y(d.length))
    .attr("width", d => x(d.x1) - x(d.x0) - 1)
    .attr("height", d => histHeight - y(d.length))
    .attr("fill", "steelblue");
}).catch(error => {
    console.error("Error loading CSV file:", error);
});

// Function to draw a single gauge with S (green) and NS (red)
function drawGauge(sPercentage, nsPercentage) {
    var svg = d3.select("#gauge-container").append("svg")
        .attr("width", width)
        .attr("height", height);

    // Define the arcs for S and NS
    var arcGenerator = d3.arc()
        .innerRadius(innerRadius)
        .outerRadius(outerRadius)
        .startAngle(-Math.PI / 2);

    // Draw "S" segment (Green)
    svg.append("path")
        .attr("d", arcGenerator.endAngle(-Math.PI / 2 + (sPercentage / 100) * Math.PI))
        .attr("transform", `translate(${width / 2},${height})`)
        .attr("fill", "green");

    // Draw "NS" segment (Red), starting where "S" ends
    svg.append("path")
        .attr("d", arcGenerator
            .startAngle(-Math.PI / 2 + (sPercentage / 100) * Math.PI)
            .endAngle(Math.PI / 2)
        )
        .attr("transform", `translate(${width / 2},${height})`)
        .attr("fill", "red");

    // Text labels for percentages
    svg.append("text")
        .attr("x", width / 2 - 60)
        .attr("y", height / 2 + 20)
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .text(`S: ${sPercentage.toFixed(1)}%`);

    svg.append("text")
        .attr("x", width / 2 + 60)
        .attr("y", height / 2 + 20)
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .text(`NS: ${nsPercentage.toFixed(1)}%`);
}
