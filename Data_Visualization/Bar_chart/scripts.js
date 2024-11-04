var padding = 100; // Define padding
var w = 1000; // Define width
var h = 400; // Define height
var currentData; // Store the current data for the selected suburb

function init() {
    // Set the dataset for the chart with Suburbs and Median Rentals
    d3.csv("Median_price_suburb.csv", function(d) {
        return {
            suburb: d.Suburbs,
            Price_2020: +d["Median Price 2020"],
            Price_2021: +d["Median Price 2021"],
            Price_2022: +d["Median Price 2022"],
            Price_2023: +d["Median Price 2023"]
        };
    }).then(function(data) {
        populateDropdown(data);
        currentData = data[0]; // Initialize with the first suburb
        updateChart(currentData); // Initialize chart with the first suburb
        console.table(data, ["suburb", "Price_2020", "Price_2021", "Price_2022", "Price_2023"]);
    });
}

function populateDropdown(data) {
    var select = d3.select("#suburb-select");
    data.forEach(function(d) {
        select.append("option")
            .attr("value", d.suburb)
            .text(d.suburb);
    });

    select.on("change", function() {
        var selectedSuburb = d3.select(this).property("value");
        currentData = data.find(function(d) { return d.suburb === selectedSuburb; });
        updateChart(currentData);
    });
}

function updateChart(data, predictionPrice) {
    d3.select("#chart").selectAll("*").remove(); // Clear previous chart

    var xScale = d3.scaleBand()
        .domain(["2020", "2021", "2022", "2023", "Prediction"])
        .range([padding, w - padding])
        .padding(0.1);

    var yScale = d3.scaleLinear()
        .domain([0, d3.max([data.Price_2020, data.Price_2021, data.Price_2022, data.Price_2023, predictionPrice || 0])])
        .range([h - padding, padding]);

    var svg = d3.select("#chart")
        .append("svg")
        .attr("width", w)
        .attr("height", h);

    var prices = [
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
        .attr("x", function(d) { return xScale(d.year); })
        .attr("y", function(d) { return yScale(d.value); })
        .attr("width", xScale.bandwidth())
        .attr("height", function(d) { return h - padding - yScale(d.value); })
        .attr("fill", "steelblue");

    // Add x-axis
    svg.append("g")
        .attr("transform", "translate(0," + (h - padding) + ")")
        .call(d3.axisBottom(xScale));

    // Add y-axis
    svg.append("g")
        .attr("transform", "translate(" + padding + ",0)")
        .call(d3.axisLeft(yScale));
}

document.getElementById("update-chart").addEventListener("click", function() {
    var predictionPrice = parseFloat(document.getElementById("prediction-price").value);
    if (!isNaN(predictionPrice)) {
        updateChart(currentData, predictionPrice);
    }
});

init();