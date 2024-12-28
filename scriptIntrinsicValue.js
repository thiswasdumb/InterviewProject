// Replace with your actual Finnhub API key
const FINNHUB_API_KEY = "cthpu71r01qm2t954n3gcthpu71r01qm2t954n40";
const ALPHA_VANTAGE_API_KEY = "PVISE221PEPXLHYB";

// DOM Elements
const intrinsicValueDisplay = document.getElementById("intrinsic-value");
const marketValueDisplay = document.getElementById("market-value");
const chartContainer = document.getElementById("chart-container");

const calculationDetails = {
    epsValue: document.getElementById("eps-value"),
    growthRateValue: document.getElementById("growth-rate-value"),
    discountRateValue: document.getElementById("discount-rate-value"),
};

const userInputs = {
    eps: document.getElementById("user-eps"),
    growthRate: document.getElementById("user-growth-rate"),
    discountRate: document.getElementById("user-discount-rate"),
};

const calculateUserBtn = document.getElementById("calculate-user-btn");

let originalValues = {
    intrinsicValue: null,
    marketValue: null,
};

const fetchTickerBtn = document.getElementById("fetch-ticker-btn");
const tickerInput = document.getElementById("ticker-input");

fetchTickerBtn.addEventListener("click", () => {
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) {
        alert("Please enter a valid stock ticker!");
        return;
    }
    calculateAndDisplayIntrinsicValue(ticker);
});


// Function: Fetch market value using Finnhub
async function fetchMarketValue(ticker) {
    try {
        const response = await fetch(`https://finnhub.io/api/v1/quote?symbol=${ticker}&token=${FINNHUB_API_KEY}`);
        const data = await response.json();
        return data.c || null;
    } catch (error) {
        console.error("Error fetching market value:", error);
        return null;
    }
}

// Function: Fetch EPS for intrinsic value calculation
async function fetchEPS(ticker) {
    try {
        const response = await fetch(`https://finnhub.io/api/v1/stock/metric?symbol=${ticker}&metric=all&token=${FINNHUB_API_KEY}`);
        const data = await response.json();
        return data.metric.epsTTM || null;
    } catch (error) {
        console.error("Error fetching EPS:", error);
        return null;
    }
}

// Function: Calculate intrinsic value
function calculateIntrinsicValue(eps, growthRate = 10, discountRate = 8, years = 10) {
    let intrinsicValue = 0;
    for (let year = 1; year <= years; year++) {
        const projectedEPS = eps * Math.pow(1 + growthRate / 100, year);
        const discountedValue = projectedEPS / Math.pow(1 + discountRate / 100, year);
        intrinsicValue += discountedValue;
    }
    return intrinsicValue.toFixed(2);
}

function plotIntrinsicValueComparison(intrinsicValue, marketValue, scenarios, userIntrinsicValue = null) {
    const data = [
        {
            x: [intrinsicValue],
            y: ["Intrinsic Value"],
            type: "bar",
            orientation: "h",
            marker: { color: "#66d9ef" },
            hovertext: [
                `Value: $${intrinsicValue}<br>Growth Rate: 10%<br>Discount Rate: 8%`,
            ],
            hoverinfo: "text",
        },
        {
            x: [marketValue],
            y: ["Market Value"],
            type: "bar",
            orientation: "h",
            marker: { color: "#fd971f" },
            hovertext: [`Value: $${marketValue}<br>Market Value (current)`],
            hoverinfo: "text",
        },
    ];

    // Add scenario bars with hover text
    Object.keys(scenarios).forEach((scenario) => {
        const { value, growthRate, discountRate } = scenarios[scenario];
        data.push({
            x: [value],
            y: [scenario],
            type: "bar",
            orientation: "h",
            marker: {
                color:
                    scenario === "Underoptimistic"
                        ? "#e74c3c"
                        : scenario === "Overoptimistic"
                            ? "#2ecc71"
                            : "#f1c40f",
            },
            hovertext: [
                `Value: $${value}<br>Growth Rate: ${growthRate}%<br>Discount Rate: ${discountRate}%`,
            ],
            hoverinfo: "text",
        });
    });

    // Add user-defined bar if available
    if (userIntrinsicValue !== null) {
        data.push({
            x: [userIntrinsicValue],
            y: ["Custom Value"],
            type: "bar",
            orientation: "h",
            marker: { color: "#82e0aa" },
            hovertext: [
                `Value: $${userIntrinsicValue}<br>User-defined Growth and Discount Rates`,
            ],
            hoverinfo: "text",
        });
    }

    const layout = {
        title: "Intrinsic Value Scenarios vs Market Value",
        xaxis: { title: "Value (USD)" },
        yaxis: { automargin: true },
        paper_bgcolor: "#2c3531",
        plot_bgcolor: "#2c3531",
        font: { color: "white" },
    };

    Plotly.newPlot(chartContainer, data, layout);
}


// Define industry-specific growth and discount rates
const industryRates = {
    Technology: { growthRate: 12, discountRate: 9 },
    Healthcare: { growthRate: 10, discountRate: 8 },
    "Consumer Discretionary": { growthRate: 9, discountRate: 8.5 },
    Finance: { growthRate: 6, discountRate: 11 },
    Energy: { growthRate: 5, discountRate: 12 },
    "Consumer Staples": { growthRate: 7, discountRate: 9 },
    Utilities: { growthRate: 4, discountRate: 10 },
    Materials: { growthRate: 8, discountRate: 9 },
    Industrials: { growthRate: 7, discountRate: 10 },
};

// Function: Fetch company profile to determine industry
async function fetchCompanyProfile(ticker) {
    try {
        const response = await fetch(`https://finnhub.io/api/v1/stock/profile2?symbol=${ticker}&token=${FINNHUB_API_KEY}`);
        const data = await response.json();
        return data.finnhubIndustry || "Unknown";
    } catch (error) {
        console.error("Error fetching company profile:", error);
        return "Unknown";
    }
}

// User-defined calculation
calculateUserBtn.addEventListener("click", () => {
    const eps = parseFloat(userInputs.eps.value);
    const growthRate = parseFloat(userInputs.growthRate.value) || 10;
    const discountRate = parseFloat(userInputs.discountRate.value) || 8;

    if (!eps || isNaN(eps)) {
        alert("Please enter a valid EPS value!");
        return;
    }

    const userIntrinsicValue = calculateIntrinsicValue(eps, growthRate, discountRate);

    // Ensure original values remain in the chart along with scenarios
    plotIntrinsicValueComparison(
        originalValues.intrinsicValue,
        originalValues.marketValue,
        originalValues.scenarios,
        userIntrinsicValue // Add user-defined bar
    );
});

// Main function: Fetch and display values
async function calculateAndDisplayIntrinsicValue(ticker) {
    try {
        const [eps, marketValue, industry] = await Promise.all([
            fetchEPS(ticker),
            fetchMarketValue(ticker),
            fetchCompanyProfile(ticker),
        ]);

        if (!eps || !marketValue) {
            alert("Failed to fetch EPS or market value. Please check the ticker or try again later.");
            return;
        }

        // Determine industry-specific rates
        const industryGrowthRate = industryRates[industry]?.growthRate || 8; // Default to 8% if not found
        const industryDiscountRate = industryRates[industry]?.discountRate || 10; // Default to 10%

        const intrinsicValue = calculateIntrinsicValue(eps);

        // Scenarios with additional details
        const scenarios = {
            Underoptimistic: {
                value: calculateIntrinsicValue(eps, 5, 12), // Conservative assumptions
                growthRate: 5,
                discountRate: 12,
            },
            Overoptimistic: {
                value: calculateIntrinsicValue(eps, 15, 6), // Aggressive assumptions
                growthRate: 15,
                discountRate: 6,
            },
            "Industry Standard": {
                value: calculateIntrinsicValue(eps, industryGrowthRate, industryDiscountRate),
                growthRate: industryGrowthRate,
                discountRate: industryDiscountRate,
            },
        };

        // Save original values for later use
        originalValues.intrinsicValue = intrinsicValue;
        originalValues.marketValue = marketValue;
        originalValues.scenarios = scenarios;

        // Update DOM
        intrinsicValueDisplay.textContent = `$${intrinsicValue}`;
        marketValueDisplay.textContent = `$${marketValue}`;
        calculationDetails.epsValue.textContent = `$${parseFloat(eps).toFixed(2)}`;
        calculationDetails.growthRateValue.textContent = `${industryGrowthRate}%`;
        calculationDetails.discountRateValue.textContent = `${industryDiscountRate}%`;

        // Plot chart with scenarios
        plotIntrinsicValueComparison(
            intrinsicValue,
            marketValue,
            scenarios
        );
    } catch (error) {
        console.error("Error calculating intrinsic value:", error);
        alert("An error occurred. Please try again later.");
    }
}
