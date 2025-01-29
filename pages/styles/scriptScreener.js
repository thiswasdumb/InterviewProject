// API Keys 
const ALPHA_VANTAGE_API_KEY = "PVISE221PEPXLHYB";
const FINNHUB_API_KEY = "cthpu71r01qm2t954n3gcthpu71r01qm2t954n40";

// DOM Elements in html
const tickerInput = document.getElementById("stock-ticker");
const fetchButton = document.getElementById("fetch-data");
const suggestionsContainer = document.getElementById("suggestions");

// Function calls Finnhub API to find potential tickers that can complete input so far
async function fetchSuggestions(query) {
    try {
        const response = await fetch(`https://finnhub.io/api/v1/search?q=${query}&token=${FINNHUB_API_KEY}`);
        const data = await response.json();
        return data.result || []; // Return array of results
    } catch (error) {
        console.error("Error fetching suggestions:", error);
        return [];
    }
}

// Function to display the suggestions found from the API
function showSuggestions(suggestions) {
    suggestionsContainer.innerHTML = ""; // Clear previous suggestions
    if (suggestions.length === 0) {
        suggestionsContainer.style.display = "none";
        return;
    }

    suggestionsContainer.style.display = "block"; // Makes sure suggestions show properly
    suggestions.forEach(suggestion => {
        const li = document.createElement("li");
        li.textContent = `${suggestion.symbol} - ${suggestion.description}`;
        li.style.cursor = "pointer";
        li.style.padding = "0.5rem";
        li.style.borderBottom = "1px solid #ddd";

        // If one of the suggestions is clicked then it fills the search (but i do not let it go through incase an incorrect one was clicked!)
        li.addEventListener("click", () => {
            tickerInput.value = suggestion.symbol;
            suggestionsContainer.innerHTML = "";
            suggestionsContainer.style.display = "none";
        });

        suggestionsContainer.appendChild(li);
    });
}

// An event listener which see if there is an input so it then can call suggestions
tickerInput.addEventListener("input", async () => {
    const query = tickerInput.value.trim();
    if (query.length < 1) {
        suggestionsContainer.innerHTML = "";
        suggestionsContainer.style.display = "none";
        return;
    }

    const suggestions = await fetchSuggestions(query);
    showSuggestions(suggestions);
});

// If the user clicks somewhere else, the suggestions go away (so they dont just stay on the screen forever that would be annoying)
document.addEventListener("click", (e) => {
    if (!e.target.closest("#stock-input")) {
        suggestionsContainer.innerHTML = "";
        suggestionsContainer.style.display = "none";
    }
});

// Event listener which checks input when stock is added and provides the data
fetchButton.addEventListener("click", async () => {
    const ticker = tickerInput.value.toUpperCase();
    if (!ticker) {
        alert("Please enter a stock ticker!");
        return;
    }

    await fetchStockData(ticker);

    // Clear the input field after fetching stock data
    tickerInput.value = "";

    // Also hide suggestions
    suggestionsContainer.innerHTML = "";
    suggestionsContainer.style.display = "none";
});


// Array for stock data potentially
let stockData = [];

// Function which checks if stock already exists in the table
function isStockInTable(ticker) {
    const rows = document.querySelectorAll("#results-table tbody tr");
    for (const row of rows) {
        const cell = row.querySelector("td");
        if (cell && cell.textContent === ticker) {
            return true;
        }
    }
    return false;
}


function formatValue(value, decimalPlaces = 2) {
    if (value === null || value === undefined || isNaN(value)) {
        return "N/A";
    }
    return parseFloat(value).toFixed(decimalPlaces);
}

// Function: Add stock to table
function addStockToTable(stock) {
    const tableBody = document.querySelector("#results-table tbody");

    // Avoid duplicates if inputted
    if (isStockInTable(stock.ticker)) {
        console.warn(`Ticker ${stock.ticker} is already in the table. Skipping duplicate.`);
        return; // Exit function if stock is already present
    }

    // Main row for the stock
    const row = document.createElement("tr");
    row.classList.add("stock-row");
    row.innerHTML = `
    <td>${stock.ticker}</td>
    <td>${stock.companyName}</td>
    <td>${formatValue(stock.peRatio)}</td>
    <td>${formatValue(stock.pbRatio)}</td>
    <td>${formatValue(stock.roe)}</td>
    <td>${formatValue(stock.dividendYield)}</td>
`;

    // Add a hidden expandable row that provides more details for each stock
    const detailRow = document.createElement("tr");
    detailRow.classList.add("details-row");
    detailRow.style.display = "none";
    detailRow.innerHTML = `
    <td colspan="6" class="details-cell" style="padding: 10px; background-color: #333;">
        <div class="details-content" style="color: #d1e8e2; font-size: 0.9rem;">
            Loading details...
        </div>
    </td>
`;

    // Add the main row and the detail row to the table (this is so when sorting occurs, they dont separate)
    tableBody.appendChild(row);
    tableBody.appendChild(detailRow);


    // Event listener for opening the expandable row
    row.addEventListener("click", async () => {
        if (detailRow.style.display === "none") {
            // Fetches and displays details if the row is expanded
            const detailsContent = detailRow.querySelector(".details-content");
            detailsContent.textContent = "Loading details..."; // Loading state if it is taking time
            detailRow.style.display = ""; // Display for the row

            const details = await fetchDetailedInfo(stock.ticker);
            if (details) {
                detailsContent.innerHTML = `
                    <strong>Industry:</strong> ${details.profile.finnhubIndustry || "N/A"}<br>
                    <strong>Market Cap:</strong> $${formatValue(details.profile.marketCapitalization)} Billion<br>
                    <strong>52-Week High:</strong> $${formatValue(details.keyStats["52WeekHigh"])}<br>
                    <strong>52-Week Low:</strong> $${formatValue(details.keyStats["52WeekLow"])}<br>
                    <strong>Beta:</strong> ${formatValue(details.keyStats.beta)}<br>
                    <strong>Recent News:</strong>
                    <ul>
                        ${details.news
                        .slice(0, 3)
                        .map(
                            news =>
                                `<li>${news.headline} - <a href="${news.url}" target="_blank" style="color: #66d9ef;">Read more</a></li>`
                        )
                        .join("")}
                    </ul>
                `;
            } else {
                detailsContent.textContent = "Failed to load details.";
            }
        } else {
            detailRow.style.display = "none";
        }
    });
}

// Function which basically calls APIs to get the detailed info
async function fetchDetailedInfo(ticker) {
    try {
        const profileResponse = await fetch(
            `https://finnhub.io/api/v1/stock/profile2?symbol=${ticker}&token=${FINNHUB_API_KEY}`
        );
        const profileData = await profileResponse.json();

        const statsResponse = await fetch(
            `https://finnhub.io/api/v1/stock/metric?symbol=${ticker}&metric=all&token=${FINNHUB_API_KEY}`
        );
        const statsData = await statsResponse.json();

        const newsResponse = await fetch(
            `https://finnhub.io/api/v1/company-news?symbol=${ticker}&from=${getFormattedDate(30)}&to=${getFormattedDate(0)}&token=${FINNHUB_API_KEY}`
        );
        const newsData = await newsResponse.json();

        return {
            profile: profileData,
            keyStats: statsData.metric || {},
            news: newsData,
        };
    } catch (error) {
        console.error("Error fetching detailed info:", error);
        return null;
    }
}

// Helper function for API calls
function getFormattedDate(daysAgo) {
    const date = new Date();
    date.setDate(date.getDate() - daysAgo);
    return date.toISOString().split("T")[0];
}

// Function to actually call API and get the data requested
async function fetchStockData(ticker) {
    try {
        // Fetch company info and metrics from Finnhub
        const profileResponse = await fetch(`https://finnhub.io/api/v1/stock/profile2?symbol=${ticker}&token=${FINNHUB_API_KEY}`);
        const profileData = await profileResponse.json();

        // Extract company name
        const companyName = profileData.name || "N/A";

        // Fetch fundamental metrics from Finnhub
        const metricsResponse = await fetch(`https://finnhub.io/api/v1/stock/metric?symbol=${ticker}&metric=all&token=${FINNHUB_API_KEY}`);
        const metricsData = await metricsResponse.json();

        if (metricsData && metricsData.metric) {
            const stock = {
                ticker: ticker,
                companyName: companyName,
                peRatio: metricsData.metric.peBasicExclExtraTTM,
                pbRatio: metricsData.metric.pbAnnual,
                roe: metricsData.metric.roeTTM,
                dividendYield: metricsData.metric.dividendYieldIndicatedAnnual,
            };

            addStockToTable(stock); // Add to table with company name
        } else {
            alert("No data found for the ticker. Please try another ticker.");
        }
    } catch (error) {
        console.error("Error fetching stock data:", error);
        alert("Failed to fetch stock data. Please try again.");
    }
}


// Apply filters to make sure outputs make sense
document.getElementById("apply-filters").addEventListener("click", () => {
    const minPE = parseFloat(document.getElementById("min-pe").value) || -Infinity;
    const maxPE = parseFloat(document.getElementById("max-pe").value) || Infinity;

    const filteredData = stockData.filter((stock) => {
        const pe = parseFloat(stock.peRatio);
        return pe >= minPE && pe <= maxPE;
    });

    const tableBody = document.querySelector("#results-table tbody");
    tableBody.innerHTML = ""; // Clear table
    filteredData.forEach(addStockToTable); // Refill table
});

// Event listener for fetching data
document.getElementById("fetch-data").addEventListener("click", () => {
    const ticker = tickerInput.value.toUpperCase();
    if (!ticker) {
        alert("Please enter a stock ticker!");
        return;
    }
    fetchStockData(ticker);
});

// Function that allows sorting based on column clicked
function sortTable(column, isNumeric = false) {
    const tableBody = document.querySelector("#results-table tbody");
    const rows = Array.from(tableBody.querySelectorAll("tr"));

    // Create an array of row pairs so that the expandable ro is basically a pair that wont leave its ticker
    let rowPairs = [];
    for (let i = 0; i < rows.length; i++) {
        if (rows[i].classList.contains("stock-row")) {
            let mainRow = rows[i];
            let detailRow = (i + 1 < rows.length && rows[i + 1].classList.contains("details-row")) ? rows[i + 1] : null;
            rowPairs.push([mainRow, detailRow]);
        }
    }

    // Clear existing sort indicators when column is clicked and replace
    document.querySelectorAll("#results-table th").forEach(th => {
        th.textContent = th.textContent.replace(" ▲", "").replace(" ▼", "");
    });

    // Toggle sort direction based on previous direction if one existed
    const currentSortDirection = tableBody.getAttribute("data-sort-direction") || "asc";
    const newSortDirection = currentSortDirection === "asc" ? "desc" : "asc";
    tableBody.setAttribute("data-sort-direction", newSortDirection);

    // Update sort indicator
    const header = document.querySelector(`#results-table th:nth-child(${column + 1})`);
    header.textContent += newSortDirection === "asc" ? " ▲" : " ▼";

    // Sort the row pairs while keeping details rows attached
    rowPairs.sort((a, b) => {
        let cellA = a[0].cells[column] ? a[0].cells[column].textContent.trim() : "";
        let cellB = b[0].cells[column] ? b[0].cells[column].textContent.trim() : "";

        if (isNumeric) {
            let numA = parseFloat(cellA);
            let numB = parseFloat(cellB);

            if (isNaN(numA)) return newSortDirection === "asc" ? 1 : -1;
            if (isNaN(numB)) return newSortDirection === "asc" ? -1 : 1;

            return newSortDirection === "asc" ? numA - numB : numB - numA;
        } else {
            return newSortDirection === "asc" ? cellA.localeCompare(cellB) : cellB.localeCompare(cellA);
        }
    });

    // Clear the table and append sorted rows back in order
    tableBody.innerHTML = "";
    rowPairs.forEach(pair => {
        tableBody.appendChild(pair[0]); // Append main stock row
        if (pair[1]) tableBody.appendChild(pair[1]); // Append detailed row if it is expanded
    });
}

// Styling rows so it is easy to put align numbers in table when hovering
document.querySelectorAll("#results-table tbody tr").forEach(row => {
    row.addEventListener("mouseover", () => {
        row.style.backgroundColor = "#333";
    });
    row.addEventListener("mouseout", () => {
        row.style.backgroundColor = "";
    });
});


