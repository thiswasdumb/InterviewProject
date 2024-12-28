

//const API_KEY = "56RI009LMO7R6F91"; "JA2N4A0Z6CODVKUF"

// Replace with your actual API keys

const ALPHA_VANTAGE_API_KEY = "PVISE221PEPXLHYB";
const FINNHUB_API_KEY = "cthpu71r01qm2t954n3gcthpu71r01qm2t954n40";

// DOM Elements
const tickerInput = document.getElementById("stock-ticker");
const fetchButton = document.getElementById("fetch-data");
const suggestionsContainer = document.getElementById("suggestions");

// Function: Fetch ticker suggestions from Finnhub
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

// Function: Display ticker suggestions
function showSuggestions(suggestions) {
    suggestionsContainer.innerHTML = ""; // Clear previous suggestions
    if (suggestions.length === 0) {
        suggestionsContainer.style.display = "none";
        return;
    }

    suggestionsContainer.style.display = "block"; // Ensure container is visible
    suggestions.forEach(suggestion => {
        const li = document.createElement("li");
        li.textContent = `${suggestion.symbol} - ${suggestion.description}`;
        li.style.cursor = "pointer";
        li.style.padding = "0.5rem";
        li.style.borderBottom = "1px solid #ddd";

        // On click, populate input and clear suggestions
        li.addEventListener("click", () => {
            tickerInput.value = suggestion.symbol;
            suggestionsContainer.innerHTML = "";
            suggestionsContainer.style.display = "none";
        });

        suggestionsContainer.appendChild(li);
    });
}

// Event Listener: Fetch autocomplete suggestions
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

// Clear suggestions when clicking outside the input
document.addEventListener("click", (e) => {
    if (!e.target.closest("#stock-input")) {
        suggestionsContainer.innerHTML = "";
        suggestionsContainer.style.display = "none";
    }
});


// Event Listener: Fetch stock data on button click
fetchButton.addEventListener("click", () => {
    const ticker = tickerInput.value.toUpperCase();
    if (!ticker) {
        alert("Please enter a stock ticker!");
        return;
    }
    fetchStockData(ticker);
});

// Mock data for multiple stocks (Replace with real data)
let stockData = [];

// Function: Check if the stock already exists in the table
function isStockInTable(ticker) {
    const rows = document.querySelectorAll("#results-table tbody tr");
    for (const row of rows) {
        const cell = row.querySelector("td"); // First cell in the row
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

    // Main row for the stock
    const row = document.createElement("tr");
    row.innerHTML = `
        <td>${stock.ticker}</td>
        <td>${stock.companyName}</td>
        <td>${formatValue(stock.peRatio)}</td>
        <td>${formatValue(stock.pbRatio)}</td>
        <td>${formatValue(stock.roe)}</td>
        <td>${formatValue(stock.dividendYield)}</td>
    `;

    // Hidden expandable row for additional details
    const detailRow = document.createElement("tr");
    detailRow.style.display = "none";
    detailRow.innerHTML = `
        <td colspan="6" class="details-cell" style="padding: 10px; background-color: #333;">
            <div class="details-content" style="color: #d1e8e2; font-size: 0.9rem;">
                Loading details...
            </div>
        </td>
    `;

    // Add the main row and the detail row to the table
    tableBody.appendChild(row);
    tableBody.appendChild(detailRow);

    // Event listener for toggling the detail row
    row.addEventListener("click", async () => {
        if (detailRow.style.display === "none") {
            // Fetch and display details if the row is expanded
            const detailsContent = detailRow.querySelector(".details-content");
            detailsContent.textContent = "Loading details..."; // Reset loading state
            detailRow.style.display = ""; // Show the detail row

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
            detailRow.style.display = "none"; // Hide the detail row
        }
    });
}

// Function: Fetch detailed info
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

// Helper Function: Format date for news API
function getFormattedDate(daysAgo) {
    const date = new Date();
    date.setDate(date.getDate() - daysAgo);
    return date.toISOString().split("T")[0];
}


// Function to show modal
function showModal(ticker, details) {
    const modal = document.createElement("div");
    modal.style.position = "fixed";
    modal.style.top = "50%";
    modal.style.left = "50%";
    modal.style.transform = "translate(-50%, -50%)";
    modal.style.backgroundColor = "#444";
    modal.style.color = "white";
    modal.style.padding = "20px";
    modal.style.borderRadius = "8px";
    modal.style.zIndex = 1000;

    const modalContent = `
        <h2>${ticker} - ${details.profile.name || "N/A"}</h2>
        <p><strong>Industry:</strong> ${details.profile.finnhubIndustry || "N/A"}</p>
        <p><strong>Market Cap:</strong> ${details.profile.marketCapitalization || "N/A"}</p>
        <h3>Recent News</h3>
        <ul>
            ${details.news.slice(0, 3).map(news => `<li>${news.headline} - <a href="${news.url}" target="_blank">Read more</a></li>`).join("")}
        </ul>
        <button id="close-modal">Close</button>
    `;

    modal.innerHTML = modalContent;
    document.body.appendChild(modal);

    // Close modal event
    document.getElementById("close-modal").addEventListener("click", () => {
        modal.remove();
    });
}


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

        // Validate and update metrics
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


// Apply filters
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

// Function: Sort table by column
function sortTable(column, isNumeric = false) {
    const tableBody = document.querySelector("#results-table tbody");
    const rows = Array.from(tableBody.rows);

    // Clear existing sort indicators
    document.querySelectorAll("#results-table th").forEach(th => {
        th.textContent = th.textContent.replace(" ▲", "").replace(" ▼", "");
    });

    // Toggle sort direction
    const currentSortDirection = tableBody.getAttribute("data-sort-direction") || "asc";
    const newSortDirection = currentSortDirection === "asc" ? "desc" : "asc";
    tableBody.setAttribute("data-sort-direction", newSortDirection);

    // Update sort indicator
    const header = document.querySelector(`#results-table th:nth-child(${column + 1})`);
    header.textContent += newSortDirection === "asc" ? " ▲" : " ▼";

    rows.sort((a, b) => {
        const cellA = a.cells[column].textContent.trim();
        const cellB = b.cells[column].textContent.trim();

        if (isNumeric) {
            const numA = parseFloat(cellA) || 0;
            const numB = parseFloat(cellB) || 0;
            return newSortDirection === "asc" ? numA - numB : numB - numA;
        } else {
            return newSortDirection === "asc" ? cellA.localeCompare(cellB) : cellB.localeCompare(cellA);
        }
    });

    rows.forEach(row => tableBody.appendChild(row));
}


document.querySelectorAll("#results-table tbody tr").forEach(row => {
    row.addEventListener("mouseover", () => {
        row.style.backgroundColor = "#333";
    });
    row.addEventListener("mouseout", () => {
        row.style.backgroundColor = "";
    });
});
