// Fetch model results from API
async function fetchModelResults() {
    try {
        const response = await fetch('/api/model-results');
        const data = await response.json();

        // Populate model results
        const modelResultsDiv = document.getElementById('model-results');
        modelResultsDiv.innerHTML = `
            <ul>
                <li><strong>Linear Regression:</strong> Predictions calculated.</li>
                <li><strong>Random Forest:</strong> Predictions calculated.</li>
                <li><strong>Gradient Boosting Machine:</strong> Predictions calculated.</li>
            </ul>
        `;

        // Populate undervalued stocks
        const undervaluedTable = document.querySelector('#undervalued-table tbody');
        data.undervalued.forEach(stock => {
            undervaluedTable.innerHTML += `
                <tr>
                    <td>${stock.Symbol}</td>
                    <td>${stock.Intrinsic_Value}</td>
                    <td>${stock.Market_Price}</td>
                    <td>${stock.Method}</td>
                </tr>
            `;
        });

        // Populate overvalued stocks
        const overvaluedTable = document.querySelector('#overvalued-table tbody');
        data.overvalued.forEach(stock => {
            overvaluedTable.innerHTML += `
                <tr>
                    <td>${stock.Symbol}</td>
                    <td>${stock.Intrinsic_Value}</td>
                    <td>${stock.Market_Price}</td>
                    <td>${stock.Method}</td>
                </tr>
            `;
        });

        // Create the chart
        createChart(data.chartData);
    } catch (error) {
        console.error('Error fetching model results:', error);
    }
}

// Create chart visualization
function createChart(chartData) {
    const symbols = chartData.map(d => d.Symbol);
    const intrinsicValues = chartData.map(d => d.Intrinsic_Value);
    const marketPrices = chartData.map(d => d.Market_Price);

    const trace1 = {
        x: symbols,
        y: intrinsicValues,
        type: 'bar',
        name: 'Intrinsic Value',
        marker: { color: '#007bff' }
    };

    const trace2 = {
        x: symbols,
        y: marketPrices,
        type: 'bar',
        name: 'Market Price',
        marker: { color: '#ff5733' }
    };

    const layout = {
        title: 'Intrinsic Value vs Market Price',
        xaxis: { title: 'Stock Symbols' },
        yaxis: { title: 'Value' },
        barmode: 'group'
    };

    Plotly.newPlot('chart', [trace1, trace2], layout);
}

// Fetch and display results on page load
document.addEventListener('DOMContentLoaded', fetchModelResults);
