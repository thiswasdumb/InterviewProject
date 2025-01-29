document.addEventListener('DOMContentLoaded', function () {
    Papa.parse('../sp500_analysis.csv', {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: function (results) {
            let data = results.data;

            // Filter out entries with invalid or missing Symbol properties
            data = data.filter(row => row.Symbol && typeof row.Symbol === 'string');

            // Sort data alphabetically by the 'Symbol' property (this being the Ticker)
            data.sort((a, b) => {
                if (a.Symbol < b.Symbol) return -1;
                if (a.Symbol > b.Symbol) return 1;
                return 0;
            });

            // Compute model effectiveness
            calculateBaselineModelEffectiveness(data);
            calculateNeuralNetworkEffectiveness();

            displayModelPredictions(data);
            displayOverUnderValuedStocks(data);
            calculateAndDisplayIndustryOvervaluation();
        },
    });

    // Load industry data and intrinsic values already precomputed in csv files
    const industryFiles = [
        { file: '../../batch-computed-models/Consumer Discretionary_predictions.csv', name: 'Consumer Discretionary' },
        { file: '../../batch-computed-models/Consumer Staples_predictions.csv', name: 'Consumer Staples' },
        { file: '../../batch-computed-models/Energy_predictions.csv', name: 'Energy' },
        { file: '../../batch-computed-models/Financials_predictions.csv', name: 'Financials' },
        { file: '../../batch-computed-models/hc_predictions.csv', name: 'Healthcare' },
        { file: '../../batch-computed-models/Industrials_predictions.csv', name: 'Industrials' },
        { file: '../../batch-computed-models/it_industry_predictions.csv', name: 'Information Technology' },
        { file: '../../batch-computed-models/Materials_predictions.csv', name: 'Materials' },
        { file: '../../batch-computed-models/Real Estate_predictions.csv', name: 'Real Estate' },
        { file: '../../batch-computed-models/Telecommunication Services_predictions.csv', name: 'Telecommunication Services' },
        { file: '../../batch-computed-models/Utilities_predictions.csv', name: 'Utilities' }
    ];

    // Function to compute Baseline Model effectiveness (MAE)
    function calculateBaselineModelEffectiveness(data) {
        const models = [
            { key: 'Linear Regression_Prediction', name: 'Linear Regression' },
            { key: 'Random Forest_Prediction', name: 'Random Forest' },
            { key: 'Gradient Boosting Machine_Prediction', name: 'Gradient Boosting' },
        ];

        let modelErrors = {};

        models.forEach(model => {
            let totalError = 0;
            let count = 0;

            data.forEach(row => {
                if (row.Intrinsic_Value && row[model.key]) {
                    const error = Math.abs(row[model.key] - row.Intrinsic_Value);
                    totalError += error;
                    count++;
                }
            });

            modelErrors[model.name] = count > 0 ? (totalError / count).toFixed(2) : 'N/A';
        });

        displayModelEffectiveness(modelErrors, "Baseline Model MAE");
    }

    // Function to compute Neural Network effectiveness (MAE)
    function calculateNeuralNetworkEffectiveness() {
        let industryErrors = {};
        let industryFilesProcessed = 0;

        industryFiles.forEach(industry => {
            Papa.parse(industry.file, {
                download: true,
                header: true,
                dynamicTyping: true,
                complete: function (results) {
                    let data = results.data;
                    let totalError = 0;
                    let count = 0;

                    data.forEach(row => {
                        if (row['Actual Close'] && row['Predicted Close']) {
                            const error = Math.abs(row['Predicted Close'] - row['Actual Close']);
                            totalError += error;
                            count++;
                        }
                    });

                    industryErrors[industry.name] = count > 0 ? (totalError / count).toFixed(2) : 'N/A';
                    industryFilesProcessed++;

                    // Display once all industries are processed
                    if (industryFilesProcessed === industryFiles.length) {
                        displayModelEffectiveness(industryErrors, "Neural Network MAE");
                    }
                }
            });
        });
    }

    // Function to display the model effectiveness at the top of the page
    function displayModelEffectiveness(errors, title) {
        const insightsContainer = document.getElementById('model-effectiveness');

        const section = document.createElement('div');
        section.className = 'model-effectiveness-section';

        const heading = document.createElement('h3');
        heading.textContent = title;
        section.appendChild(heading);

        const table = document.createElement('table');
        table.className = 'model-effectiveness-table';

        const thead = document.createElement('thead');
        thead.innerHTML = `<tr><th>Model / Industry</th><th>Mean Absolute Error (MAE)</th></tr>`;
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        Object.entries(errors).forEach(([key, value]) => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${key}</td><td>${value}</td>`;
            tbody.appendChild(row);
        });

        table.appendChild(tbody);
        section.appendChild(table);
        insightsContainer.appendChild(section);
    }

    Papa.parse('../../batch-computed-models/intrinsic_value_predictions.csv', {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: function (results) {
            const intrinsicValues = results.data;

            // Load and display industry data
            loadIndustryData(intrinsicValues);
        },
    });

    function calculateAndDisplayIndustryOvervaluation() {
        const industryOvervaluations = [];
        let industryFilesProcessed = 0;

        // Parse intrinsic values once and store them in a map
        Papa.parse('../../batch-computed-models/intrinsic_value_predictions.csv', {
            download: true,
            header: true,
            dynamicTyping: true,
            complete: function (results) {
                const intrinsicValuesMap = results.data.reduce((map, row) => {
                    if (row.Ticker && row['Predicted Next Intrinsic Value']) {
                        map[row.Ticker] = parseFloat(row['Predicted Next Intrinsic Value']);
                    }
                    return map;
                }, {});

                // Now process each industry file
                industryFiles.forEach(industry => {
                    Papa.parse(industry.file, {
                        download: true,
                        header: true,
                        dynamicTyping: true,
                        complete: function (results) {
                            const data = results.data;

                            let totalOvervaluation = 0;
                            let validStockCount = 0;

                            data.forEach(row => {
                                if (row.Ticker && row['Predicted Close']) {
                                    const predictedClose = parseFloat(row['Predicted Close']);
                                    const predictedIntrinsic = intrinsicValuesMap[row.Ticker];

                                    if (predictedIntrinsic && predictedIntrinsic > 0) {
                                        const overvaluation = ((predictedClose - predictedIntrinsic) / predictedIntrinsic) * 100;
                                        totalOvervaluation += overvaluation;
                                        validStockCount++;
                                    }
                                }
                            });

                            if (validStockCount > 0) {
                                const averageOvervaluation = totalOvervaluation / validStockCount;
                                industryOvervaluations.push({
                                    industry: industry.name,
                                    overvaluation: averageOvervaluation
                                });
                            }

                            industryFilesProcessed++;

                            // When all files are processed, display the bar chart
                            if (industryFilesProcessed === industryFiles.length) {
                                displayIndustryOvervaluationChart(industryOvervaluations);
                            }
                        }
                    });
                });
            }
        });
    }

    function displayIndustryOvervaluationChart(industryOvervaluations) {
        if (industryOvervaluations.length === 0) {
            console.warn('No data available for the chart');
            return; // Exit if no data is available
        }

        // Sort industries by overvaluation for better visualization
        industryOvervaluations.sort((a, b) => b.overvaluation - a.overvaluation);

        // Extract data for the chart
        const industries = industryOvervaluations.map(item => item.industry);
        const overvaluationValues = industryOvervaluations.map(item => item.overvaluation);

        // Create the bar chart
        const ctx = document.getElementById('industry-overvaluation-chart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: industries,
                datasets: [{
                    label: 'Average Overvaluation (%)',
                    data: overvaluationValues,
                    backgroundColor: '#66d9ef',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Average Overvaluation by Industry',
                        color: 'white',
                        font: {
                            size: 24
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: 'white',
                            font: {
                                size: 12
                            }
                        },
                        title: {
                            display: true,
                            text: 'Overvaluation (%)',
                            color: 'white'
                        }
                    },
                    x: {
                        ticks: {
                            color: 'white',
                            font: {
                                size: 12
                            }
                        },
                        title: {
                            display: true,
                            text: 'Industry',
                            color: 'white',
                            font: {
                                size: 18
                            }
                        }
                    }
                }
            }
        });
    }

    // Loading neural network data
    function loadIndustryData(intrinsicValues) {
        const industrySection = document.getElementById('industry-predictions');

        industryFiles.forEach(industry => {
            Papa.parse(industry.file, {
                download: true,
                header: true,
                dynamicTyping: true,
                complete: function (results) {
                    const data = results.data;

                    // Create a div for each industry section
                    const industryDiv = document.createElement('div');
                    industryDiv.classList.add('industry-section');

                    const industryTitle = document.createElement('h3');
                    industryTitle.textContent = industry.name;
                    industryDiv.appendChild(industryTitle);

                    const table = document.createElement('table');
                    const thead = document.createElement('thead');
                    thead.innerHTML = `
                        <tr>
                            <th>Ticker</th>
                            <th>Predicted Close</th>
                            <th>Predicted Intrinsic Value</th>
                        </tr>
                    `;
                    table.appendChild(thead);

                    const tbody = document.createElement('tbody');
                    table.appendChild(tbody);
                    industryDiv.appendChild(table);

                    // "Show More" button
                    const showMoreButton = document.createElement('button');
                    showMoreButton.textContent = 'Show More';
                    showMoreButton.className = 'show-more-button'; // Assign a class for styling
                    industryDiv.appendChild(showMoreButton);

                    let displayedCount = 0;
                    const displayIncrement = 10;

                    function loadMoreRows() {
                        const rowsToDisplay = data.slice(displayedCount, displayedCount + displayIncrement);
                        rowsToDisplay.forEach(row => {
                            const intrinsicValueRow = intrinsicValues.find(iv => iv.Ticker === row.Ticker);
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td>${row.Ticker}</td>
                                <td>${formatToTwoDecimalPlaces(row['Predicted Close'])}</td>
                                <td>${intrinsicValueRow ? formatToTwoDecimalPlaces(intrinsicValueRow['Predicted Next Intrinsic Value']) : 'N/A'}</td>
                            `;
                            tbody.appendChild(tr);
                        });

                        if (tbody.rows.length > 0) {
                            tbody.deleteRow(tbody.rows.length - 1);
                        }

                        displayedCount += rowsToDisplay.length;

                        // Hide the "Show More" button if all rows are displayed
                        if (displayedCount >= data.length) {
                            showMoreButton.style.display = 'none';
                        }
                    }

                    // Load the initial set of rows
                    loadMoreRows();

                    // Add event listener to the "Show More" button
                    showMoreButton.addEventListener('click', loadMoreRows);

                    // Append the industry section to the main section
                    industrySection.appendChild(industryDiv);
                }
            });
        });
    }
});

// Function to format numbers to two decimal places
function formatToTwoDecimalPlaces(number) {
    return number !== null && number !== undefined ? number.toFixed(2) : 'N/A';
}
function displayModelPredictions(data) {
    const models = [
        { id: 'linear-regression-table', key: 'Linear Regression_Prediction' },
        { id: 'random-forest-table', key: 'Random Forest_Prediction' },
        { id: 'gradient-boosting-table', key: 'Gradient Boosting Machine_Prediction' },
    ];

    const displayIncrement = 10;

    models.forEach((model) => {
        const tableBody = document.getElementById(model.id).getElementsByTagName('tbody')[0];
        let displayedCount = 0;

        function loadMoreRows() {
            const rowsToDisplay = data.slice(displayedCount, displayedCount + displayIncrement);
            rowsToDisplay.forEach((row) => {
                const intrinsicValue = row["Intrinsic_Value"] !== undefined && row["Intrinsic_Value"] !== null
                    ? parseFloat(row["Intrinsic_Value"]).toFixed(2)
                    : 'N/A';

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.Symbol}</td>
                    <td>${intrinsicValue}</td>
                    <td>${formatToTwoDecimalPlaces(row[model.key])}</td>
                `;
                tableBody.appendChild(tr);
            });
            displayedCount += rowsToDisplay.length;

            // Hide the "Show More" button if all rows are displayed
            if (displayedCount >= data.length) {
                showMoreButton.style.display = 'none';
            }
        }

        // Initially load the first set of rows
        loadMoreRows();

        const showMoreButton = document.createElement('button');
        showMoreButton.textContent = 'Show More';
        showMoreButton.className = 'show-more-button'; // Assign a class for styling
        showMoreButton.addEventListener('click', loadMoreRows);
        tableBody.parentElement.appendChild(showMoreButton);
    });
}

// Function to display overvalued and undervalued stocks in table
function displayOverUnderValuedStocks(data) {
    // Load the intrinsic values from the CSV
    Papa.parse('../../batch-computed-models/intrinsic_value_predictions.csv', {
        download: true,
        header: true,
        complete: function (results) {
            const intrinsicValues = results.data.reduce((map, row) => {
                map[row.Ticker] = parseFloat(row['Most Recent Intrinsic Value']);
                return map;
            }, {});

            const undervaluedTableBody = document.getElementById('undervalued-table').getElementsByTagName('tbody')[0];
            const overvaluedTableBody = document.getElementById('overvalued-table').getElementsByTagName('tbody')[0];

            // Filter and calculate the difference for undervalued stocks
            const undervaluedStocks = data
                .filter(row => row.Undervalued)
                .map(row => {
                    const intrinsicValue = intrinsicValues[row.Symbol] || 0;
                    return {
                        ...row,
                        Intrinsic_Value: intrinsicValue,
                        difference: intrinsicValue - row.Market_Price
                    };
                })
                .sort((a, b) => b.difference - a.difference) // Sort by difference in descending order
                .slice(0, 5); // Select top 5

            // Filter and calculate the difference for overvalued stocks
            const overvaluedStocks = data
                .filter(row => row.Overvalued)
                .map(row => {
                    const intrinsicValue = intrinsicValues[row.Symbol] || 0;
                    return {
                        ...row,
                        Intrinsic_Value: intrinsicValue,
                        difference: row.Market_Price - intrinsicValue
                    };
                })
                .sort((a, b) => b.difference - a.difference) // Sort by difference in descending order
                .slice(0, 5); // Select top 5

            // Display undervalued stocks
            undervaluedStocks.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.Symbol}</td>
                    <td>${row.Intrinsic_Value.toFixed(2)}</td>
                    <td>${row.Market_Price.toFixed(2)}</td>
                    <td>${row.difference.toFixed(2)}</td>
                `;
                undervaluedTableBody.appendChild(tr);
            });

            // Display overvalued stocks
            overvaluedStocks.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.Symbol}</td>
                    <td>${row.Intrinsic_Value.toFixed(2)}</td>
                    <td>${row.Market_Price.toFixed(2)}</td>
                    <td>${row.difference.toFixed(2)}</td>
                `;
                overvaluedTableBody.appendChild(tr);
            });
        }
    });
}
