// Dashboard JavaScript for Chart.js visualizations

// Chart.js default configuration
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.plugins.legend.position = 'top';
Chart.defaults.plugins.legend.labels.usePointStyle = true;

// Color palette for charts
const colors = {
    primary: '#0d6efd',
    secondary: '#6c757d',
    success: '#198754',
    info: '#0dcaf0',
    warning: '#ffc107',
    danger: '#dc3545',
    light: '#f8f9fa',
    dark: '#212529'
};

// Chart instances
let actualVsPredictedChart = null;
let anomalyChart = null;
let cityAnalysisChart = null;

/**
 * Initialize all charts with provided data
 */
function initializeCharts(chartData) {
    console.log('Initializing charts with data:', chartData);
    
    // Initialize Actual vs Predicted Chart
    if (chartData.actual_vs_predicted) {
        initializeActualVsPredictedChart(chartData.actual_vs_predicted);
    }
    
    // Initialize Anomaly Distribution Chart
    if (chartData.anomaly_distribution) {
        initializeAnomalyChart(chartData.anomaly_distribution);
    }
    
    // Initialize City Analysis Chart
    if (chartData.city_analysis && chartData.city_analysis.labels) {
        initializeCityAnalysisChart(chartData.city_analysis);
    }
}

/**
 * Initialize Actual vs Predicted delivery times chart
 */
function initializeActualVsPredictedChart(data) {
    const ctx = document.getElementById('actualVsPredictedChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (actualVsPredictedChart) {
        actualVsPredictedChart.destroy();
    }
    
    actualVsPredictedChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels,
            datasets: [
                {
                    label: 'Actual Delivery Time',
                    data: data.actual,
                    borderColor: colors.danger,
                    backgroundColor: colors.danger + '20',
                    pointBackgroundColor: colors.danger,
                    pointBorderColor: colors.danger,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Predicted Delivery Time',
                    data: data.predicted,
                    borderColor: colors.primary,
                    backgroundColor: colors.primary + '20',
                    pointBackgroundColor: colors.primary,
                    pointBorderColor: colors.primary,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Delivery Time Prediction Accuracy',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: colors.primary,
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + ' hours';
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Delivery Records',
                        font: {
                            weight: 'bold'
                        }
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Delivery Time (hours)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

/**
 * Initialize Anomaly Distribution chart
 */
function initializeAnomalyChart(data) {
    const ctx = document.getElementById('anomalyChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (anomalyChart) {
        anomalyChart.destroy();
    }
    
    anomalyChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal Deliveries', 'Anomalies'],
            datasets: [{
                data: [data.normal, data.anomalies],
                backgroundColor: [
                    colors.success,
                    colors.warning
                ],
                borderColor: [
                    colors.success,
                    colors.warning
                ],
                borderWidth: 2,
                hoverOffset: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Delivery Status Distribution',
                    font: {
                        size: 14,
                        weight: 'bold'
                    }
                },
                legend: {
                    display: true,
                    position: 'bottom'
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: colors.primary,
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return context.label + ': ' + context.parsed + ' (' + percentage + '%)';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Initialize City Analysis chart
 */
function initializeCityAnalysisChart(data) {
    const ctx = document.getElementById('cityAnalysisChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (cityAnalysisChart) {
        cityAnalysisChart.destroy();
    }
    
    cityAnalysisChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Average Delivery Time (hours)',
                data: data.data,
                backgroundColor: [
                    colors.primary,
                    colors.info,
                    colors.success,
                    colors.warning,
                    colors.secondary
                ],
                borderColor: [
                    colors.primary,
                    colors.info,
                    colors.success,
                    colors.warning,
                    colors.secondary
                ],
                borderWidth: 2,
                borderRadius: 4,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Delivery Performance by City',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: colors.primary,
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return 'Avg Time: ' + context.parsed.y.toFixed(2) + ' hours';
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Cities',
                        font: {
                            weight: 'bold'
                        }
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Average Delivery Time (hours)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

/**
 * Update charts with new data
 */
function updateCharts(newData) {
    if (actualVsPredictedChart && newData.actual_vs_predicted) {
        actualVsPredictedChart.data = newData.actual_vs_predicted;
        actualVsPredictedChart.update();
    }
    
    if (anomalyChart && newData.anomaly_distribution) {
        anomalyChart.data.datasets[0].data = [
            newData.anomaly_distribution.normal,
            newData.anomaly_distribution.anomalies
        ];
        anomalyChart.update();
    }
    
    if (cityAnalysisChart && newData.city_analysis) {
        cityAnalysisChart.data = newData.city_analysis;
        cityAnalysisChart.update();
    }
}

/**
 * Animate number counting for statistics cards
 */
function animateNumbers() {
    const numberElements = document.querySelectorAll('[data-animate="number"]');
    
    numberElements.forEach(element => {
        const finalValue = parseInt(element.textContent.replace(/,/g, ''));
        const duration = 2000; // 2 seconds
        const startTime = Date.now();
        
        function updateNumber() {
            const currentTime = Date.now();
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const currentValue = Math.floor(finalValue * progress);
            element.textContent = currentValue.toLocaleString();
            
            if (progress < 1) {
                requestAnimationFrame(updateNumber);
            }
        }
        
        updateNumber();
    });
}

/**
 * Handle responsive chart resizing
 */
function handleResize() {
    if (actualVsPredictedChart) actualVsPredictedChart.resize();
    if (anomalyChart) anomalyChart.resize();
    if (cityAnalysisChart) cityAnalysisChart.resize();
}

// Event listeners
window.addEventListener('resize', handleResize);

// Initialize animations when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Initialize tooltips if Bootstrap is loaded
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});

// Export functions for use in templates
window.dashboardCharts = {
    initializeCharts,
    updateCharts,
    animateNumbers
};
