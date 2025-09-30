// SyncFit Dashboard Application
// Connects to your Python API and Storyblok to display real-time health monitoring data

// Configuration
const API_BASE_URL = 'http://localhost:8000/api';
const STORYBLOK_TOKEN = 'r9XuCd2PULaUy2DP10uZEAtt-89031663458228--Pex6e8oVG1xLopiUoTn'; // Add your Storyblok preview token
const STORYBLOK_SPACE_ID = '287039462257479'; // Your actual space ID

// Global variables
let alertsData = [];
let usersData = [];
let currentFilter = 'all';

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('SyncFit Dashboard Initializing...');
    loadDashboardData();
    initializeCharts();
    setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
});

// Load all dashboard data
async function loadDashboardData() {
    try {
        // For demo, using sample data. Replace with actual API calls
        await loadMetrics();
        await loadAlerts();
        await loadChurnPredictions();
        updateCharts();
        console.log('Dashboard data loaded successfully');
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

// Load key metrics
async function loadMetrics() {
    // Sample data - replace with actual API call
    // const response = await fetch(`${API_BASE_URL}/metrics`);
    // const data = await response.json();
    
    // Demo data
    const metrics = {
        totalUsers: 100,
        activeAlerts: 23,
        highRiskUsers: 11,
        interventionsToday: 7,
        emergencyAlerts: 3
    };
    
    // Update UI
    document.getElementById('totalUsers').textContent = metrics.totalUsers;
    document.getElementById('activeAlerts').textContent = metrics.activeAlerts;
    document.getElementById('highRisk').textContent = metrics.highRiskUsers;
    document.getElementById('interventions').textContent = metrics.interventionsToday;
    
    // Show emergency banner if needed
    if (metrics.emergencyAlerts > 0) {
        document.getElementById('emergencyBanner').classList.remove('hidden');
        document.getElementById('emergencyCount').textContent = 
            `${metrics.emergencyAlerts} users require immediate intervention`;
    }
}

// Load alerts data
async function loadAlerts() {
    // Sample alerts data - replace with actual API/Storyblok call
    alertsData = [
        {
            userId: 'user_0',
            userName: 'John Smith',
            severity: 'emergency',
            alertType: 'HIGH : Call 911',
            inactiveHours: 72.5,
            riskScore: 0.94,
            message: 'No activity for 72 hours, immediate intervention required'
        },
        {
            userId: 'user_1',
            userName: 'Mary Johnson',
            severity: 'high',
            alertType: 'MIDDLE : Notify Doctor',
            inactiveHours: 48.2,
            riskScore: 0.76,
            message: 'Significant activity decline detected'
        },
        {
            userId: 'user_2',
            userName: 'Robert Davis',
            severity: 'medium',
            alertType: 'LOW : Ping User',
            inactiveHours: 24.1,
            riskScore: 0.45,
            message: 'Moderate inactivity detected'
        },
        {
            userId: 'user_3',
            userName: 'Sarah Wilson',
            severity: 'emergency',
            alertType: 'HIGH : Call 911',
            inactiveHours: 96.3,
            riskScore: 0.98,
            message: 'Critical - No response for 4 days'
        },
        {
            userId: 'user_4',
            userName: 'Michael Brown',
            severity: 'high',
            alertType: 'MIDDLE : Notify Doctor',
            inactiveHours: 36.7,
            riskScore: 0.68,
            message: 'Activity 70% below normal'
        }
    ];
    
    displayAlerts(alertsData);
}

// Display alerts in table
function displayAlerts(alerts) {
    const tbody = document.getElementById('alertsTable');
    tbody.innerHTML = '';
    
    // Filter alerts if needed
    let filteredAlerts = alerts;
    if (currentFilter !== 'all') {
        filteredAlerts = alerts.filter(alert => alert.severity === currentFilter);
    }
    
    filteredAlerts.forEach(alert => {
        const row = document.createElement('tr');
        row.className = 'hover:bg-cyan-900/20 transition-colors';
        
        // Determine severity badge color for dark theme
        let severityClass = '';
        let severityIcon = '';
        switch(alert.severity) {
            case 'emergency':
                severityClass = 'bg-red-900/50 text-red-300 border border-red-500/30';
                severityIcon = 'fa-exclamation-triangle';
                break;
            case 'high':
                severityClass = 'bg-orange-900/50 text-orange-300 border border-orange-500/30';
                severityIcon = 'fa-exclamation-circle';
                break;
            case 'medium':
                severityClass = 'bg-yellow-900/50 text-yellow-300 border border-yellow-500/30';
                severityIcon = 'fa-exclamation';
                break;
            default:
                severityClass = 'bg-gray-900/50 text-gray-300 border border-gray-500/30';
                severityIcon = 'fa-info-circle';
        }
        
        row.innerHTML = `
            <td class="px-3 py-3 whitespace-nowrap">
                <div class="flex items-center">
                    <div class="flex-shrink-0 h-8 w-8 bg-cyan-900/30 rounded-full flex items-center justify-center border border-cyan-700/30">
                        <i class="fas fa-user text-cyan-400 text-xs"></i>
                    </div>
                    <div class="ml-2">
                        <div class="text-xs font-medium text-white">${alert.userName}</div>
                        <div class="text-xs text-cyan-500">${alert.userId}</div>
                    </div>
                </div>
            </td>
            <td class="px-3 py-3 whitespace-nowrap">
                <span class="px-2 inline-flex text-xs leading-4 font-semibold rounded-full ${severityClass}">
                    <i class="fas ${severityIcon} mr-1 text-xs"></i>
                    ${alert.severity.toUpperCase()}
                </span>
            </td>
            <td class="px-3 py-3 whitespace-nowrap text-xs text-cyan-100">
                ${alert.alertType}
            </td>
            <td class="px-3 py-3 whitespace-nowrap text-xs text-cyan-300">
                ${alert.inactiveHours.toFixed(1)}h
            </td>
            <td class="px-3 py-3 whitespace-nowrap">
                <div class="text-xs text-white font-semibold">${(alert.riskScore * 100).toFixed(0)}%</div>
                <div class="w-12 bg-cyan-900/30 rounded-full h-1.5 mt-1">
                    <div class="h-1.5 rounded-full" style="width: ${alert.riskScore * 100}%; background: linear-gradient(90deg, #00b3b0 0%, #00e5e2 100%);"></div>
                </div>
            </td>
            <td class="px-3 py-3 whitespace-nowrap text-xs">
                <button onclick="viewUserDetails('${alert.userId}')" class="text-cyan-400 hover:text-cyan-200 mr-2 transition-colors">
                    <i class="fas fa-eye"></i>
                </button>
                <button onclick="triggerIntervention('${alert.userId}')" class="text-green-400 hover:text-green-200 transition-colors">
                    <i class="fas fa-phone"></i>
                </button>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

// Filter alerts
function filterAlerts(filter) {
    currentFilter = filter;
    displayAlerts(alertsData);
}

// Initialize charts
function initializeCharts() {
    // Activity Trends Chart
    const activityCtx = document.getElementById('activityChart').getContext('2d');
    window.activityChart = new Chart(activityCtx, {
        type: 'line',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Average Steps',
                data: [8500, 9200, 7800, 8100, 9500, 10200, 8900],
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4
            }, {
                label: 'Active Users',
                data: [92, 94, 89, 91, 95, 97, 93],
                borderColor: 'rgb(34, 197, 94)',
                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                tension: 0.4,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    min: 0,
                    max: 15000,  // Fixed max for steps
                    ticks: {
                        stepSize: 3000
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    min: 80,
                    max: 100,  // Fixed range for active users percentage
                    grid: {
                        drawOnChartArea: false,
                    },
                    ticks: {
                        stepSize: 5
                    }
                }
            }
        }
    });
    
    // Risk Distribution Chart
    const riskCtx = document.getElementById('riskChart').getContext('2d');
    window.riskChart = new Chart(riskCtx, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical'],
            datasets: [{
                data: [65, 19, 11, 5],
                backgroundColor: [
                    'rgba(34, 197, 94, 0.8)',
                    'rgba(251, 191, 36, 0.8)',
                    'rgba(251, 146, 60, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                }
            }
        }
    });
}

// Update charts with new data
function updateCharts() {
    // Update with real data when available
    if (window.activityChart) {
        window.activityChart.update();
    }
    if (window.riskChart) {
        window.riskChart.update();
    }
}

// View user details
function viewUserDetails(userId) {
    const modal = document.getElementById('userModal');
    const modalContent = document.getElementById('modalContent');
    
    // Find user data
    const alert = alertsData.find(a => a.userId === userId);
    
    if (alert) {
        modalContent.innerHTML = `
            <div class="space-y-4">
                <div class="bg-gray-50 p-4 rounded-lg">
                    <h4 class="font-semibold text-gray-700 mb-2">User Information</h4>
                    <p><span class="font-medium">Name:</span> ${alert.userName}</p>
                    <p><span class="font-medium">ID:</span> ${alert.userId}</p>
                    <p><span class="font-medium">Last Active:</span> ${alert.inactiveHours.toFixed(1)} hours ago</p>
                </div>
                
                <div class="bg-red-50 p-4 rounded-lg">
                    <h4 class="font-semibold text-gray-700 mb-2">Alert Details</h4>
                    <p><span class="font-medium">Severity:</span> <span class="text-red-600">${alert.severity.toUpperCase()}</span></p>
                    <p><span class="font-medium">Type:</span> ${alert.alertType}</p>
                    <p><span class="font-medium">ML Risk Score:</span> ${(alert.riskScore * 100).toFixed(0)}%</p>
                    <p class="mt-2 text-sm text-gray-600">${alert.message}</p>
                </div>
                
                <div class="bg-blue-50 p-4 rounded-lg">
                    <h4 class="font-semibold text-gray-700 mb-2">Health Metrics (Last 7 Days)</h4>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                        <p><span class="font-medium">Avg Steps:</span> 3,245</p>
                        <p><span class="font-medium">Avg Heart Rate:</span> 72 bpm</p>
                        <p><span class="font-medium">Sleep Quality:</span> Poor</p>
                        <p><span class="font-medium">Activity Level:</span> 23%</p>
                    </div>
                </div>
                
                <div class="flex space-x-3">
                    <button onclick="triggerIntervention('${userId}')" class="flex-1 bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700">
                        <i class="fas fa-phone mr-2"></i>Contact User
                    </button>
                    <button onclick="notifyDoctor('${userId}')" class="flex-1 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700">
                        <i class="fas fa-user-md mr-2"></i>Notify Doctor
                    </button>
                    ${alert.severity === 'emergency' ? `
                    <button onclick="call911('${userId}')" class="flex-1 bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700">
                        <i class="fas fa-ambulance mr-2"></i>Call 911
                    </button>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    modal.classList.remove('hidden');
}

// Close modal
function closeModal() {
    document.getElementById('userModal').classList.add('hidden');
}

// Trigger intervention
async function triggerIntervention(userId) {
    console.log(`Triggering intervention for user: ${userId}`);
    // Call your API to trigger intervention
    alert(`Intervention triggered for user ${userId}. Contacting emergency contact...`);
}

// Notify doctor
async function notifyDoctor(userId) {
    console.log(`Notifying doctor for user: ${userId}`);
    alert(`Doctor notification sent for user ${userId}`);
}

// Call 911
async function call911(userId) {
    console.log(`Emergency call for user: ${userId}`);
    if (confirm(`This will trigger emergency services for user ${userId}. Proceed?`)) {
        alert(`911 Emergency services contacted for user ${userId}`);
    }
}

// Refresh data
function refreshData() {
    console.log('Refreshing dashboard data...');
    loadDashboardData();
}

// Show emergency modal
function showEmergencyModal() {
    // Filter for emergency alerts
    const emergencyAlerts = alertsData.filter(a => a.severity === 'emergency');
    currentFilter = 'emergency';
    displayAlerts(emergencyAlerts);
    
    // Scroll to alerts table
    document.querySelector('.bg-white.rounded-lg.shadow').scrollIntoView({ behavior: 'smooth' });
}

// Load churn predictions
async function loadChurnPredictions() {
    // This would connect to your ML model predictions
    console.log('Loading churn predictions from ML model...');
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        loadDashboardData,
        filterAlerts,
        viewUserDetails
    };
}
