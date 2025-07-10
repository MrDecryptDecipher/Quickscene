// Quickscene Web Interface JavaScript

class QuicksceneApp {
    constructor() {
        this.apiBase = '/api/v1';
        this.currentQuery = null;
        this.analytics = null;
        
        this.initializeElements();
        this.bindEvents();
        this.loadSystemStatus();
        this.loadAnalytics();
        
        // Auto-refresh analytics every 30 seconds
        setInterval(() => this.loadAnalytics(), 30000);
    }
    
    initializeElements() {
        // Search elements
        this.searchInput = document.getElementById('searchInput');
        this.searchBtn = document.getElementById('searchBtn');
        this.topKInput = document.getElementById('topK');
        this.thresholdInput = document.getElementById('threshold');
        this.thresholdValue = document.getElementById('thresholdValue');
        
        // Status elements
        this.videoCount = document.getElementById('videoCount');
        this.responseTime = document.getElementById('responseTime');
        this.searchType = document.getElementById('searchType');
        
        // Results elements
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.resultsContainer = document.getElementById('resultsContainer');
        this.resultsList = document.getElementById('resultsList');
        this.resultsCount = document.getElementById('resultsCount');
        this.queryTime = document.getElementById('queryTime');
        this.noResults = document.getElementById('noResults');
        this.errorMessage = document.getElementById('errorMessage');
        this.errorText = document.getElementById('errorText');
        
        // Analytics elements
        this.totalQueries = document.getElementById('totalQueries');
        this.avgResponseTime = document.getElementById('avgResponseTime');
        this.popularQuery = document.getElementById('popularQuery');
        this.successRate = document.getElementById('successRate');
        
        // Modal elements
        this.batchModal = document.getElementById('batchModal');
        this.helpModal = document.getElementById('helpModal');
        this.batchQueries = document.getElementById('batchQueries');
        this.batchTopK = document.getElementById('batchTopK');
        
        // Quick action buttons
        this.batchQueryBtn = document.getElementById('batchQueryBtn');
        this.analyticsBtn = document.getElementById('analyticsBtn');
        this.helpBtn = document.getElementById('helpBtn');
    }
    
    bindEvents() {
        // Search events
        this.searchBtn.addEventListener('click', () => this.performSearch());
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.performSearch();
        });
        
        // Threshold slider
        this.thresholdInput.addEventListener('input', (e) => {
            this.thresholdValue.textContent = e.target.value;
        });
        
        // Quick actions
        this.batchQueryBtn.addEventListener('click', () => this.showBatchModal());
        this.analyticsBtn.addEventListener('click', () => this.toggleAnalytics());
        this.helpBtn.addEventListener('click', () => this.showHelpModal());
        
        // Modal events
        document.getElementById('closeBatchModal').addEventListener('click', () => this.hideBatchModal());
        document.getElementById('closeHelpModal').addEventListener('click', () => this.hideHelpModal());
        document.getElementById('runBatchQuery').addEventListener('click', () => this.performBatchQuery());
        
        // Close modals on background click
        this.batchModal.addEventListener('click', (e) => {
            if (e.target === this.batchModal) this.hideBatchModal();
        });
        this.helpModal.addEventListener('click', (e) => {
            if (e.target === this.helpModal) this.hideHelpModal();
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideBatchModal();
                this.hideHelpModal();
            }
            if (e.ctrlKey && e.key === '/') {
                e.preventDefault();
                this.searchInput.focus();
            }
        });
    }
    
    async loadSystemStatus() {
        try {
            const response = await fetch(`${this.apiBase}/status`);
            const data = await response.json();
            
            if (response.ok) {
                this.videoCount.textContent = data.total_videos || '-';
                this.updateStatus('ready');
            } else {
                this.updateStatus('error');
                console.error('Status error:', data);
            }
        } catch (error) {
            this.updateStatus('error');
            console.error('Failed to load system status:', error);
        }
    }
    
    async loadAnalytics() {
        try {
            const response = await fetch(`${this.apiBase}/analytics?limit=100`);
            const data = await response.json();
            
            if (response.ok) {
                this.analytics = data;
                this.updateAnalyticsDisplay();
            }
        } catch (error) {
            console.error('Failed to load analytics:', error);
        }
    }
    
    updateAnalyticsDisplay() {
        if (!this.analytics) return;
        
        const stats = this.analytics.performance_stats;
        const popular = this.analytics.popular_queries;
        
        this.totalQueries.textContent = this.analytics.total_queries || '0';
        this.avgResponseTime.textContent = stats.avg_response_time_ms ? 
            `${stats.avg_response_time_ms}ms` : '-';
        this.popularQuery.textContent = popular && popular.length > 0 ? 
            popular[0].query : '-';
        this.successRate.textContent = stats.queries_under_1000ms ? 
            `${stats.queries_under_1000ms}%` : '-';
    }
    
    updateStatus(status) {
        // Update visual status indicators
        const statusItems = document.querySelectorAll('.status-item');
        statusItems.forEach(item => {
            item.classList.remove('status-ready', 'status-error');
            item.classList.add(`status-${status}`);
        });
    }
    
    async performSearch() {
        const query = this.searchInput.value.trim();
        if (!query) {
            this.showError('Please enter a search query');
            return;
        }
        
        this.currentQuery = query;
        this.showLoading();
        
        const startTime = performance.now();
        
        try {
            const response = await fetch(`${this.apiBase}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    top_k: parseInt(this.topKInput.value),
                    similarity_threshold: parseFloat(this.thresholdInput.value)
                })
            });
            
            const data = await response.json();
            const endTime = performance.now();
            const clientTime = Math.round(endTime - startTime);
            
            if (response.ok) {
                this.displayResults(data, clientTime);
                this.updateSearchStats(data, clientTime);
            } else {
                this.showError(data.error?.message || 'Search failed');
            }
            
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
        }
    }
    
    async performBatchQuery() {
        const queriesText = this.batchQueries.value.trim();
        if (!queriesText) {
            alert('Please enter at least one query');
            return;
        }
        
        const queries = queriesText.split('\n')
            .map(q => q.trim())
            .filter(q => q.length > 0);
        
        if (queries.length === 0) {
            alert('Please enter valid queries');
            return;
        }
        
        this.hideBatchModal();
        this.showLoading();
        
        try {
            const response = await fetch(`${this.apiBase}/batch-query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    queries: queries,
                    top_k: parseInt(this.batchTopK.value)
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.displayBatchResults(data);
            } else {
                this.showError(data.error?.message || 'Batch query failed');
            }
            
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
        }
    }
    
    displayResults(data, clientTime) {
        this.hideLoading();
        
        if (!data.results || data.results.length === 0) {
            this.showNoResults();
            return;
        }
        
        this.resultsCount.textContent = data.results.length;
        this.queryTime.textContent = data.query_time_ms || clientTime;
        
        this.resultsList.innerHTML = '';
        
        data.results.forEach(result => {
            const resultElement = this.createResultElement(result);
            this.resultsList.appendChild(resultElement);
        });
        
        this.showResults();
    }
    
    displayBatchResults(data) {
        this.hideLoading();
        
        const allResults = [];
        Object.values(data.results).forEach(queryResult => {
            if (queryResult.results) {
                allResults.push(...queryResult.results);
            }
        });
        
        if (allResults.length === 0) {
            this.showNoResults();
            return;
        }
        
        this.resultsCount.textContent = allResults.length;
        this.queryTime.textContent = data.total_time_ms || 0;
        
        this.resultsList.innerHTML = '';
        
        // Group results by query
        Object.entries(data.results).forEach(([query, queryResult]) => {
            if (queryResult.results && queryResult.results.length > 0) {
                // Add query header
                const queryHeader = document.createElement('div');
                queryHeader.className = 'query-header';
                queryHeader.innerHTML = `
                    <h3>Results for: "${query}"</h3>
                    <span class="query-stats">${queryResult.results.length} results in ${queryResult.query_time_ms}ms</span>
                `;
                this.resultsList.appendChild(queryHeader);
                
                // Add results
                queryResult.results.forEach(result => {
                    const resultElement = this.createResultElement(result);
                    this.resultsList.appendChild(resultElement);
                });
            }
        });
        
        this.showResults();
    }
    
    createResultElement(result) {
        const div = document.createElement('div');
        div.className = 'result-item';
        
        const confidence = Math.round(result.confidence * 100);
        const confidenceColor = confidence > 70 ? 'var(--success-color)' : 
                               confidence > 40 ? 'var(--warning-color)' : 
                               'var(--error-color)';
        
        div.innerHTML = `
            <div class="result-header">
                <div class="result-video">${result.video_id}</div>
                <div class="result-timestamp">${result.timestamp}</div>
            </div>
            <div class="result-dialogue">"${result.dialogue}"</div>
            <div class="result-meta">
                <div class="result-confidence">
                    <span>Confidence:</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%; background: ${confidenceColor}"></div>
                    </div>
                    <span>${confidence}%</span>
                </div>
                <div>Duration: ${result.start_time} - ${result.end_time}</div>
                <div>Type: ${result.search_type}</div>
            </div>
        `;
        
        return div;
    }
    
    updateSearchStats(data, clientTime) {
        this.responseTime.textContent = `${data.query_time_ms || clientTime}`;
        this.searchType.textContent = data.search_type || '-';
    }
    
    showLoading() {
        this.hideAllSections();
        this.loadingSpinner.classList.remove('hidden');
    }
    
    hideLoading() {
        this.loadingSpinner.classList.add('hidden');
    }
    
    showResults() {
        this.hideAllSections();
        this.resultsContainer.classList.remove('hidden');
    }
    
    showNoResults() {
        this.hideAllSections();
        this.noResults.classList.remove('hidden');
    }
    
    showError(message) {
        this.hideAllSections();
        this.errorText.textContent = message;
        this.errorMessage.classList.remove('hidden');
    }
    
    hideAllSections() {
        this.loadingSpinner.classList.add('hidden');
        this.resultsContainer.classList.add('hidden');
        this.noResults.classList.add('hidden');
        this.errorMessage.classList.add('hidden');
    }
    
    showBatchModal() {
        this.batchModal.classList.remove('hidden');
        this.batchQueries.focus();
    }
    
    hideBatchModal() {
        this.batchModal.classList.add('hidden');
    }
    
    showHelpModal() {
        this.helpModal.classList.remove('hidden');
    }
    
    hideHelpModal() {
        this.helpModal.classList.add('hidden');
    }
    
    toggleAnalytics() {
        const analyticsSection = document.querySelector('.analytics-section');
        if (analyticsSection.style.display === 'none') {
            analyticsSection.style.display = 'block';
            this.loadAnalytics();
        } else {
            analyticsSection.style.display = 'none';
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new QuicksceneApp();
});

// Add some CSS for batch query headers
const style = document.createElement('style');
style.textContent = `
    .query-header {
        padding: 20px 30px;
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border-bottom: 1px solid var(--border-color);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    
    .query-header h3 {
        margin: 0;
        color: var(--primary-color);
        font-size: 1.1rem;
    }
    
    .query-stats {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    .status-ready .status-item {
        border-left: 3px solid var(--success-color);
    }
    
    .status-error .status-item {
        border-left: 3px solid var(--error-color);
    }
`;
document.head.appendChild(style);
