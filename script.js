// Cybersecurity RAG Web Application - Frontend Logic

document.addEventListener('DOMContentLoaded', function() {
    // API Configuration
    const API_BASE_URL = 'http://localhost:5000/api';
    
    // DOM Elements
    const queryInput = document.getElementById('queryInput');
    const queryType = document.getElementById('queryType');
    const askButton = document.getElementById('askButton');
    const charCount = document.getElementById('charCount');
    const systemTime = document.getElementById('systemTime');
    const systemStatus = document.getElementById('systemStatus');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const analysisResults = document.getElementById('analysisResults');
    const resultsDisplay = document.getElementById('resultsDisplay');
    const sourcesList = document.getElementById('sourcesList');
    const sourceCount = document.getElementById('sourceCount');
    const totalDocs = document.getElementById('totalDocs');
    const totalChunks = document.getElementById('totalChunks');
    const lastUpdate = document.getElementById('lastUpdate');
    const refreshDocs = document.getElementById('refreshDocs');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const processingTime = document.getElementById('processingTime');
    const systemVersion = document.getElementById('systemVersion');
    const embeddingModel = document.getElementById('embeddingModel');
    const securityLevel = document.getElementById('securityLevel');
    const sourceModal = document.getElementById('sourceModal');
    const modalClose = document.querySelector('.modal-close');
    const modalBody = document.getElementById('modalBody');
    
    // Quick query buttons
    const quickButtons = document.querySelectorAll('.quick-buttons .btn-secondary');
    
    // State
    let currentResults = null;
    let systemStats = {
        documents: 0,
        chunks: 0,
        lastUpdate: 'Never'
    };
    
    // Initialize
    init();
    
    function init() {
        // Update system time
        updateSystemTime();
        setInterval(updateSystemTime, 1000);
        
        // Load system statistics
        loadSystemStats();
        
        // Set up event listeners
        setupEventListeners();
        
        // Check system health
        checkSystemHealth();
        
        // Update character count
        updateCharCount();
        
        // Set initial security level based on query type
        updateSecurityLevel();
        
        // Set system version
        systemVersion.textContent = 'v1.0.0 | Simple RAG Mode';
        embeddingModel.textContent = 'Embedding: Simple Mode';
    }
    
    function setupEventListeners() {
        // Query input events
        queryInput.addEventListener('input', updateCharCount);
        queryInput.addEventListener('keydown', handleQueryKeydown);
        
        // Query type change
        queryType.addEventListener('change', updateSecurityLevel);
        
        // Ask button
        askButton.addEventListener('click', processQuery);
        
        // Quick query buttons
        quickButtons.forEach(button => {
            button.addEventListener('click', function() {
                const query = this.getAttribute('data-query');
                queryInput.value = query;
                updateCharCount();
                processQuery();
            });
        });
        
        // Refresh documents
        refreshDocs.addEventListener('click', loadSystemStats);
        
        // Modal close
        modalClose.addEventListener('click', () => {
            sourceModal.style.display = 'none';
        });
        
        // Close modal when clicking outside
        window.addEventListener('click', (event) => {
            if (event.target === sourceModal) {
                sourceModal.style.display = 'none';
            }
        });
        
        // File upload button (hidden - for future use)
        const uploadButton = document.createElement('button');
        uploadButton.id = 'uploadButton';
        uploadButton.className = 'btn-secondary';
        uploadButton.innerHTML = '<i class="fas fa-upload"></i> UPLOAD DOCUMENTS';
        uploadButton.style.display = 'none';
        document.querySelector('.document-section').appendChild(uploadButton);
        
        uploadButton.addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.multiple = true;
            input.accept = '.pdf,.txt,.md,.json,.csv,.log,.docx';
            input.onchange = handleFileUpload;
            input.click();
        });
    }
    
    function handleFileUpload(event) {
        const files = event.target.files;
        if (!files.length) return;
        
        showNotification(`Uploading ${files.length} file(s)...`, 'info');
        
        Array.from(files).forEach(file => {
            const formData = new FormData();
            formData.append('file', file);
            
            fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification(`Uploaded ${file.name}`, 'success');
                    loadSystemStats(); // Refresh stats
                } else {
                    showNotification(`Failed to upload ${file.name}`, 'error');
                }
            })
            .catch(error => {
                showNotification(`Upload error: ${error.message}`, 'error');
            });
        });
    }
    
    function updateSystemTime() {
        const now = new Date();
        const utcTime = now.toUTCString().split(' ')[4];
        systemTime.textContent = `${utcTime} UTC`;
    }
    
    function updateCharCount() {
        const count = queryInput.value.length;
        charCount.textContent = count;
        
        // Update button state
        askButton.disabled = count === 0;
        askButton.style.opacity = count === 0 ? '0.5' : '1';
        askButton.style.cursor = count === 0 ? 'not-allowed' : 'pointer';
    }
    
    function updateSecurityLevel() {
        const selectedType = queryType.value;
        const levels = {
            'general': 'UNCLASSIFIED',
            'log_analysis': 'INTERNAL USE',
            'vulnerability': 'CONFIDENTIAL',
            'incident': 'RESTRICTED',
            'threat': 'SECRET',
            'policy': 'INTERNAL ONLY'
        };
        
        const level = levels[selectedType] || 'UNCLASSIFIED';
        securityLevel.textContent = level;
        
        // Update color based on level
        const colors = {
            'UNCLASSIFIED': '#00ff9d',
            'INTERNAL USE': '#00d4ff',
            'CONFIDENTIAL': '#ffd700',
            'RESTRICTED': '#ff6b35',
            'SECRET': '#ff375f',
            'INTERNAL ONLY': '#8a2be2'
        };
        
        securityLevel.style.color = colors[level] || '#00ff9d';
    }
    
    function handleQueryKeydown(event) {
        if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            processQuery();
        }
    }
    
    async function checkSystemHealth() {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            if (response.ok) {
                const data = await response.json();
                updateSystemStatus('online', data.status.toUpperCase());
            } else {
                updateSystemStatus('warning', 'DEGRADED');
            }
        } catch (error) {
            updateSystemStatus('critical', 'OFFLINE');
            console.error('Health check failed:', error);
            showNotification('Cannot connect to backend server', 'error');
        }
    }
    
    function updateSystemStatus(status, text) {
        const dot = systemStatus.querySelector('.status-dot');
        const textElement = systemStatus.querySelector('.status-text');
        
        // Remove existing status classes
        dot.classList.remove('status-online', 'status-warning', 'status-critical');
        systemStatus.classList.remove('status-online', 'status-warning', 'status-critical');
        
        // Add new status
        dot.classList.add(`status-${status}`);
        systemStatus.classList.add(`status-${status}`);
        textElement.textContent = text;
        
        // Update colors
        const colors = {
            'online': '#00ff9d',
            'warning': '#ffd700',
            'critical': '#ff375f'
        };
        
        const color = colors[status] || '#707090';
        dot.style.backgroundColor = color;
        systemStatus.style.borderColor = color;
        systemStatus.style.background = `rgba(${hexToRgb(color)}, 0.1)`;
    }
    
    function hexToRgb(hex) {
        // Remove # if present
        hex = hex.replace('#', '');
        
        // Parse hex
        const r = parseInt(hex.substring(0, 2), 16);
        const g = parseInt(hex.substring(2, 4), 16);
        const b = parseInt(hex.substring(4, 6), 16);
        
        return `${r}, ${g}, ${b}`;
    }
    
    async function loadSystemStats() {
        try {
            // For simple backend, we don't have stats endpoint
            // So we'll use health endpoint to check status
            const response = await fetch(`${API_BASE_URL}/health`);
            if (response.ok) {
                totalDocs.textContent = '0';
                totalChunks.textContent = '0';
                lastUpdate.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                
                // Show upload button since we have upload endpoint
                const uploadButton = document.getElementById('uploadButton');
                if (uploadButton) {
                    uploadButton.style.display = 'inline-flex';
                }
            }
        } catch (error) {
            console.error('Failed to load system stats:', error);
        }
    }
    
    async function processQuery() {
        const question = queryInput.value.trim();
        
        if (!question) {
            showNotification('Please enter a security query', 'warning');
            return;
        }
        
        // Show loading indicator
        showLoading(true);
        
        // Clear previous results
        clearResults();
        
        try {
            // Make API call
            const startTime = Date.now();
            const response = await fetch(`${API_BASE_URL}/ask`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question
                    // Simple backend doesn't use context parameter
                })
            });
            
            const queryTime = Date.now() - startTime;
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Display results
            displayResults(data, queryTime);
            
            // Show success notification
            showNotification('Analysis complete', 'success');
            
        } catch (error) {
            console.error('Query failed:', error);
            
            // Display error
            displayError(error.message);
            
            // Show error notification
            showNotification('Analysis failed', 'error');
            
        } finally {
            // Hide loading indicator
            showLoading(false);
        }
    }
    
    function showLoading(show) {
        if (show) {
            loadingIndicator.style.display = 'flex';
            resultsDisplay.style.display = 'none';
            askButton.disabled = true;
            askButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ANALYZING...';
            
            // Animate loading steps
            const steps = document.querySelectorAll('.loading-steps .step');
            let currentStep = 0;
            
            const stepInterval = setInterval(() => {
                steps.forEach(step => step.classList.remove('active'));
                
                if (currentStep < steps.length) {
                    steps[currentStep].classList.add('active');
                    currentStep++;
                } else {
                    clearInterval(stepInterval);
                }
            }, 800);
            
        } else {
            loadingIndicator.style.display = 'none';
            resultsDisplay.style.display = 'block';
            askButton.disabled = false;
            askButton.innerHTML = '<i class="fas fa-play"></i> EXECUTE ANALYSIS';
        }
    }
    
    function clearResults() {
        analysisResults.innerHTML = '';
        analysisResults.style.display = 'none';
        sourcesList.innerHTML = '';
        sourceCount.textContent = '0 sources';
        confidenceFill.style.width = '0%';
        confidenceValue.textContent = '0%';
        processingTime.textContent = '--.--s';
    }
    
    function displayResults(data, queryTime) {
        currentResults = data;
        
        // Show analysis results section
        analysisResults.style.display = 'block';
        
        // Get query context from dropdown
        const selectedType = queryType.value;
        const contextMap = {
            'general': 'GENERAL SECURITY',
            'log_analysis': 'LOG ANALYSIS',
            'vulnerability': 'VULNERABILITY ASSESSMENT',
            'incident': 'INCIDENT RESPONSE',
            'threat': 'THREAT INTELLIGENCE',
            'policy': 'POLICY & COMPLIANCE'
        };
        const securityContext = contextMap[selectedType] || 'GENERAL SECURITY';
        
        // Create result HTML
        const resultHtml = `
            <div class="analysis-result">
                <div class="result-header">
                    <div class="result-title">SECURITY ANALYSIS REPORT</div>
                    <div class="result-context">${securityContext}</div>
                </div>
                
                <div class="result-content">
                    ${formatAnswer(data.answer || 'No answer provided.')}
                </div>
                
                <div class="result-footer">
                    <div class="recommendations">
                        <div class="footer-title">
                            <i class="fas fa-bolt"></i>
                            NEXT STEPS
                        </div>
                        <div class="recommendation-item">
                            <i class="fas fa-chevron-right"></i>
                            Install full RAG dependencies for advanced analysis
                        </div>
                        <div class="recommendation-item">
                            <i class="fas fa-chevron-right"></i>
                            Upload security documents for contextual analysis
                        </div>
                        <div class="recommendation-item">
                            <i class="fas fa-chevron-right"></i>
                            Enable vector search for better relevance
                        </div>
                    </div>
                    
                    <div class="limitations">
                        <div class="footer-title">
                            <i class="fas fa-info-circle"></i>
                            SYSTEM STATUS
                        </div>
                        <div class="limitation-item">
                            <i class="fas fa-chevron-right"></i>
                            Running in simple demonstration mode
                        </div>
                        <div class="limitation-item">
                            <i class="fas fa-chevron-right"></i>
                            Document analysis capabilities limited
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        analysisResults.innerHTML = resultHtml;
        
        // Display sources (empty for simple backend)
        displaySources([]);
        
        // Update metrics
        updateMetrics(data, queryTime);
        
        // Scroll to results
        analysisResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    function formatAnswer(answer) {
        if (!answer) return '<p>No answer provided.</p>';
        
        // Convert markdown-like formatting to HTML
        let formatted = answer
            // Headers
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/__(.*?)__/g, '<strong>$1</strong>')
            
            // Italic
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/_(.*?)_/g, '<em>$1</em>')
            
            // Code
            .replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Handle lists
        const lines = formatted.split('\n');
        let inList = false;
        let listItems = [];
        
        formatted = lines.map(line => {
            if (line.trim().match(/^[•\-*]\s+/)) {
                if (!inList) {
                    inList = true;
                    return '<ul><li>' + line.replace(/^[•\-*]\s+/, '') + '</li>';
                }
                return '<li>' + line.replace(/^[•\-*]\s+/, '') + '</li>';
            } else if (line.trim().match(/^\d+\.\s+/)) {
                if (!inList) {
                    inList = true;
                    return '<ol><li>' + line.replace(/^\d+\.\s+/, '') + '</li>';
                }
                return '<li>' + line.replace(/^\d+\.\s+/, '') + '</li>';
            } else {
                if (inList) {
                    inList = false;
                    return '</ul>' + line;
                }
                return line;
            }
        }).join('\n');
        
        // Close any open list
        if (inList) {
            formatted += '</ul>';
        }
        
        // Wrap in paragraphs
        const paragraphs = formatted.split(/\n\n+/);
        formatted = paragraphs.map(p => {
            p = p.trim();
            if (!p) return '';
            if (p.startsWith('<')) return p; // Already HTML
            return `<p>${p.replace(/\n/g, '<br>')}</p>`;
        }).join('');
        
        return formatted;
    }
    
    function displaySources(sources) {
        if (!sources || sources.length === 0) {
            sourcesList.innerHTML = `
                <div class="source-empty">
                    <i class="fas fa-info-circle"></i>
                    <p>Simple mode - no document sources available</p>
                    <p style="font-size: 0.8em; margin-top: 10px;">
                        Upload documents to enable source tracking
                    </p>
                </div>
            `;
            sourceCount.textContent = '0 sources';
            return;
        }
        
        sourceCount.textContent = `${sources.length} source${sources.length !== 1 ? 's' : ''}`;
        
        const sourcesHtml = sources.map((source, index) => {
            const safeSource = source || {};
            const docName = safeSource.document || 'Unknown Document';
            const docType = safeSource.type || 'unknown';
            const securityLevel = safeSource.security_level || 'UNCLASSIFIED';
            const relevanceScore = safeSource.relevance_score || 0;
            const context = safeSource.context || 'general';
            
            return `
            <div class="source-item" data-index="${index}">
                <div class="source-header">
                    <div class="source-title">${docName}</div>
                    <div class="source-score">${(relevanceScore * 100).toFixed(0)}%</div>
                </div>
                <div class="source-meta">
                    <div class="source-type">
                        <i class="fas fa-file-alt"></i>
                        ${docType.replace('_', ' ').toUpperCase()}
                    </div>
                    <div class="source-security">
                        <i class="fas fa-shield-alt"></i>
                        ${securityLevel}
                    </div>
                </div>
                <div class="source-preview">
                    Context: ${context.replace('_', ' ')}
                </div>
            </div>
            `;
        }).join('');
        
        sourcesList.innerHTML = sourcesHtml;
    }
    
    function updateMetrics(data, queryTime) {
        // Update confidence
        const confidencePercent = Math.round((data.confidence || 0.8) * 100);
        confidenceFill.style.width = `${confidencePercent}%`;
        confidenceValue.textContent = `${confidencePercent}%`;
        
        // Update processing time
        const displayTime = queryTime < 1000 ? `${queryTime}ms` : `${(queryTime / 1000).toFixed(2)}s`;
        processingTime.textContent = displayTime;
    }
    
    function displayError(errorMessage) {
        analysisResults.style.display = 'block';
        analysisResults.innerHTML = `
            <div class="analysis-result">
                <div class="result-header">
                    <div class="result-title">ANALYSIS ERROR</div>
                    <div class="result-context" style="color: #ff375f;">SYSTEM ERROR</div>
                </div>
                
                <div class="result-content">
                    <h3>Security Analysis Failed</h3>
                    <p>The system encountered an error while processing your security query.</p>
                    
                    <div class="highlight" style="border-left-color: #ff375f;">
                        <strong>Error Details:</strong><br>
                        ${errorMessage || 'Unknown error occurred'}
                    </div>
                    
                    <h4>Troubleshooting Steps:</h4>
                    <ul>
                        <li>Check if backend server is running: <code>http://localhost:5000</code></li>
                        <li>Verify the server responds at: <code>${API_BASE_URL}/health</code></li>
                        <li>Check browser console (F12) for detailed errors</li>
                        <li>Make sure CORS is enabled on the backend</li>
                        <li>Try restarting the backend server</li>
                    </ul>
                    
                    <h4>Quick Diagnostic:</h4>
                    <button id="testConnection" class="btn-secondary" style="margin: 10px 0;">
                        <i class="fas fa-plug"></i> Test Backend Connection
                    </button>
                    <div id="testResult" style="margin-top: 10px; display: none;"></div>
                    
                    <h4>Backend Status:</h4>
                    <pre id="backendStatus" style="background: #1a1f35; padding: 10px; border-radius: 4px; overflow: auto;">
Testing connection...
                    </pre>
                </div>
            </div>
        `;
        
        // Test backend connection automatically
        testBackendConnection();
        
        // Add manual test button handler
        setTimeout(() => {
            const testBtn = document.getElementById('testConnection');
            if (testBtn) {
                testBtn.addEventListener('click', testBackendConnection);
            }
        }, 100);
        
        // Update metrics for error state
        confidenceFill.style.width = '0%';
        confidenceValue.textContent = '0%';
        processingTime.textContent = '--.--s';
    }
    
    async function testBackendConnection() {
        const statusElement = document.getElementById('backendStatus');
        const testBtn = document.getElementById('testConnection');
        
        if (testBtn) {
            testBtn.disabled = true;
            testBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';
        }
        
        if (statusElement) {
            statusElement.textContent = 'Testing connection...';
        }
        
        try {
            // Test health endpoint
            const healthResponse = await fetch(`${API_BASE_URL}/health`);
            const healthData = await healthResponse.json();
            
            // Test ask endpoint
            const askResponse = await fetch(`${API_BASE_URL}/ask`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: 'test'})
            });
            const askData = await askResponse.json();
            
            if (statusElement) {
                statusElement.innerHTML = `
✅ <strong>Backend is fully operational!</strong>

<strong>Health Endpoint (${API_BASE_URL}/health):</strong>
• Status: ${healthData.status}
• Version: ${healthData.version}

<strong>Ask Endpoint (${API_BASE_URL}/ask):</strong>
• Response: ${askData.answer ? '✓ Received answer' : '✗ No answer'}
• Confidence: ${askData.confidence || 'N/A'}

<strong>Recommendation:</strong>
The backend is working correctly. The issue might be with:
1. The specific query you sent
2. Browser CORS settings
3. Network connectivity between frontend and backend
                `.trim();
            }
            
            showNotification('Backend connection successful!', 'success');
            
        } catch (error) {
            if (statusElement) {
                statusElement.innerHTML = `
❌ <strong>Connection Failed!</strong>

<strong>Error:</strong> ${error.message}

<strong>Troubleshooting:</strong>
1. Make sure the backend server is running
2. Check if port 5000 is not blocked
3. Verify the backend is accessible at: ${API_BASE_URL}
4. Check for CORS errors in browser console (F12)

<strong>To start backend:</strong>
cd backend
python simple_app.py
                `.trim();
            }
            
            showNotification('Backend connection failed', 'error');
        }
        
        if (testBtn) {
            testBtn.disabled = false;
            testBtn.innerHTML = '<i class="fas fa-plug"></i> Test Backend Connection';
        }
    }
    
    function showNotification(message, type) {
        // Remove existing notification
        const existingNotification = document.querySelector('.notification');
        if (existingNotification) {
            existingNotification.remove();
        }
        
        // Create notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        
        const icons = {
            'success': 'fa-check-circle',
            'error': 'fa-exclamation-circle',
            'warning': 'fa-exclamation-triangle',
            'info': 'fa-info-circle'
        };
        
        notification.innerHTML = `
            <i class="fas ${icons[type] || 'fa-info-circle'}"></i>
            <span>${message}</span>
        `;
        
        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            background: ${getNotificationBg(type)};
            color: white;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-family: var(--font-mono);
            font-size: 0.9rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        // Add keyframe animations
        if (!document.querySelector('#notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes slideOut {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOut 0.3s ease-out';
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);
    }
    
    function getNotificationBg(type) {
        const colors = {
            'success': '#00a86b',
            'error': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8'
        };
        return colors[type] || '#17a2b8';
    }
    
    // Export functionality for debugging
    window.exportResults = function() {
        if (!currentResults) {
            showNotification('No results to export', 'warning');
            return;
        }
        
        const exportData = {
            timestamp: new Date().toISOString(),
            query: queryInput.value,
            context: queryType.value,
            results: currentResults
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        
        const exportFileDefaultName = `security-analysis-${Date.now()}.json`;
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
        
        showNotification('Results exported successfully', 'success');
    };
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        // Ctrl/Cmd + E to export
        if ((event.ctrlKey || event.metaKey) && event.key === 'e') {
            event.preventDefault();
            window.exportResults();
        }
        
        // Ctrl/Cmd + / to focus query input
        if ((event.ctrlKey || event.metaKey) && event.key === '/') {
            event.preventDefault();
            queryInput.focus();
        }
        
        // Escape to clear query
        if (event.key === 'Escape' && document.activeElement === queryInput) {
            queryInput.value = '';
            updateCharCount();
        }
    });
    
    // Test backend connection on startup
    setTimeout(checkSystemHealth, 1000);
});