/**
 * ZTE Router Dashboard - Client-side JavaScript
 * Handles WebSocket connection, UI updates, and API calls
 */

// Session ID for API calls
const sessionId = 'session_' + Math.random().toString(36).substr(2, 9);

// WebSocket connection
let socket = null;
let isConnected = false;

// Current signal data for AI analysis
let currentSignalData = null;

// Parsed AI recommendations for quick apply
let parsedAiRecommendation = null;

// Selected Ollama model (from login screen)
let selectedOllamaModel = 'llama3';
let autoOptimizeInterval = null;
const AUTO_OPTIMIZE_COOLDOWN = 3 * 60 * 1000; // 3 minutes

// ============= Ollama Model Loading =============

async function loadOllamaModels() {
    const loginSelect = document.getElementById('login-ollama-model');
    const dashboardSelect = document.getElementById('ollama-model');
    const statusEl = document.getElementById('ollama-login-status');
    const dashboardStatusEl = document.getElementById('ollama-status');

    try {
        const response = await fetch('/api/ollama/models');
        const result = await response.json();

        if (result.success && result.models.length > 0) {
            const models = result.models;

            // Preference order for "best" models
            const preference = ['llama3', 'qwen2', 'mistral', 'llama3.1', 'llama3.2', 'gemma2', 'phi3'];
            let bestModel = models[0];
            for (const pref of preference) {
                const found = models.find(m => m.toLowerCase().includes(pref));
                if (found) {
                    bestModel = found;
                    break;
                }
            }

            // Populate login dropdown
            loginSelect.innerHTML = models.map(m =>
                `<option value="${m}" ${m === bestModel ? 'selected' : ''}>${m}</option>`
            ).join('');

            // Populate dashboard dropdown  
            if (dashboardSelect) {
                dashboardSelect.innerHTML = models.map(m =>
                    `<option value="${m}" ${m === bestModel ? 'selected' : ''}>${m}</option>`
                ).join('');
            }

            selectedOllamaModel = bestModel;

            const msg = `✓ ${models.length} models available (Auto-selected: ${bestModel})`;
            statusEl.textContent = msg;
            statusEl.className = 'text-success';
            if (dashboardStatusEl) {
                dashboardStatusEl.textContent = `Online: ${bestModel}`;
                dashboardStatusEl.className = 'ollama-status-badge status-online';
            }
        } else {
            loginSelect.innerHTML = '<option value="">No models found</option>';
            statusEl.textContent = result.error || '✗ Ollama not available';
            statusEl.className = 'text-error';
            if (dashboardStatusEl) {
                dashboardStatusEl.textContent = 'Offline';
                dashboardStatusEl.className = 'ollama-status-badge status-offline';
            }
        }
    } catch (e) {
        loginSelect.innerHTML = '<option value="">Ollama offline</option>';
        statusEl.textContent = '✗ Cannot connect to Ollama';
        statusEl.className = 'text-error';
    }
}

// ============= UI Helpers =============

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function showError(elementId, message) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    el.classList.remove('hidden');
}

function hideError(elementId) {
    document.getElementById(elementId).classList.add('hidden');
}

function setLoading(button, loading) {
    if (loading) {
        button.disabled = true;
        button.dataset.originalText = button.textContent;
        button.textContent = 'Loading...';
    } else {
        button.disabled = false;
        button.textContent = button.dataset.originalText || button.textContent;
    }
}

// AI Activity Log
function addAiLog(message, type = 'thought') {
    const logEl = document.getElementById('ai-log');
    if (!logEl) return;

    const entry = document.createElement('div');
    entry.className = `log-entry ai-${type}`;

    // Add timestamp
    const now = new Date();
    const time = now.getHours().toString().padStart(2, '0') + ':' +
        now.getMinutes().toString().padStart(2, '0') + ':' +
        now.getSeconds().toString().padStart(2, '0');

    entry.innerHTML = `<span style="color: #555; font-size: 0.7rem; margin-right: 8px;">[${time}]</span> ${message}`;

    logEl.appendChild(entry);
    logEl.scrollTop = logEl.scrollHeight;
}

// ============= Signal Quality Helpers =============

function getSignalClass(rsrp) {
    const val = parseFloat(rsrp);
    if (isNaN(val)) return '';
    if (val >= -80) return 'signal-excellent';
    if (val >= -100) return 'signal-good';
    if (val >= -110) return 'signal-fair';
    return 'signal-poor';
}

function getSinrClass(sinr) {
    const val = parseFloat(sinr);
    if (isNaN(val)) return '';
    if (val >= 20) return 'signal-excellent';
    if (val >= 10) return 'signal-good';
    if (val >= 0) return 'signal-fair';
    return 'signal-poor';
}

// ============= Cell Card Rendering =============

function renderLteCell(cell, index) {
    const isMainCell = index === 0;
    return `
        <div class="cell-card">
            <div class="cell-header">
                <span class="cell-band">${cell.band}</span>
                <span class="cell-pci">PCI: ${cell.pci}</span>
            </div>
            <div class="cell-metrics">
                ${cell.rsrp1 ? `
                <div class="metric">
                    <span class="metric-label">RSRP1</span>
                    <span class="metric-value ${getSignalClass(cell.rsrp1)}">${cell.rsrp1} dBm</span>
                </div>` : ''}
                ${cell.sinr1 ? `
                <div class="metric">
                    <span class="metric-label">SINR1</span>
                    <span class="metric-value ${getSinrClass(cell.sinr1)}">${cell.sinr1} dB</span>
                </div>` : ''}
                ${isMainCell && cell.rsrp2 ? `
                <div class="metric">
                    <span class="metric-label">RSRP2</span>
                    <span class="metric-value ${getSignalClass(cell.rsrp2)}">${cell.rsrp2} dBm</span>
                </div>` : ''}
                ${isMainCell && cell.sinr2 ? `
                <div class="metric">
                    <span class="metric-label">SINR2</span>
                    <span class="metric-value ${getSinrClass(cell.sinr2)}">${cell.sinr2} dB</span>
                </div>` : ''}
                ${cell.rsrq ? `
                <div class="metric">
                    <span class="metric-label">RSRQ</span>
                    <span class="metric-value">${cell.rsrq} dB</span>
                </div>` : ''}
                ${cell.earfcn ? `
                <div class="metric">
                    <span class="metric-label">EARFCN</span>
                    <span class="metric-value">${cell.earfcn}</span>
                </div>` : ''}
                ${cell.bandwidth ? `
                <div class="metric">
                    <span class="metric-label">BW</span>
                    <span class="metric-value">${cell.bandwidth} MHz</span>
                </div>` : ''}
            </div>
        </div>
    `;
}

function renderNrCell(cell, index) {
    return `
        <div class="cell-card">
            <div class="cell-header">
                <span class="cell-band">${cell.band}</span>
                <span class="cell-pci">PCI: ${cell.pci}</span>
            </div>
            <div class="cell-metrics">
                ${cell.rsrp1 ? `
                <div class="metric">
                    <span class="metric-label">RSRP1</span>
                    <span class="metric-value ${getSignalClass(cell.rsrp1)}">${cell.rsrp1} dBm</span>
                </div>` : ''}
                ${cell.rsrp2 ? `
                <div class="metric">
                    <span class="metric-label">RSRP2</span>
                    <span class="metric-value ${getSignalClass(cell.rsrp2)}">${cell.rsrp2} dBm</span>
                </div>` : ''}
                ${cell.sinr && cell.sinr !== '?????' ? `
                <div class="metric">
                    <span class="metric-label">SINR</span>
                    <span class="metric-value ${getSinrClass(cell.sinr)}">${cell.sinr} dB</span>
                </div>` : ''}
                ${cell.rsrq ? `
                <div class="metric">
                    <span class="metric-label">RSRQ</span>
                    <span class="metric-value">${cell.rsrq} dB</span>
                </div>` : ''}
                ${cell.arfcn ? `
                <div class="metric">
                    <span class="metric-label">ARFCN</span>
                    <span class="metric-value">${cell.arfcn}</span>
                </div>` : ''}
                ${cell.bandwidth ? `
                <div class="metric">
                    <span class="metric-label">BW</span>
                    <span class="metric-value">${cell.bandwidth} MHz</span>
                </div>` : ''}
            </div>
        </div>
    `;
}

// ============= UI Update =============

function updateDashboard(data) {
    try {
        console.log("[DEBUG] updateDashboard with data:", data ? "YES" : "NULL");
        currentSignalData = data;

        // Connection info
        const setEl = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val || '-';
        };

        setEl('provider', data.provider);
        setEl('network-type', data.network_type);
        setEl('band-info', data.band_info);
        setEl('wan-ip', data.wan_ip);

        // Temperatures
        const temps = data.temperatures || {};
        const tempRow = document.getElementById('temp-row');
        const tempEl = document.getElementById('temperatures');

        if (Object.keys(temps).length > 0 && tempEl) {
            const tempStr = Object.entries(temps)
                .map(([k, v]) => `${k.charAt(0).toUpperCase()}: ${v}°C`)
                .join('  ');
            tempEl.textContent = tempStr;
            if (tempRow) tempRow.classList.remove('hidden');
        } else if (tempRow) {
            tempRow.classList.add('hidden');
        }

        // LTE cells
        const lteCellsContainer = document.getElementById('lte-cells');
        const lteSection = document.getElementById('lte-cells-container');

        if (data.lte_cells && data.lte_cells.length > 0) {
            if (lteCellsContainer) {
                lteCellsContainer.innerHTML = data.lte_cells
                    .map((cell, i) => renderLteCell(cell, i))
                    .join('');
            }
            if (lteSection) lteSection.classList.remove('hidden');

            // Auto-fill cell lock inputs
            const primaryCell = data.lte_cells[0];
            const pciInp = document.getElementById('lte-lock-pci');
            const earfcnInp = document.getElementById('lte-lock-earfcn');
            if (pciInp && !pciInp.value) pciInp.placeholder = primaryCell.pci;
            if (earfcnInp && !earfcnInp.value) earfcnInp.placeholder = primaryCell.earfcn;
        } else {
            if (lteCellsContainer) lteCellsContainer.innerHTML = '<p style="color: var(--text-muted)">No LTE cells</p>';
        }

        // NR cells
        const nrCellsContainer = document.getElementById('nr-cells');
        const nrSection = document.getElementById('nr-cells-container');

        if (data.nr_cells && data.nr_cells.length > 0) {
            if (nrCellsContainer) {
                nrCellsContainer.innerHTML = data.nr_cells
                    .map((cell, i) => renderNrCell(cell, i))
                    .join('');
            }
            if (nrSection) nrSection.classList.remove('hidden');

            // Auto-fill NR cell lock inputs
            const primaryNr = data.nr_cells[0];
            const nrPciInp = document.getElementById('nr-lock-pci');
            const nrArfcnInp = document.getElementById('nr-lock-arfcn');
            const nrBandInp = document.getElementById('nr-lock-band');

            if (nrPciInp && !nrPciInp.value) nrPciInp.placeholder = primaryNr.pci;
            if (nrArfcnInp && !nrArfcnInp.value) nrArfcnInp.placeholder = primaryNr.arfcn;
            if (nrBandInp && !nrBandInp.value) {
                const band = primaryNr.band ? primaryNr.band.replace('n', '') : '';
                nrBandInp.placeholder = band;
            }
        } else {
            if (nrCellsContainer) nrCellsContainer.innerHTML = '<p style="color: var(--text-muted)">No 5G cells</p>';
        }
    } catch (e) {
        console.error('Update Dashboard Error:', e);
    }
}

// ============= API Calls =============

async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
            'X-Session-ID': sessionId
        }
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    const response = await fetch(`/api/${endpoint}`, options);
    return response.json();
}

// ============= Connection Handlers =============

async function connect() {
    const ipAddress = document.getElementById('ip-address').value;
    const password = document.getElementById('password').value;
    const btn = document.getElementById('btn-connect');

    if (!password) {
        showError('login-error', 'Please enter the router password');
        return;
    }

    setLoading(btn, true);
    hideError('login-error');

    try {
        const result = await apiCall('connect', 'POST', { ip_address: ipAddress, password });

        if (result.success) {
            isConnected = true;

            // Set the selected Ollama model from login
            selectedOllamaModel = document.getElementById('login-ollama-model').value || 'llama3';
            const dashboardSelect = document.getElementById('ollama-model');
            if (dashboardSelect && selectedOllamaModel) {
                dashboardSelect.value = selectedOllamaModel;
                // Also notify backend of the selected model
                await apiCall('ollama/model', 'POST', { model: selectedOllamaModel });
            }

            document.getElementById('login-section').classList.add('hidden');
            document.getElementById('dashboard').classList.remove('hidden');
            updateDashboard(result.signal_info);
            startPolling(ipAddress, password);
            checkOllama();
            showToast('Connected to router', 'success');
        } else {
            showError('login-error', result.error || 'Connection failed');
        }
    } catch (e) {
        showError('login-error', 'Network error: ' + e.message);
    }

    setLoading(btn, false);
}

async function disconnect() {
    stopPolling();
    await apiCall('disconnect', 'POST');
    isConnected = false;
    document.getElementById('dashboard').classList.add('hidden');
    document.getElementById('login-section').classList.remove('hidden');
    showToast('Disconnected', 'info');
}

// ============= Polling =============

let pollingInterval = null;

function startPolling(ipAddress, password) {
    // Use HTTP polling instead of WebSocket for simplicity
    pollingInterval = setInterval(async () => {
        if (!isConnected) return;

        try {
            const result = await apiCall('signal');
            if (result.success) {
                updateDashboard(result.signal_info);
            }
        } catch (e) {
            console.error('Polling error:', e);
        }
    }, 1000);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

// ============= Control Actions =============

async function setNetworkMode(mode) {
    try {
        const result = await apiCall('network_mode', 'POST', { mode });
        if (result.success) {
            showToast(`Network mode set to ${mode}`, 'success');
        } else {
            showToast(result.error || 'Failed to set network mode', 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

async function setLteBands(bands) {
    try {
        const result = await apiCall('lte_bands', 'POST', { bands });
        if (result.success) {
            showToast(`LTE bands set to ${bands}`, 'success');
        } else {
            showToast(result.error || 'Failed to set LTE bands', 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

async function setNrBands(bands) {
    try {
        const result = await apiCall('nr_bands', 'POST', { bands });
        if (result.success) {
            showToast(`5G bands set to ${bands}`, 'success');
        } else {
            showToast(result.error || 'Failed to set 5G bands', 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

async function lockLteCell(pci, earfcn) {
    try {
        const result = await apiCall('lte_cell_lock', 'POST', { pci, earfcn });
        if (result.success) {
            showToast(pci === 0 ? 'LTE cell unlocked' : `LTE cell locked to PCI ${pci}`, 'success');
        } else {
            showToast(result.error || 'Failed to lock LTE cell', 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

async function lockNrCell(pci, arfcn, band, scs) {
    try {
        const result = await apiCall('nr_cell_lock', 'POST', { pci, arfcn, band, scs });
        if (result.success) {
            showToast(pci === 0 ? '5G cell unlocked' : `5G cell locked to PCI ${pci}`, 'success');
        } else {
            showToast(result.error || 'Failed to lock 5G cell', 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

async function setBridgeMode(enable) {
    if (!confirm(`${enable ? 'Enable' : 'Disable'} bridge mode? Router will reboot.`)) return;

    try {
        const result = await apiCall('bridge_mode', 'POST', { enable });
        if (result.success) {
            showToast(`Bridge mode ${enable ? 'enabled' : 'disabled'}. Rebooting...`, 'success');
        } else {
            showToast(result.error || 'Failed', 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

async function setArpProxy(enable) {
    if (!confirm(`${enable ? 'Enable' : 'Disable'} ARP proxy? Router will reboot.`)) return;

    try {
        const result = await apiCall('arp_proxy', 'POST', { enable });
        if (result.success) {
            showToast(`ARP proxy ${enable ? 'enabled' : 'disabled'}. Rebooting...`, 'success');
        } else {
            showToast(result.error || 'Failed', 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

async function rebootRouter() {
    if (!confirm('Reboot the router?')) return;

    try {
        const result = await apiCall('reboot', 'POST');
        showToast('Router rebooting...', 'success');
        disconnect();
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

// ============= Ollama AI =============

async function checkOllama() {
    const statusEl = document.getElementById('ollama-status');

    try {
        const result = await apiCall('ollama/check');

        if (result.available) {
            statusEl.textContent = `✓ Ollama connected - Model: ${result.model}`;
            statusEl.className = 'ollama-status connected';
        } else {
            statusEl.textContent = `✗ ${result.error || 'Ollama not available'}`;
            if (result.hint) statusEl.textContent += ` (${result.hint})`;
            statusEl.className = 'ollama-status disconnected';
        }
    } catch (e) {
        statusEl.textContent = '✗ Could not check Ollama status';
        statusEl.className = 'ollama-status disconnected';
    }
}

async function getAiRecommendation(isAuto = false) {
    console.log("[DEBUG] getAiRecommendation called, isAuto:", isAuto);
    if (!currentSignalData) {
        console.log("[DEBUG] currentSignalData is NULL, exiting.");
        if (!isAuto) showToast('No signal data available', 'error');
        return;
    }

    const btn = document.getElementById('btn-ai-recommend');
    const loadingEl = document.getElementById('ai-loading');
    const recommendationEl = document.getElementById('ai-recommendation');
    const contentEl = document.getElementById('ai-content');
    const actionsEl = document.getElementById('ai-actions');
    const logEl = document.getElementById('ai-log');

    // Reset UI
    if (!isAuto && logEl) logEl.innerHTML = '';
    addAiLog(`${isAuto ? '[AUTO]' : '[MANUAL]'} Starting download speed optimization...`, 'action');

    // Update model if changed
    const modelDropdown = document.getElementById('ollama-model');
    if (modelDropdown) {
        const model = modelDropdown.value;
        if (model) {
            await apiCall('ollama/model', 'POST', { model });
        }
    }

    setLoading(btn, true);
    if (loadingEl) loadingEl.classList.remove('hidden');
    recommendationEl.classList.add('hidden');

    try {
        addAiLog('Analyzing multi-band carrier aggregation for max throughput...', 'thought');
        console.log("[DEBUG] Calling /api/ollama/recommend...");
        const result = await apiCall('ollama/recommend', 'POST');
        console.log("[DEBUG] AI Recommendation result:", result);

        if (result.success) {
            addAiLog('Analysis received. Goal: PRIMARY DOWNLOAD SPEED.', 'action');
            if (contentEl) {
                // Use marked for well-formatted output
                if (typeof marked !== 'undefined') {
                    contentEl.innerHTML = marked.parse(result.recommendation);
                } else {
                    contentEl.textContent = result.recommendation;
                }
            }
            recommendationEl.classList.remove('hidden');

            parsedAiRecommendation = result.parsed;
            if (parsedAiRecommendation) {
                addAiLog('Optimal configuration summary:', 'action');
                if (parsedAiRecommendation.lte_bands) addAiLog(`→ LTE: ${parsedAiRecommendation.lte_bands}`, 'result');
                if (parsedAiRecommendation.nr_bands) addAiLog(`→ 5G: ${parsedAiRecommendation.nr_bands}`, 'result');
                if (parsedAiRecommendation.network_mode) addAiLog(`→ Mode: ${parsedAiRecommendation.network_mode}`, 'result');

                actionsEl.classList.remove('hidden');

                const lteBtnEl = document.getElementById('btn-apply-lte');
                const nrBtnEl = document.getElementById('btn-apply-nr');
                const modeBtnEl = document.getElementById('btn-apply-mode');

                if (lteBtnEl) {
                    if (parsedAiRecommendation.lte_bands) {
                        lteBtnEl.textContent = `Apply LTE: ${parsedAiRecommendation.lte_bands}`;
                        lteBtnEl.classList.remove('hidden');
                    } else lteBtnEl.classList.add('hidden');
                }
                if (nrBtnEl) {
                    if (parsedAiRecommendation.nr_bands) {
                        nrBtnEl.textContent = `Apply 5G: ${parsedAiRecommendation.nr_bands}`;
                        nrBtnEl.classList.remove('hidden');
                    } else nrBtnEl.classList.add('hidden');
                }
                if (modeBtnEl) {
                    if (parsedAiRecommendation.network_mode) {
                        modeBtnEl.textContent = `Apply Mode: ${parsedAiRecommendation.network_mode}`;
                        modeBtnEl.classList.remove('hidden');
                    } else modeBtnEl.classList.add('hidden');
                }

                // AUTOMATICALLY APPLY IF IN FULL AUTO MODE
                const autoToggle = document.getElementById('auto-optimize-toggle');
                if (isAuto && autoToggle && autoToggle.checked) {
                    addAiLog('AUTO-MODE: Applying best bands for download speed...', 'action');
                    if (parsedAiRecommendation.network_mode) await applyAiRecommendation('mode');
                    if (parsedAiRecommendation.lte_bands) await applyAiRecommendation('lte');
                    if (parsedAiRecommendation.nr_bands) await applyAiRecommendation('nr');
                    addAiLog('Full optimization cycle finished.', 'result');
                }
            } else {
                addAiLog('AI provided general advice but no specific band unlocks.', 'thought');
            }

            if (!isAuto) showToast('AI recommendation received', 'success');
        } else {
            addAiLog(`Error: ${result.error}`, 'thought');
            if (!isAuto) showToast(result.error || 'Failed to get recommendation', 'error');
        }
    } catch (e) {
        addAiLog(`Fatal software error: ${e.message}`, 'thought');
        if (!isAuto) showToast('Error: ' + e.message, 'error');
    }

    if (loadingEl) loadingEl.classList.add('hidden');
    setLoading(btn, false);
}

async function applyAiRecommendation(type) {
    if (!parsedAiRecommendation) return;

    try {
        let result;
        if (type === 'lte' && parsedAiRecommendation.lte_bands) {
            addAiLog(`Applying bands: ${parsedAiRecommendation.lte_bands}`, 'action');
            result = await apiCall('lte_bands', 'POST', { bands: parsedAiRecommendation.lte_bands });
        } else if (type === 'nr' && parsedAiRecommendation.nr_bands) {
            addAiLog(`Applying 5G bands: ${parsedAiRecommendation.nr_bands}`, 'action');
            result = await apiCall('nr_bands', 'POST', { bands: parsedAiRecommendation.nr_bands });
        } else if (type === 'mode' && parsedAiRecommendation.network_mode) {
            addAiLog(`Switching mode to: ${parsedAiRecommendation.network_mode}`, 'action');
            result = await apiCall('network_mode', 'POST', { mode: parsedAiRecommendation.network_mode });
        }

        if (result && result.success) {
            addAiLog(`Successfully updated ${type.toUpperCase()}!`, 'result');
            showToast(`${type} optimization applied`, 'success');
        } else if (result) {
            addAiLog(`Failed to update ${type}: ${result.error}`, 'thought');
            showToast(`Update failed: ${result.error}`, 'error');
        }
    } catch (e) {
        addAiLog(`Execution Error: ${e.message}`, 'thought');
    }
}

// ============= High-Density One-View One-View Helpers =============

function startAutoOptimize() {
    if (autoOptimizeInterval) stopAutoOptimize();

    // Immediate initial run
    getAiRecommendation(true);

    autoOptimizeInterval = setInterval(() => {
        getAiRecommendation(true);
    }, AUTO_OPTIMIZE_COOLDOWN);
}

function stopAutoOptimize() {
    if (autoOptimizeInterval) {
        clearInterval(autoOptimizeInterval);
        autoOptimizeInterval = null;
    }
}

// ============= Event Listeners =============

document.addEventListener('DOMContentLoaded', () => {
    // Load Ollama models on page load
    loadOllamaModels();

    // Helper for safe element binding
    const bind = (id, event, handler) => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener(event, handler);
        } else {
            console.warn(`[DEBUG] Element NOT found for binding: ${id}`);
        }
    };

    // Connect/Disconnect
    bind('btn-connect', 'click', connect);
    bind('btn-disconnect', 'click', disconnect);

    // Enter key on password field
    const passInp = document.getElementById('password');
    if (passInp) {
        passInp.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') connect();
        });
    }

    // Network mode buttons
    document.querySelectorAll('.btn-mode').forEach(btn => {
        btn.addEventListener('click', () => setNetworkMode(btn.dataset.mode));
    });

    // LTE band buttons
    document.querySelectorAll('.btn-band-lte').forEach(btn => {
        btn.addEventListener('click', () => setLteBands(btn.dataset.bands));
    });

    // Custom LTE bands
    bind('btn-custom-lte', 'click', () => {
        const bands = document.getElementById('custom-lte-bands').value;
        if (bands) setLteBands(bands);
    });

    // NR band buttons
    document.querySelectorAll('.btn-band-nr').forEach(btn => {
        btn.addEventListener('click', () => setNrBands(btn.dataset.bands));
    });

    // Custom NR bands
    bind('btn-custom-nr', 'click', () => {
        const bands = document.getElementById('custom-nr-bands').value;
        if (bands) setNrBands(bands);
    });

    // LTE cell lock
    bind('btn-lock-lte', 'click', () => {
        const pci = parseInt(document.getElementById('lte-lock-pci').value ||
            document.getElementById('lte-lock-pci').placeholder);
        const earfcn = parseInt(document.getElementById('lte-lock-earfcn').value ||
            document.getElementById('lte-lock-earfcn').placeholder);
        if (!isNaN(pci) && !isNaN(earfcn)) lockLteCell(pci, earfcn);
    });

    bind('btn-unlock-lte', 'click', () => lockLteCell(0, 0));

    // NR cell lock
    bind('btn-lock-nr', 'click', () => {
        const pci = parseInt(document.getElementById('nr-lock-pci').value ||
            document.getElementById('nr-lock-pci').placeholder);
        const arfcn = parseInt(document.getElementById('nr-lock-arfcn').value ||
            document.getElementById('nr-lock-arfcn').placeholder);
        const band = parseInt(document.getElementById('nr-lock-band').value ||
            document.getElementById('nr-lock-band').placeholder);
        const scs = parseInt(document.getElementById('nr-lock-scs').value) || 30;
        if (!isNaN(pci) && !isNaN(arfcn) && !isNaN(band)) lockNrCell(pci, arfcn, band, scs);
    });

    bind('btn-unlock-nr', 'click', () => lockNrCell(0, 0, 0, 0));

    // Bridge mode
    bind('btn-bridge-on', 'click', () => setBridgeMode(true));
    bind('btn-bridge-off', 'click', () => setBridgeMode(false));

    // ARP proxy
    bind('btn-arp-on', 'click', () => setArpProxy(true));
    bind('btn-arp-off', 'click', () => setArpProxy(false));

    // Reboot
    bind('btn-reboot', 'click', rebootRouter);

    // AI
    bind('btn-ai-recommend', 'click', () => getAiRecommendation(false));

    // AI quick apply buttons
    bind('btn-apply-lte', 'click', () => applyAiRecommendation('lte'));
    bind('btn-apply-nr', 'click', () => applyAiRecommendation('nr'));
    bind('btn-apply-mode', 'click', () => applyAiRecommendation('mode'));

    // Auto-Optimize Toggle
    const autoToggle = document.getElementById('auto-optimize-toggle');
    if (autoToggle) {
        autoToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                addAiLog('Full Auto-Optimize enabled (3m cycle).', 'action');
                startAutoOptimize();
            } else {
                addAiLog('Full Auto-Optimize disabled.', 'thought');
                stopAutoOptimize();
            }
        });
    }
});
