/**
 * Algo Trading Web Dashboard — App Logic
 * 
 * Dual-loop architecture:
 *   - FAST loop (100ms): Price-only updates via /api/v2/price
 *   - SLOW loop (7s):    Full prediction + sentiment via /api/v1/predict
 *   - Chat: On-demand via /api/v2/chat
 */

const API_BASE = window.location.origin;
const PRICE_INTERVAL_MS = 1000;  // 1s (backend caches for 5s, no benefit from faster)
const DATA_INTERVAL_MS = 7000;   // 7s for full data
const CLOCK_INTERVAL_MS = 1000;

// ── 104 Tickers (VN100 + Viettel Group) ──────────────────────
const TICKERS = [
  "AAA","ACB","BCM","BID","BMP","BVH","BWE","CCQ","CII","CMG",
  "CRE","CTD","CTG","CTR","DBC","DCM","DGC","DGW","DHC","DIG",
  "DPM","DPR","DXG","EIB","EVF","FCN","FPT","FRT","FTS","GAS",
  "GEX","GMD","GVR","HAG","HCM","HDBank","HDG","HHV","HPG","HSG",
  "HT1","HVN","IDC","IMP","KBC","KDC","KDH","KOS","LPB","MBB",
  "MCH","MIG","MSB","MSN","MWG","NAB","NKG","NLG","NT2","NVL",
  "OCB","PC1","PDR","PHR","PLX","PNJ","POW","PPC","PVD","PVS",
  "PVT","REE","SAB","SBT","SCS","SHB","SJS","SSI","STB","SZC",
  "TCB","TCH","TDM","TLG","TPB","TRA","VCB","VCI","VGI","VHC",
  "VHM","VIB","VIC","VJC","VND","VNM","VOS","VPB","VPI","VRE",
  "VSH","VTK","VTO","VTP"
];

let currentTicker = "FPT";
let chatHistory = [];
let lastPredictionData = null;
let priceTimerId = null;
let dataTimerId = null;

// ══════════════════════════════════════════════════════════════
// INITIALIZATION
// ══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  initTickerDropdown();
  initChatInput();
  startClock();
  
  // Initial fetch
  fetchFullData();
  fetchPrice();
  
  // Start dual loops
  priceTimerId = setInterval(fetchPrice, PRICE_INTERVAL_MS);
  dataTimerId = setInterval(fetchFullData, DATA_INTERVAL_MS);
});


// ══════════════════════════════════════════════════════════════
// TICKER SEARCH DROPDOWN
// ══════════════════════════════════════════════════════════════

function initTickerDropdown() {
  const input = document.getElementById('tickerInput');
  const dropdown = document.getElementById('tickerDropdown');
  let highlightIdx = -1;

  function renderOptions(filter) {
    const filtered = filter
      ? TICKERS.filter(t => t.toLowerCase().includes(filter.toLowerCase()))
      : TICKERS;
    
    dropdown.innerHTML = filtered.map((t, i) => 
      `<div class="ticker-option${i === highlightIdx ? ' highlighted' : ''}" data-ticker="${t}">
        <span class="ticker-code">${t}</span>
      </div>`
    ).join('');
    
    dropdown.classList.toggle('active', filtered.length > 0 && document.activeElement === input);
    
    // Click handlers
    dropdown.querySelectorAll('.ticker-option').forEach(opt => {
      opt.addEventListener('mousedown', (e) => {
        e.preventDefault();
        selectTicker(opt.dataset.ticker);
      });
    });
  }

  input.addEventListener('focus', () => {
    highlightIdx = -1;
    renderOptions(input.value);
  });

  input.addEventListener('blur', () => {
    setTimeout(() => dropdown.classList.remove('active'), 200);
  });

  input.addEventListener('input', (e) => {
    highlightIdx = -1;
    renderOptions(e.target.value);
  });

  input.addEventListener('keydown', (e) => {
    const items = dropdown.querySelectorAll('.ticker-option');
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      highlightIdx = Math.min(highlightIdx + 1, items.length - 1);
      renderOptions(input.value);
      items[highlightIdx]?.scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      highlightIdx = Math.max(highlightIdx - 1, 0);
      renderOptions(input.value);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (highlightIdx >= 0 && items[highlightIdx]) {
        selectTicker(items[highlightIdx].dataset.ticker);
      } else if (input.value.trim()) {
        selectTicker(input.value.trim().toUpperCase());
      }
    } else if (e.key === 'Escape') {
      dropdown.classList.remove('active');
      input.blur();
    }
  });
}

function selectTicker(ticker) {
  ticker = ticker.toUpperCase();
  currentTicker = ticker;
  document.getElementById('tickerInput').value = ticker;
  document.getElementById('tickerDropdown').classList.remove('active');
  
  // Reset UI
  document.getElementById('livePrice').textContent = '—';
  document.getElementById('liveChange').textContent = 'Loading...';
  document.getElementById('liveChange').className = 'live-change neutral';
  lastPredictionData = null;
  
  // Immediate fetch
  fetchPrice();
  fetchFullData();
}


// ══════════════════════════════════════════════════════════════
// FAST LOOP: Price Updates (100ms)
// ══════════════════════════════════════════════════════════════

async function fetchPrice() {
  try {
    const res = await fetch(`${API_BASE}/api/v2/price?ticker=${currentTicker}`, {
      signal: AbortSignal.timeout(2000)
    });
    if (!res.ok) return;
    const data = await res.json();
    updatePrice(data.price, data.change);
  } catch {
    // Silent fail — price endpoint might not exist yet, fall back to prediction data
  }
}

function updatePrice(price, change) {
  if (price == null || price === 0) return;
  
  const priceEl = document.getElementById('livePrice');
  const changeEl = document.getElementById('liveChange');
  
  const oldPrice = priceEl.textContent;
  const newPriceStr = formatNumber(price);
  
  if (oldPrice !== newPriceStr) {
    priceEl.textContent = newPriceStr;
    priceEl.classList.add('pulse-update');
    setTimeout(() => priceEl.classList.remove('pulse-update'), 600);
  }
  
  if (change != null) {
    const sign = change >= 0 ? '+' : '';
    changeEl.textContent = `${sign}${change.toFixed(2)}%`;
    changeEl.className = `live-change ${change > 0 ? 'up' : change < 0 ? 'down' : 'neutral'}`;
  }
}


// ══════════════════════════════════════════════════════════════
// SLOW LOOP: Full Prediction Data (7s)
// ══════════════════════════════════════════════════════════════

async function fetchFullData() {
  const statusEl = document.getElementById('aiStatus');
  
  try {
    statusEl.textContent = 'AI: SYNCING';
    statusEl.className = 'status-badge syncing';
    
    const res = await fetch(`${API_BASE}/api/v1/predict?ticker=${currentTicker}`, {
      signal: AbortSignal.timeout(15000)
    });
    
    if (!res.ok) {
      // Try to still show price even if predict fails
      statusEl.textContent = 'AI: FALLBACK';
      statusEl.className = 'status-badge offline';
      return;
    }
    
    const data = await res.json();
    lastPredictionData = data;
    
    statusEl.textContent = 'AI: ONLINE';
    statusEl.className = 'status-badge online';
    
    renderTechnical(data);
    renderSentiment(data);
    renderFusion(data);
    
  } catch (err) {
    if (err.name !== 'AbortError') {
      statusEl.textContent = 'AI: OFFLINE';
      statusEl.className = 'status-badge offline';
    }
  }
}


// ══════════════════════════════════════════════════════════════
// RENDER: Technical Panel
// ══════════════════════════════════════════════════════════════

function renderTechnical(data) {
  const tech = data.technical;
  if (!tech) return;
  
  const horizons = tech.horizons || [];
  const tbody = document.querySelector('#horizonTable tbody');
  const labels = { '1w': '1W', '1m': '1M', '6m': '6M', 'short': '1W', 'mid': '1M', 'long': '6M' };
  
  tbody.innerHTML = '';
  
  if (horizons.length === 0) {
    tbody.innerHTML = '<tr><td colspan="3" class="text-muted">No horizon data</td></tr>';
  }
  
  horizons.forEach(h => {
    const probs = h.trend_probs || {};
    const pUp = probs.up || 0;
    const pDown = probs.down || 0;
    
    let trendHtml;
    if (pUp > 0.55) {
      trendHtml = `<span class="text-green fw-700">UP (${(pUp * 100).toFixed(0)}%)</span>`;
    } else if (pDown > 0.55) {
      trendHtml = `<span class="text-red fw-700">DOWN (${(pDown * 100).toFixed(0)}%)</span>`;
    } else {
      trendHtml = `<span class="text-yellow">SIDE</span>`;
    }
    
    const range = h.expected_range || {};
    const ceiling = range.ceiling_90th || 0;
    const currentPrice = parseFloat(document.getElementById('livePrice').textContent.replace(/,/g, '')) || 0;
    let forecastHtml = '<span class="text-muted">—</span>';
    if (ceiling > 0 && currentPrice > 0) {
      const upside = ((ceiling - currentPrice) / currentPrice) * 100;
      forecastHtml = `<span class="${upside > 0 ? 'text-green' : 'text-red'} mono">${upside >= 0 ? '+' : ''}${upside.toFixed(1)}%</span>`;
    }
    
    const label = labels[h.horizon] || h.horizon;
    const row = document.createElement('tr');
    row.innerHTML = `<td class="fw-700">${label}</td><td>${trendHtml}</td><td>${forecastHtml}</td>`;
    tbody.appendChild(row);
  });
  
  // Range box
  const h0 = horizons[0] || {};
  const range = h0.expected_range || {};
  document.getElementById('rangeFloor').textContent = formatNumber(range.bottom_10th || 0);
  document.getElementById('rangePivot').textContent = formatNumber(range.median_50th || 0);
  document.getElementById('rangeCeiling').textContent = formatNumber(range.ceiling_90th || 0);
  
  // Gauges
  const probs = h0.trend_probs || {};
  const bullPct = ((probs.up || 0) * 100).toFixed(0);
  const bearPct = ((probs.down || 0) * 100).toFixed(0);
  
  document.getElementById('gaugeBull').style.width = `${bullPct}%`;
  document.getElementById('gaugeBullVal').textContent = `${bullPct}%`;
  document.getElementById('gaugeBear').style.width = `${bearPct}%`;
  document.getElementById('gaugeBearVal').textContent = `${bearPct}%`;
}


// ══════════════════════════════════════════════════════════════
// RENDER: Sentiment Panel
// ══════════════════════════════════════════════════════════════

function renderSentiment(data) {
  const sent = data.sentiment;
  
  const badge = document.getElementById('regimeBadge');
  const scoreEl = document.getElementById('sentimentScore');
  const headlinesEl = document.getElementById('headlinesBox');
  const rationaleEl = document.getElementById('rationaleBox');
  
  if (!sent) {
    badge.textContent = 'NO DATA';
    badge.className = 'regime-badge neutral';
    scoreEl.textContent = '—';
    headlinesEl.textContent = 'Không có dữ liệu sentiment.';
    return;
  }
  
  // Regime Badge
  const regime = (sent.sentiment_regime || 'neutral').toLowerCase();
  let badgeClass = 'neutral';
  let badgeText = regime.toUpperCase();
  if (regime.includes('greed') || regime.includes('bullish') || regime.includes('positive')) {
    badgeClass = 'bullish';
  } else if (regime.includes('fear') || regime.includes('bearish') || regime.includes('negative')) {
    badgeClass = 'bearish';
  }
  badge.textContent = badgeText;
  badge.className = `regime-badge ${badgeClass}`;
  
  // Score
  const score = sent.sentiment_score || 0;
  scoreEl.textContent = (score >= 0 ? '+' : '') + score.toFixed(2);
  scoreEl.className = `info-value mono ${score > 0 ? 'text-green' : score < 0 ? 'text-red' : 'text-yellow'}`;
  
  // Headlines from source_breakdown
  if (sent.source_breakdown && sent.source_breakdown.length > 0) {
    headlinesEl.innerHTML = sent.source_breakdown.map(s => 
      `<div style="margin-bottom: 6px;">• ${escapeHtml(s.headline || s.source)}</div>`
    ).join('');
  } else if (sent.summary) {
    headlinesEl.textContent = sent.summary;
  }
  
  // Rationale from fusion
  if (data.fusion?.rationale) {
    rationaleEl.textContent = data.fusion.rationale;
  }
}


// ══════════════════════════════════════════════════════════════
// RENDER: Fusion & Risk Panel
// ══════════════════════════════════════════════════════════════

function renderFusion(data) {
  const fusion = data.fusion;
  const risk = data.risk;
  
  // Action Badge
  const actionEl = document.getElementById('actionBadge');
  const action = fusion?.action || 'STANDBY';
  actionEl.textContent = action;
  
  if (action.includes('BUY')) {
    actionEl.className = 'action-badge buy';
  } else if (action.includes('SELL')) {
    actionEl.className = 'action-badge sell';
  } else {
    actionEl.className = 'action-badge hold';
  }
  
  // Fusion info
  document.getElementById('consensusScore').textContent = (fusion?.confidence || 0).toFixed(3);
  
  // Risk info
  const allocation = risk?.position_size_suggestion || 0;
  document.getElementById('allocation').textContent = `${(allocation * 100).toFixed(1)}% portfolio`;
  
  const vetoEl = document.getElementById('vetoStatus');
  if (risk?.veto_flag) {
    vetoEl.textContent = 'BLOCKED (VETO)';
    vetoEl.className = 'info-value veto-blocked';
  } else {
    vetoEl.textContent = 'CLEARED';
    vetoEl.className = 'info-value veto-cleared';
  }
  
  // Constraints
  const constraintsRow = document.getElementById('constraintsRow');
  if (risk?.constraints_hit && risk.constraints_hit.length > 0) {
    constraintsRow.style.display = 'flex';
    document.getElementById('constraints').textContent = risk.constraints_hit.join(', ');
  } else {
    constraintsRow.style.display = 'none';
  }
  
  // Accuracy
  const accEl = document.getElementById('modelAcc');
  const acc = risk?.model_accuracy_1w;
  accEl.textContent = acc ? `${(acc * 100).toFixed(1)}%` : '—';
  
  // Fusion rationale
  if (fusion?.rationale) {
    document.getElementById('fusionRationale').textContent = fusion.rationale;
  }
}


// ══════════════════════════════════════════════════════════════
// CHAT: LLM Integration
// ══════════════════════════════════════════════════════════════

function initChatInput() {
  const input = document.getElementById('chatInput');
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const btn = document.getElementById('chatSendBtn');
  const messagesEl = document.getElementById('chatMessages');
  
  const message = input.value.trim();
  if (!message) return;
  
  // Add user message to UI
  appendChatMessage('user', message);
  input.value = '';
  btn.disabled = true;
  
  // Add thinking indicator
  const thinkingEl = appendChatMessage('thinking', 'AI đang suy nghĩ...');
  
  try {
    const res = await fetch(`${API_BASE}/api/v2/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        history: chatHistory,
        ticker: currentTicker
      }),
      signal: AbortSignal.timeout(60000)
    });
    
    // Remove thinking indicator
    thinkingEl.remove();
    
    if (!res.ok) {
      appendChatMessage('assistant', `Lỗi: Server trả về ${res.status}. Kiểm tra Ollama đã chạy chưa.`);
      btn.disabled = false;
      return;
    }
    
    const data = await res.json();
    const reply = data.response || data.reply || data.answer || 'Không có phản hồi.';
    
    appendChatMessage('assistant', reply);
    
    // Update history for context
    chatHistory.push({ role: 'user', content: message });
    chatHistory.push({ role: 'assistant', content: reply });
    
    // Keep history manageable (last 10 exchanges)
    if (chatHistory.length > 20) {
      chatHistory = chatHistory.slice(-20);
    }
    
  } catch (err) {
    thinkingEl.remove();
    if (err.name === 'TimeoutError' || err.name === 'AbortError') {
      appendChatMessage('assistant', 'Timeout — Ollama model quá lâu. Hãy thử lại.');
    } else {
      appendChatMessage('assistant', `Lỗi kết nối: ${err.message}`);
    }
  }
  
  btn.disabled = false;
  input.focus();
}

function appendChatMessage(role, content) {
  const messagesEl = document.getElementById('chatMessages');
  const div = document.createElement('div');
  
  if (role === 'user') {
    div.className = 'chat-msg user';
    div.textContent = content;
  } else if (role === 'thinking') {
    div.className = 'chat-msg thinking';
    div.textContent = content;
  } else {
    div.className = 'chat-msg assistant';
    div.innerHTML = `<div class="msg-label">AI Agent</div>${formatChatContent(content)}`;
  }
  
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function formatChatContent(text) {
  // Basic markdown-like formatting
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}


// ══════════════════════════════════════════════════════════════
// CLOCK
// ══════════════════════════════════════════════════════════════

function startClock() {
  function tick() {
    document.getElementById('clock').textContent = 
      new Date().toLocaleTimeString('vi-VN', { hour12: false });
  }
  tick();
  setInterval(tick, CLOCK_INTERVAL_MS);
}


// ══════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════

function formatNumber(n) {
  if (n == null || isNaN(n)) return '—';
  return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
