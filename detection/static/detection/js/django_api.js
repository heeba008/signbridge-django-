// ===== DJANGO REST API INTEGRATION =====
// This file connects the SignBridge frontend to the Django backend.
// Every detected sign is POST-ed to /api/history/ and saved in SQLite.

// ── Helpers ──────────────────────────────────────────────────────────────────

function getCookie(name) {
  const cookies = document.cookie.split(';');
  for (let c of cookies) {
    c = c.trim();
    if (c.startsWith(name + '=')) return decodeURIComponent(c.slice(name.length + 1));
  }
  return null;
}

function showToast(msg = '✓ Saved to Django DB') {
  const toast = document.getElementById('saveToast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2000);
}

const API_HEADERS = {
  'Content-Type': 'application/json',
  'X-CSRFToken': DJANGO_CSRF,
};

// ── Save a detected sign to Django ────────────────────────────────────────────

async function saveSignToDjango(sign, info, confidence) {
  try {
    const payload = {
      sign: sign,
      meaning: info.meaning || 'Unknown',
      category: info.category || 'Unknown',
      confidence: Math.round(confidence * 100),  // store as percentage int
      session_id: SESSION_ID,
    };
    const res = await fetch('/api/history/', {
      method: 'POST',
      headers: API_HEADERS,
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      showToast(`✓ "${sign}" saved to DB`);
    }
  } catch (err) {
    console.warn('Django API save failed:', err);
  }
}

// ── Load history from Django DB ───────────────────────────────────────────────

async function loadHistoryFromDjango() {
  try {
    showToast('Loading from DB…');
    const res = await fetch(`/api/history/?session_id=${SESSION_ID}`);
    const data = await res.json();

    if (data.results && data.results.length > 0) {
      // Inject into the in-memory history array and re-render
      signHistory = data.results.map(item => ({
        sign: item.sign,
        info: { meaning: item.meaning, category: item.category, emoji: getEmoji(item.sign) },
        confidence: item.confidence / 100,
        time: item.detected_at.split(' ')[1] || item.detected_at,
      }));
      renderHistory();
      showToast(`✓ Loaded ${data.count} signs from DB`);
    } else {
      showToast('No DB history for this session');
    }
  } catch (err) {
    console.warn('Django load failed:', err);
    showToast('⚠ DB load failed');
  }
}

// ── Clear Django DB history ───────────────────────────────────────────────────

async function clearDjangoHistory() {
  if (!confirm('Clear ALL sign history from the Django database?')) return;
  try {
    const res = await fetch('/api/history/clear/', {
      method: 'DELETE',
      headers: API_HEADERS,
    });
    const data = await res.json();
    showToast(`✓ Deleted ${data.deleted} records from DB`);
  } catch (err) {
    showToast('⚠ DB clear failed');
  }
}

// ── Save sentence to Django ───────────────────────────────────────────────────

async function saveSentenceToDjango() {
  if (!sentenceWords || sentenceWords.length === 0) {
    showToast('⚠ No sentence to save');
    return;
  }
  const text = sentenceWords.join(' ');
  try {
    const res = await fetch('/api/sentences/', {
      method: 'POST',
      headers: API_HEADERS,
      body: JSON.stringify({ text, session_id: SESSION_ID }),
    });
    if (res.ok) {
      showToast(`✓ Sentence saved to DB`);
    }
  } catch (err) {
    showToast('⚠ Sentence save failed');
  }
}

// ── Helper: emoji lookup ──────────────────────────────────────────────────────

function getEmoji(sign) {
  if (typeof ASL_SIGNS !== 'undefined' && ASL_SIGNS[sign]) {
    return ASL_SIGNS[sign].emoji || '✋';
  }
  return '✋';
}

// ── Patch addToHistory to also save to Django ─────────────────────────────────
// We wrap the original addToHistory so it also fires the API call.

const _originalAddToHistory = addToHistory;
window.addToHistory = function(sign, info, confidence) {
  _originalAddToHistory(sign, info, confidence);
  saveSignToDjango(sign, info, confidence);
};

console.log('%c⚙ Django REST API connected', 'color:#44b78b;font-size:0.9rem;font-weight:bold');
console.log('%cEndpoints: /api/history/ · /api/stats/ · /api/sentences/', 'color:#9898b8');
