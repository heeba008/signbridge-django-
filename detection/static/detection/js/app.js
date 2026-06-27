// ===== SIGN LANGUAGE DETECTION APP =====

// --- ASL Gesture Database ---
const ASL_SIGNS = {
  // Letters
  A: { meaning: "Letter A", category: "Alphabet", emoji: "✊", description: "Fist with thumb beside index finger" },
  B: { meaning: "Letter B", category: "Alphabet", emoji: "🖐️", description: "Flat hand, fingers up, thumb across" },
  C: { meaning: "Letter C", category: "Alphabet", emoji: "🤏", description: "Curved hand forming letter C" },
  D: { meaning: "Letter D", category: "Alphabet", emoji: "👆", description: "Index finger up, others curled into thumb" },
  E: { meaning: "Letter E", category: "Alphabet", emoji: "✋", description: "Fingers bent, touching thumb" },
  F: { meaning: "Letter F", category: "Alphabet", emoji: "👌", description: "Index and thumb touching, other fingers up" },
  G: { meaning: "Letter G", category: "Alphabet", emoji: "👉", description: "Index and thumb pointing sideways" },
  H: { meaning: "Letter H", category: "Alphabet", emoji: "✌️", description: "Index and middle finger pointing sideways" },
  I: { meaning: "Letter I", category: "Alphabet", emoji: "🤙", description: "Pinky finger up, fist formed" },
  J: { meaning: "Letter J", category: "Alphabet", emoji: "🤙", description: "Pinky draws letter J in the air" },
  K: { meaning: "Letter K", category: "Alphabet", emoji: "✌️", description: "Index, middle up, thumb between them" },
  L: { meaning: "Letter L", category: "Alphabet", emoji: "🤟", description: "L shape with index and thumb" },
  M: { meaning: "Letter M", category: "Alphabet", emoji: "✊", description: "Three fingers over thumb" },
  N: { meaning: "Letter N", category: "Alphabet", emoji: "✊", description: "Two fingers over thumb" },
  O: { meaning: "Letter O", category: "Alphabet", emoji: "👌", description: "All fingers curved to form O" },
  P: { meaning: "Letter P", category: "Alphabet", emoji: "👇", description: "Like K but pointing downward" },
  Q: { meaning: "Letter Q", category: "Alphabet", emoji: "👇", description: "Like G but pointing downward" },
  R: { meaning: "Letter R", category: "Alphabet", emoji: "✌️", description: "Crossed index and middle fingers" },
  S: { meaning: "Letter S", category: "Alphabet", emoji: "✊", description: "Fist with thumb over fingers" },
  T: { meaning: "Letter T", category: "Alphabet", emoji: "✊", description: "Thumb between index and middle" },
  U: { meaning: "Letter U", category: "Alphabet", emoji: "✌️", description: "Index and middle fingers up together" },
  V: { meaning: "Letter V", category: "Alphabet", emoji: "✌️", description: "V shape with index and middle" },
  W: { meaning: "Letter W", category: "Alphabet", emoji: "🖖", description: "Three fingers spread up" },
  X: { meaning: "Letter X", category: "Alphabet", emoji: "☝️", description: "Index finger hooked" },
  Y: { meaning: "Letter Y", category: "Alphabet", emoji: "🤙", description: "Thumb and pinky out" },
  Z: { meaning: "Letter Z", category: "Alphabet", emoji: "☝️", description: "Index finger draws Z in the air" },

  // Numbers
  "1": { meaning: "Number 1", category: "Numbers", emoji: "☝️", description: "Index finger pointing up" },
  "2": { meaning: "Number 2", category: "Numbers", emoji: "✌️", description: "Peace sign / two fingers" },
  "3": { meaning: "Number 3", category: "Numbers", emoji: "🤟", description: "Thumb, index, middle extended" },
  "4": { meaning: "Number 4", category: "Numbers", emoji: "🖖", description: "Four fingers up, thumb folded" },
  "5": { meaning: "Number 5", category: "Numbers", emoji: "🖐️", description: "All five fingers spread" },

  // Common phrases/gestures
  "HELLO": { meaning: "Hello / Hi", category: "Phrases", emoji: "👋", description: "Open hand wave to forehead" },
  "THANK YOU": { meaning: "Thank You", category: "Phrases", emoji: "🙏", description: "Flat hand from chin outward" },
  "YES": { meaning: "Yes", category: "Phrases", emoji: "✅", description: "Fist nodding up and down" },
  "NO": { meaning: "No", category: "Phrases", emoji: "❌", description: "Index and middle tap thumb together" },
  "PLEASE": { meaning: "Please", category: "Phrases", emoji: "🙏", description: "Flat hand circular motion on chest" },
  "SORRY": { meaning: "Sorry", category: "Phrases", emoji: "😔", description: "Fist circles on chest" },
  "LOVE": { meaning: "I Love You", category: "Phrases", emoji: "🤟", description: "Thumb, index, pinky extended" },
  "HELP": { meaning: "Help", category: "Phrases", emoji: "🆘", description: "Thumbs up on flat palm, lifted" },
  "STOP": { meaning: "Stop", category: "Phrases", emoji: "✋", description: "Flat hand chopped down" },
  "MORE": { meaning: "More", category: "Phrases", emoji: "👐", description: "Fingertips tap together twice" },
};

// --- Gesture Classifier using MediaPipe landmarks ---
class GestureClassifier {
  constructor() {
    this.lastSign = null;
    this.stableCount = 0;
    this.STABLE_THRESHOLD = 8;
  }

  // Calculate angle between three points
  angle(A, B, C) {
    const radians = Math.atan2(C.y - B.y, C.x - B.x) - Math.atan2(A.y - B.y, A.x - B.x);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    if (angle > 180) angle = 360 - angle;
    return angle;
  }

  // Check if finger is extended
  isExtended(landmarks, tip, pip, mcp) {
    return landmarks[tip].y < landmarks[pip].y && landmarks[pip].y < landmarks[mcp].y;
  }

  // Get finger states: [thumb, index, middle, ring, pinky]
  getFingerStates(landmarks) {
    const fingers = [];

    // Thumb (special case - check x axis for right hand)
    const thumbExtended = landmarks[4].x < landmarks[3].x;
    fingers.push(thumbExtended);

    // Other fingers: tip.y < pip.y means extended (since y increases downward)
    fingers.push(landmarks[8].y < landmarks[6].y);   // Index
    fingers.push(landmarks[12].y < landmarks[10].y); // Middle
    fingers.push(landmarks[16].y < landmarks[14].y); // Ring
    fingers.push(landmarks[20].y < landmarks[18].y); // Pinky

    return fingers;
  }

  // Euclidean distance between two landmarks
  dist(a, b) {
    return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
  }

  classify(landmarks) {
    if (!landmarks || landmarks.length < 21) return null;

    const f = this.getFingerStates(landmarks);
    const [thumb, index, middle, ring, pinky] = f;

    // --- Phrases / Special Gestures ---
    // ILY / LOVE: thumb + index + pinky up
    if (thumb && index && !middle && !ring && pinky) {
      return { sign: "LOVE", confidence: 0.92 };
    }

    // HELLO: all fingers up, open hand
    if (thumb && index && middle && ring && pinky) {
      const spread = this.dist(landmarks[4], landmarks[20]);
      if (spread > 0.35) return { sign: "HELLO", confidence: 0.88 };
      return { sign: "5", confidence: 0.85 };
    }

    // Number 4: four fingers up, thumb down
    if (!thumb && index && middle && ring && pinky) {
      return { sign: "4", confidence: 0.87 };
    }

    // V / Peace / Number 2: index + middle up
    if (!thumb && index && middle && !ring && !pinky) {
      const cross = Math.abs(landmarks[8].x - landmarks[12].x);
      if (cross < 0.03) return { sign: "R", confidence: 0.78 }; // crossed = R
      return { sign: "V", confidence: 0.85 };
    }

    // Number 1 / D / Index up
    if (!thumb && index && !middle && !ring && !pinky) {
      return { sign: "1", confidence: 0.88 };
    }

    // L: thumb + index up
    if (thumb && index && !middle && !ring && !pinky) {
      return { sign: "L", confidence: 0.90 };
    }

    // Y: thumb + pinky up
    if (thumb && !index && !middle && !ring && pinky) {
      return { sign: "Y", confidence: 0.88 };
    }

    // Fist / A / S / E
    if (!thumb && !index && !middle && !ring && !pinky) {
      const thumbPos = landmarks[4].x;
      const indexKnuckle = landmarks[5].x;
      if (Math.abs(thumbPos - indexKnuckle) < 0.05) return { sign: "A", confidence: 0.78 };
      return { sign: "S", confidence: 0.75 };
    }

    // OK / F / Number 3 like
    if (thumb && !index && middle && ring && pinky) {
      const tipDist = this.dist(landmarks[4], landmarks[8]);
      if (tipDist < 0.06) return { sign: "O", confidence: 0.82 };
      return { sign: "C", confidence: 0.74 };
    }

    // U: index + middle together up
    if (!thumb && index && middle && !ring && !pinky) {
      return { sign: "U", confidence: 0.80 };
    }

    // W: index + middle + ring up
    if (!thumb && index && middle && ring && !pinky) {
      return { sign: "W", confidence: 0.82 };
    }

    // STOP / B: flat hand, fingers together
    if (!thumb && index && middle && ring && pinky) {
      return { sign: "B", confidence: 0.78 };
    }

    // Number 3: thumb + index + middle
    if (thumb && index && middle && !ring && !pinky) {
      return { sign: "3", confidence: 0.83 };
    }

    // I / pinky only
    if (!thumb && !index && !middle && !ring && pinky) {
      return { sign: "I", confidence: 0.85 };
    }

    // K / U variants
    if (!thumb && index && middle && !ring && !pinky) {
      return { sign: "K", confidence: 0.72 };
    }

    return { sign: "?", confidence: 0.3 };
  }

  // Stabilize output: only emit when same sign detected consistently
  stabilize(result) {
    if (!result) { this.stableCount = 0; return null; }

    if (result.sign === this.lastSign) {
      this.stableCount++;
    } else {
      this.lastSign = result.sign;
      this.stableCount = 1;
    }

    if (this.stableCount >= this.STABLE_THRESHOLD) {
      return result;
    }
    return { sign: result.sign, confidence: result.confidence, pending: true };
  }
}

// ===== APP STATE =====
let stream = null;
let camera = null;
let hands = null;
let animFrame = null;
let isDetecting = false;
let currentMode = 'asl';
let isMirrored = true;
let sessionStart = null;
let sessionTimer = null;
let signHistory = [];
let sentenceWords = [];
let totalSigns = 0;
let confidenceSum = 0;
let confidenceCount = 0;
let lastEmittedSign = null;
let lastEmitTime = 0;
const EMIT_COOLDOWN = 1500; // ms between sign emissions to history

const classifier = new GestureClassifier();
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');

// ===== MEDIAPIPE SETUP =====
function initMediaPipe() {
  hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.6,
  });

  hands.onResults(onResults);
}

function onResults(results) {
  // Clear canvas
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const handIndicator = document.getElementById('handIndicator');

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    handIndicator.classList.add('visible');

    for (const landmarks of results.multiHandLandmarks) {
      // Draw connections
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
        color: 'rgba(0,255,204,0.5)',
        lineWidth: 2
      });
      // Draw landmarks
      drawLandmarks(ctx, landmarks, {
        color: 'rgba(0,255,204,0.9)',
        fillColor: 'rgba(0,255,204,0.3)',
        lineWidth: 1,
        radius: 4
      });

      // Classify
      const raw = classifier.classify(landmarks);
      const result = classifier.stabilize(raw);

      if (result) {
        updateSignDisplay(result);
      }
    }
  } else {
    handIndicator.classList.remove('visible');
    // Fade out display slowly
    if (document.getElementById('signDisplay').textContent !== '—') {
      // Keep showing last sign briefly
    }
  }
}

// ===== START / STOP =====
async function startDetection() {
  try {
    setStatus('active', 'Requesting camera access…');
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
    });

    video.srcObject = stream;
    video.style.display = 'block';
    document.getElementById('videoIdle').style.display = 'none';

    await new Promise(resolve => video.onloadedmetadata = resolve);
    video.play();

    // Size canvas to video
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;

    initMediaPipe();

    camera = new Camera(video, {
      onFrame: async () => {
        if (hands) await hands.send({ image: video });
      },
      width: 640, height: 480,
    });
    await camera.start();

    isDetecting = true;
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    setStatus('active', 'Detecting — show your hand to the camera');

    // Start session timer
    sessionStart = Date.now();
    sessionTimer = setInterval(updateSessionTimer, 1000);

  } catch (err) {
    setStatus('', 'Camera access denied or unavailable');
    console.error(err);
    alert('Camera access required. Please allow camera permissions and try again.');
  }
}

function stopDetection() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  if (camera) { camera.stop(); camera = null; }
  if (hands) { hands.close(); hands = null; }

  video.style.display = 'none';
  document.getElementById('videoIdle').style.display = 'flex';
  document.getElementById('handIndicator').classList.remove('visible');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  isDetecting = false;
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
  setStatus('', 'Camera stopped');

  clearInterval(sessionTimer);
}

function flipCamera() {
  isMirrored = !isMirrored;
  video.style.transform = isMirrored ? 'scaleX(-1)' : 'scaleX(1)';
  overlay.style.transform = isMirrored ? 'scaleX(-1)' : 'scaleX(1)';
}

// ===== DISPLAY UPDATE =====
let displayTimeout = null;
let lastSignData = null;

function updateSignDisplay(result) {
  if (result.pending) return;

  const signKey = result.sign;
  const signInfo = ASL_SIGNS[signKey] || { meaning: 'Unknown gesture', category: '', emoji: '❓' };

  const displayEl = document.getElementById('signDisplay');
  const meaningEl = document.getElementById('signMeaning');
  const categoryEl = document.getElementById('signCategory');
  const confFill = document.getElementById('confFill');
  const confVal = document.getElementById('confVal');

  const conf = Math.round(result.confidence * 100);

  // Update display
  if (displayEl.textContent !== signKey) {
    displayEl.classList.remove('sign-flash');
    void displayEl.offsetWidth; // reflow
    displayEl.classList.add('sign-flash');
  }

  displayEl.textContent = signKey;
  meaningEl.textContent = signInfo.meaning;
  categoryEl.textContent = signInfo.category;
  confFill.style.width = conf + '%';
  confVal.textContent = conf + '%';
  setStatus('detecting', `Detected: ${signInfo.meaning}`);

  lastSignData = { sign: signKey, info: signInfo, confidence: result.confidence };

  // Auto-add to history with cooldown
  const now = Date.now();
  if (signKey !== lastEmittedSign || (now - lastEmitTime) > EMIT_COOLDOWN) {
    if (signKey !== '?') {
      addToHistory(signKey, signInfo, result.confidence);
      lastEmittedSign = signKey;
      lastEmitTime = now;
    }
  }

  // Update stats
  confidenceSum += result.confidence;
  confidenceCount++;
  document.getElementById('statAccuracy').textContent =
    Math.round((confidenceSum / confidenceCount) * 100) + '%';
}

// ===== HISTORY =====
function addToHistory(sign, info, confidence) {
  totalSigns++;
  document.getElementById('statTotal').textContent = totalSigns;

  const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const item = { sign, info, confidence, time };
  signHistory.unshift(item);
  if (signHistory.length > 50) signHistory.pop();
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById('historyList');
  if (signHistory.length === 0) {
    list.innerHTML = '<p class="history-empty">No signs detected yet.</p>';
    return;
  }
  list.innerHTML = signHistory.slice(0, 20).map(item => `
    <div class="history-item">
      <span class="history-sign">${item.info.emoji || item.sign}</span>
      <span class="history-meaning">${item.info.meaning}</span>
      <span class="history-conf">${Math.round(item.confidence * 100)}%</span>
      <span class="history-time">${item.time}</span>
    </div>
  `).join('');
}

function clearHistory() {
  signHistory = [];
  totalSigns = 0;
  confidenceSum = 0;
  confidenceCount = 0;
  document.getElementById('statTotal').textContent = '0';
  document.getElementById('statAccuracy').textContent = '—';
  renderHistory();
}

// ===== SENTENCE BUILDER =====
function addToSentence() {
  if (!lastSignData || lastSignData.sign === '?') return;
  const word = lastSignData.sign;
  sentenceWords.push(word);
  renderSentence();
}

function renderSentence() {
  const box = document.getElementById('sentenceBox');
  if (sentenceWords.length === 0) {
    box.innerHTML = '<span class="sentence-placeholder">Detected signs will appear here…</span>';
    return;
  }
  box.innerHTML = sentenceWords.map(w => `<span class="sentence-word">${w}</span>`).join(' ');
}

function clearSentence() {
  sentenceWords = [];
  renderSentence();
}

function speakSentence() {
  if (sentenceWords.length === 0) return;
  const text = sentenceWords.join(' ');
  speak(text);
}

// ===== TEXT-TO-SPEECH =====
let voices = [];

function loadVoices() {
  voices = speechSynthesis.getVoices();
  const select = document.getElementById('voiceSelect');
  select.innerHTML = voices.map((v, i) =>
    `<option value="${i}">${v.name} (${v.lang})</option>`
  ).join('');
}

speechSynthesis.onvoiceschanged = loadVoices;
loadVoices();

function speak(text) {
  if (!text) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const voiceIdx = document.getElementById('voiceSelect').value;
  if (voices[voiceIdx]) utterance.voice = voices[voiceIdx];
  utterance.rate = parseFloat(document.getElementById('speechRate').value);
  utterance.pitch = parseFloat(document.getElementById('speechPitch').value);
  speechSynthesis.speak(utterance);
}

// ===== MODE =====
function setMode(mode, btn) {
  currentMode = mode;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
}

// ===== STATUS =====
function setStatus(type, text) {
  const dot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  dot.className = 'status-dot';
  if (type) dot.classList.add(type);
  statusText.textContent = text;
}

// ===== SESSION TIMER =====
function updateSessionTimer() {
  if (!sessionStart) return;
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const s = String(elapsed % 60).padStart(2, '0');
  document.getElementById('statSession').textContent = `${m}:${s}`;
}

// ===== ALPHABET REFERENCE GRID =====
function buildAlphabetGrid() {
  const grid = document.getElementById('alphabetGrid');
  const entries = Object.entries(ASL_SIGNS);
  grid.innerHTML = entries.map(([key, val]) => `
    <div class="alpha-card" onclick="previewSign('${key}')" title="${val.description}">
      <span class="alpha-letter">${key}</span>
      <span class="alpha-emoji">${val.emoji}</span>
      <span class="alpha-desc">${val.category}</span>
    </div>
  `).join('');
}

function previewSign(key) {
  const info = ASL_SIGNS[key];
  if (!info) return;
  document.getElementById('signDisplay').textContent = key;
  document.getElementById('signMeaning').textContent = info.meaning;
  document.getElementById('signCategory').textContent = info.category;
  speak(info.meaning);
}

// ===== INIT =====
buildAlphabetGrid();
setStatus('', 'Camera off — click START to begin');

// Make canvas resize-safe
window.addEventListener('resize', () => {
  if (video.videoWidth) {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }
});

console.log('%c✋ SignBridge Loaded', 'color:#00ffcc;font-size:1.2rem;font-weight:bold');
console.log('%c21-landmark hand tracking active', 'color:#9898b8');
