const $ = (id) => document.getElementById(id);
const video = $('camera');
let dictionaryEntries = [], activeSign = -1;
async function loadDictionary() {
  try {
    const [metadata, references] = await Promise.all([fetch('/assets/azsl-gru-v3.labels.json').then(response => response.json()), fetch('/assets/signs/manifest.json').then(response => response.json())]);
    dictionaryEntries = metadata.labels.map(label => ({ label, image: references.find(reference => reference.label === label)?.image }));
    $('word-total').textContent = `${metadata.labels.length} söz`;
    for (const [index, entry] of dictionaryEntries.entries()) { const item = document.createElement('button'); item.type = 'button'; item.textContent = entry.label; item.addEventListener('click', () => openSign(index)); $('word-list').append(item); }
  } catch { $('word-list').textContent = 'Lüğət yüklənə bilmədi.'; }
}
function openSign(index) { activeSign = (index + dictionaryEntries.length) % dictionaryEntries.length; const entry = dictionaryEntries[activeSign]; $('sign-title').textContent = entry.label; $('sign-image').src = entry.image; $('sign-image').alt = `${entry.label} işarəsinin nümunəsi`; if (!$('sign-modal').open) $('sign-modal').showModal(); }
function closeSign() { const modal = $('sign-modal'); if (modal.classList.contains('closing')) return; modal.classList.add('closing'); modal.addEventListener('animationend', () => { modal.classList.remove('closing'); modal.close(); }, { once: true }); }
$('close-sign').addEventListener('click', closeSign);
$('sign-modal').addEventListener('click', event => { if (event.target === event.currentTarget) closeSign(); });
$('previous-sign').addEventListener('click', () => openSign(activeSign - 1));
$('next-sign').addEventListener('click', () => openSign(activeSign + 1));
$('try-sign').addEventListener('click', () => { const entry = dictionaryEntries[activeSign]; $('practice-label').textContent = entry.label; $('practice-copy').textContent = 'Nümunəyə baxın, sonra jesti kamerada təkrarlayın.'; $('practice-image').src = entry.image; $('practice-image').alt = `${entry.label} işarəsinin nümunəsi`; $('practice-image').hidden = false; $('practice-card').dataset.empty = 'false'; closeSign(); showScreen('live'); $('workspace').scrollIntoView({ behavior: 'smooth', block: 'start' }); status(`${entry.label} işarəsini sınayın`, 'Nümunədəki hərəkəti göstərin, sonra kameranı yandırın.'); });
$('practice-card').addEventListener('click', () => { if (activeSign < 0) showScreen('dictionary'); else openSign(activeSign); });
function showScreen(name) { document.querySelectorAll('.app-screen').forEach(screen => screen.hidden = screen.id !== `${name}-screen`); document.querySelectorAll('.app-tab').forEach(tab => tab.classList.toggle('active', tab.dataset.screen === name)); }
if (document.querySelectorAll) document.querySelectorAll('.app-tab').forEach(tab => tab.addEventListener('click', () => showScreen(tab.dataset.screen)));
if (document.querySelectorAll) document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(tab => { const active = tab === button; tab.classList.toggle('active', active); tab.setAttribute('aria-selected', String(active)); });
  document.querySelectorAll('.tab-panel').forEach(panel => { panel.hidden = panel.id !== `${button.dataset.tab}-panel`; });
}));
loadDictionary();
let session = 0, worker, stream, timer, watchdog, busy = false, running = false;
// Defaults are replaced with validation-calibrated values when the model is ready.
let wordConfidence = 0.90, wordMargin = 0.20;
let wordCandidate = '', wordCount = 0, modelActive = false;
function setConfidence(value, confirmed = false) {
  const confidence = Math.max(0, Math.min(1, value || 0));
  $('confidence').hidden = !value;
  $('confidence-value').textContent = `${Math.round(confidence * 100)}%`;
  $('confidence-fill').style.width = `${Math.round(confidence * 100)}%`;
  document.body.dataset.confirmed = String(confirmed);
}
function showRecognition(prediction, hands) {
  const valid = hands > 0 && prediction && typeof prediction.label === 'string' && prediction.label.trim() &&
    Number.isFinite(prediction.confidence) && prediction.confidence >= wordConfidence && prediction.confidence <= 1 &&
    Number.isFinite(prediction.runnerUp) && prediction.runnerUp >= 0 &&
    prediction.confidence + prediction.runnerUp <= 1.000001 &&
    prediction.confidence - prediction.runnerUp >= wordMargin;
  setConfidence(prediction?.confidence);
  if (!prediction || hands === 0 || !Number.isFinite(prediction.confidence)) {
    wordCandidate = ''; wordCount = 0;
    status('İşarəni tanımaq mümkün olmadı', 'Əlləriniz tam görünsün və işarəni yenidən göstərin.');
    return;
  }
  if (!valid) {
    wordCandidate = ''; wordCount = 0;
    status(`Ehtimal edilən söz: ${prediction.label}`, `Etibarlılıq ${Math.round(prediction.confidence * 100)}%. Hələ təsdiqlənməyib — işarəni sabit saxlayın.`);
    return;
  }
  if (wordCandidate === prediction.label) wordCount++;
  else { wordCandidate = prediction.label; wordCount = 1; }
  if (wordCount >= 3) { setConfidence(prediction.confidence, true); status(prediction.label, 'Tanınan söz'); }
  else status(`Ehtimal edilən söz: ${prediction.label}`, `Etibarlılıq ${Math.round(prediction.confidence * 100)}%. Təsdiq üçün sabit saxlayın (${wordCount}/3).`);
}
let cancelInit, lastFrame = -1, candidate = -1, candidateSince = 0;
function status(title, hint, error = false) {
  if ($('result').textContent !== title) $('result').textContent = title;
  if ($('hint').textContent !== hint) $('hint').textContent = hint;
  document.body.dataset.error = String(error);
}
function stop(title = 'Kamera dayandırılıb', hint = 'Davam etmək üçün kameranı yenidən yandırın.') {
  session++;
  running = false;
  clearTimeout(timer); clearTimeout(watchdog);
  cancelInit?.(); cancelInit = undefined;
  worker?.terminate(); worker = undefined;
  stream?.getTracks().forEach((track) => track.stop()); stream = undefined;
  video.pause(); video.srcObject = null;
  busy = false; lastFrame = -1; candidate = -1; wordCandidate = ''; wordCount = 0; modelActive = false; setConfidence(0);
  $('viewfinder').dataset.active = 'false';
  $('placeholder').hidden = false; $('view-label').hidden = true;
  $('start').disabled = false; $('stop').disabled = true;
  $('start-label').textContent = 'Kameranı yandır';
  $('camera-state').textContent = 'Söndürülüb';
  status(title, hint);
}
function fail(title, hint) { stop(title, hint); document.body.dataset.error = 'true'; }
const errors = {
  NotAllowedError: ['Kameraya icazə verilməyib', 'Brauzerin sayt ayarlarında kameraya icazə verin və yenidən cəhd edin.'],
  NotFoundError: ['Kamera tapılmadı', 'Kameranın qoşulduğunu yoxlayın və yenidən cəhd edin.'],
  NotReadableError: ['Kamera əlçatan deyil', 'Kameradan istifadə edən digər tətbiqləri bağlayın.'],
  SecurityError: ['Kameraya giriş məhduddur', 'Saytı təhlükəsiz HTTPS bağlantısı ilə açın.']
};
function schedule(id) { timer = setTimeout(() => frame(id), 85); }
async function frame(id) {
  if (id !== session || !running) return;
  if (busy || video.readyState < 2 || video.currentTime === lastFrame) { schedule(id); return; }
  busy = true; lastFrame = video.currentTime;
  try {
    const bitmap = await createImageBitmap(video);
    if (id !== session) { bitmap.close(); return; }
    watchdog = setTimeout(() => fail('Emal dayandırıldı', 'Yenidən cəhd edin və ya brauzeri yeniləyin.'), 15000);
    worker.postMessage({ type: 'frame', bitmap, timestamp: performance.now() }, [bitmap]);
  } catch { if (id === session) fail('Görüntü emal olunmadı', 'Kameranı yenidən yandırın.'); }
}
$('start').addEventListener('click', async () => {
  if ($('start').disabled) return;
  if (!isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    fail('Kamera dəstəklənmir', 'Saytı HTTPS ilə, yenilənmiş brauzerdə açın.'); return;
  }
  if (!window.Worker || !window.createImageBitmap || !window.OffscreenCanvas) {
    fail('Brauzeri yeniləyin', 'Bu funksiya üçün yenilənmiş Chrome, Edge, Firefox və ya Safari lazımdır.'); return;
  }
  const id = ++session;
  $('start').disabled = true; $('stop').disabled = false;
  $('start-label').textContent = 'Hazırlanır…'; $('camera-state').textContent = 'Hazırlanır';
  status('Kameraya icazə verin', 'Brauzer sorğusunda kameraya girişə icazə verin.');
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 24, max: 30 }, facingMode: 'user' }, audio: false });
    if (id !== session) { acquired.getTracks().forEach(t => t.stop()); return; }
    stream = acquired;
    stream.getVideoTracks()[0].addEventListener('ended', () => { if (id === session) fail('Kamera bağlantısı kəsildi', 'Kameranı yoxlayın və yenidən cəhd edin.'); });
    video.srcObject = stream;
    await video.play();
    if (id !== session) return;
    $('viewfinder').dataset.active = 'true'; $('placeholder').hidden = true; $('view-label').hidden = false;
    status('Əl aşkarlama hazırlanır', 'İlk açılışda bu bir qədər vaxt apara bilər.');
    worker = new Worker('/detector-worker.js');
    const activeWorker = worker;
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('ModelTimeout')), 45000);
      cancelInit = () => { clearTimeout(timeout); reject(new Error('Cancelled')); };
      activeWorker.onerror = () => { clearTimeout(timeout); reject(new Error('ModelLoad')); };
      activeWorker.onmessage = ({ data }) => {
        if (data.type === 'ready') {
          if (Number.isFinite(data.acceptance?.confidence)) wordConfidence = data.acceptance.confidence;
          if (Number.isFinite(data.acceptance?.margin)) wordMargin = data.acceptance.margin;
          clearTimeout(timeout); resolve();
        }
        if (data.type === 'error') { clearTimeout(timeout); reject(new Error(data.message || 'ModelLoad')); }
      };
      activeWorker.postMessage({ type: 'init' });
    });
    if (id !== session) return;
    cancelInit = undefined;
    worker.onerror = () => { if (id === session) fail('Emal xətası', 'Kameranı yenidən yandırın.'); };
    worker.onmessage = ({ data }) => {
      if (id !== session) return;
      clearTimeout(watchdog); busy = false;
      if (data.type === 'error') { fail('Emal xətası', 'Kameranı yenidən yandırın.'); return; }
      if (data.type !== 'result') return;
      // Only a real word classifier may supply this field; hand detection is not a word prediction.
      if (Object.hasOwn(data, 'recognition')) {
        modelActive = true;
        showRecognition(data.recognition, data.hands);
        schedule(id);
        return;
      }
      if (modelActive) { schedule(id); return; }
      const now = performance.now();
      if (candidate !== data.hands) { candidate = data.hands; candidateSince = now; }
      if (data.hands && data.frames < 24) {
        status('İşarə öyrənilir…', `Kadrlar toplanır: ${data.frames}/24. İşarəni sabit saxlayın.`);
      } else if (now - candidateSince >= 250) {
        status(candidate ? (candidate === 1 ? 'Bir əl aşkarlandı' : 'İki əl aşkarlandı') : 'Əllərinizi göstərin', candidate ? 'İşarəni sabit saxlayın, söz tanınır.' : 'Əllərinizin işıqlı və tam görünən yerdə olduğuna əmin olun.');
      }
      schedule(id);
    };
    running = true; $('camera-state').textContent = 'Aktiv'; $('start-label').textContent = 'Kamera aktivdir';
    status('Əllərinizi göstərin', 'Hər iki əlinizi kameranın qarşısına gətirin.');
    schedule(id);
  } catch (error) {
    if (id !== session) return;
    const message = errors[error.name] || ['Başlatmaq mümkün olmadı', 'Model yüklənmədi. Səhifəni yeniləyin və yenidən cəhd edin.'];
    console.error('Camera/model startup failed:', error.message || error);
    fail(...message);
  }
});
$('stop').addEventListener('click', () => stop());
document.addEventListener('visibilitychange', () => {
  if (document.hidden && $('stop').disabled === false) stop('Kamera dayandırılıb', 'Səhifədən ayrıldığınız üçün kamera söndürüldü. Davam etmək üçün yenidən yandırın.');
});
window.addEventListener('pagehide', () => stop());
