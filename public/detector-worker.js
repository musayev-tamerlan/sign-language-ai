// The worker keeps camera inference off the UI thread.
let detector, session, labels, acceptance, frames = [], frameNumber = 0, smoothedScores;
function feature(landmarks) {
  const hands = [0, 1].map(index => (landmarks[index] || []).map(({x,y,z}) => [x,y,z]));
  while (hands.length < 2) hands.push(Array.from({length:21}, () => [0,0,0]));
  const present = hands.map(hand => hand.some(point => point.some(value => value !== 0)));
  const sorted = hands.map((hand, index) => ({hand, present: present[index]})).sort((a,b) => (a.present ? a.hand[0][0] : Infinity) - (b.present ? b.hand[0][0] : Infinity));
  return sorted.flatMap(({hand, present: isPresent}) => {
    if (!isPresent) return new Array(66).fill(0);
    const wrist = hand[0], distances = hand.slice(1).map(p => Math.hypot(p[0]-wrist[0],p[1]-wrist[1])).sort((a,b)=>a-b);
    const scale = distances[Math.floor(distances.length / 2)] || 1;
    return hand.flatMap(p => p.map((value, axis) => (value - wrist[axis]) / scale)).concat(wrist);
  });
}
function softmax(logits) { const max=Math.max(...logits), values=logits.map(x=>Math.exp(x-max)), sum=values.reduce((a,b)=>a+b,0); return values.map(x=>x/sum); }
async function recognize() {
  // Live signing is often held still. Start after two seconds and resample the
  // collected movement to the 64-frame shape used during training.
  if (frames.length < 24 || frameNumber % 4) return undefined;
  const sampled = Array.from({length: 64}, (_, index) => frames[Math.round(index * (frames.length - 1) / 63)]);
  if (sampled.filter(frame => frame.some(value => value !== 0)).length < 48) return null;
  const input = new ort.Tensor('float32', Float32Array.from(sampled.flat()), [1,64,132]);
  const logits = (await session.run({landmarks: input})).logits.data, scores = softmax(Array.from(logits));
  smoothedScores = smoothedScores ? scores.map((score, index) => smoothedScores[index] * .65 + score * .35) : scores;
  const ordered = smoothedScores.map((score,index)=>({score,index})).sort((a,b)=>b.score-a.score);
  return {label: labels[ordered[0].index], confidence: ordered[0].score, runnerUp: ordered[1].score};
}
self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      const vision = await import('/vendor/vision/vision_bundle.mjs');
      self.ort = await import('/vendor/ort/ort.wasm.min.mjs'); ort.env.wasm.wasmPaths = '/vendor/ort/'; ort.env.wasm.numThreads = 1;
      const [metadata, fileset] = await Promise.all([fetch('/assets/azsl-gru-v3.labels.json').then(r => r.json()), vision.FilesetResolver.forVisionTasks('/vendor/vision/wasm')]);
      labels = metadata.labels; acceptance = metadata.acceptance.selected;
      session = await ort.InferenceSession.create('/assets/azsl-gru-v3.onnx', {executionProviders:['wasm']});
      detector = await vision.HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: '/assets/hand_landmarker.task', delegate: 'CPU' },
        runningMode: 'VIDEO', numHands: 2
      });
      self.postMessage({ type: 'ready', acceptance });
    } else if (data.type === 'frame') {
      try {
        const result = detector.detectForVideo(data.bitmap, data.timestamp);
        frames.push(feature(result.landmarks)); if (frames.length > 64) frames.shift(); frameNumber++;
        const recognition = await recognize(), message = { type: 'result', hands: result.landmarks.length, frames: frames.length };
        if (recognition !== undefined) message.recognition = recognition;
        self.postMessage(message);
      } finally { data.bitmap.close(); }
    }
  } catch (error) { console.error('Recognition worker error:', error); self.postMessage({ type: 'error', message: error?.message || 'Worker error' }); }
};
