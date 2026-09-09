const fs = require('node:fs/promises');
const path = require('node:path');
const { createHash } = require('node:crypto');
const root = path.resolve(__dirname, '..');
const digest = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1';
const url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
async function main() {
  for (const file of ['azsl-gru-v3.onnx', 'azsl-gru-v3.labels.json', 'signs/manifest.json']) {
    try { await fs.access(path.join(root, 'public/assets', file)); } catch { throw new Error('Missing browser recognition asset: ' + file); }
  }
  const output = path.join(root, 'public/assets/hand_landmarker.task');
  let bytes;
  for (const file of [output, path.join(root, 'training/models/hand_landmarker.task')]) {
    try { const candidate = await fs.readFile(file); if (createHash('sha256').update(candidate).digest('hex') === digest) { bytes = candidate; break; } } catch {}
  }
  if (!bytes) {
    const response = await fetch(url, { signal: AbortSignal.timeout(120000) });
    if (!response.ok) throw new Error('Model download failed: ' + response.status);
    bytes = Buffer.from(await response.arrayBuffer());
  }
  if (createHash('sha256').update(bytes).digest('hex') !== digest) throw new Error('Model checksum mismatch');
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.writeFile(output + '.tmp', bytes);
  await fs.rename(output + '.tmp', output);
  console.log('Browser model ready (SHA-256 verified).');
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
