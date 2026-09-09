const { test } = require('node:test');
const assert = require('node:assert/strict');
const app = require('../server');
test('production routes, assets and privacy boundaries', async () => {
  const server = app.listen(0, '127.0.0.1');
  await new Promise(r => server.once('listening',r));
  const base = 'http://127.0.0.1:' + server.address().port;
  try {
    const page = await fetch(base); assert.equal(page.status,200);
    assert.match(await page.text(), /lang="az"/);
    assert.match(page.headers.get('content-security-policy'), /worker-src 'self'/);
    assert.equal(page.headers.get('x-powered-by'), null);
    assert.equal(page.headers.get('set-cookie'), null);
    const health = await fetch(base+'/api/health'); assert.equal(health.status,200);
    assert.equal((await health.json()).recognition,'azsl-words-100');
    for (const route of ['/app.js','/detector-worker.js','/styles.css','/vendor/vision/vision_bundle.mjs','/vendor/vision/wasm/vision_wasm_internal.wasm','/vendor/ort/ort.wasm.min.mjs','/vendor/ort/ort-wasm-simd-threaded.wasm','/assets/hand_landmarker.task','/assets/azsl-gru-v3.onnx','/assets/azsl-gru-v3.labels.json','/assets/signs/manifest.json','/assets/signs/000.jpg']) {
      const response = await fetch(base+route); assert.equal(response.status,200,route); await response.arrayBuffer();
    }
    for (const route of ['/node_modules/express/package.json','/training/requirements.txt','/.env','/AzSLD_Words_100.zip']) assert.equal((await fetch(base+route)).status,404,route);
  } finally { await new Promise(r => server.close(r)); }
});
