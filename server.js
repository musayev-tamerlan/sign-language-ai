const express = require('express');
const path = require('node:path');
const fs = require('node:fs');
const app = express();
app.disable('x-powered-by');
app.use((_req, res, next) => {
  res.set({
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'no-referrer',
    'X-Frame-Options': 'DENY',
    'Permissions-Policy': 'camera=(self), microphone=(), geolocation=()',
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
  });
  next();
});
app.get('/api/health', (_req, res) => {
  const assets = ['hand_landmarker.task', 'azsl-gru-v3.onnx', 'azsl-gru-v3.labels.json'];
  const ready = assets.every(file => fs.existsSync(path.join(__dirname, 'public/assets', file)));
  res.set('Cache-Control', 'no-store').status(ready ? 200 : 503).json({ ok: ready, service: 'sign-language-ai', recognition: 'azsl-words-100' });
});
// Expose only the browser runtime, never the entire node_modules directory.
app.use('/vendor/vision', express.static(path.join(__dirname, 'node_modules/@mediapipe/tasks-vision'), { maxAge: '1d', index: false, dotfiles: 'deny' }));
app.use('/vendor/ort', express.static(path.join(__dirname, 'node_modules/onnxruntime-web/dist'), { maxAge: '1d', index: false, dotfiles: 'deny' }));
app.use(express.static(path.join(__dirname, 'public'), { maxAge: 0, dotfiles: 'deny' }));
app.use((_req, res) => res.status(404).type('text').send('Səhifə tapılmadı.'));
app.use((_error, _req, res, _next) => res.status(500).type('text').send('Server xətası. Yenidən cəhd edin.'));
if (require.main === module) {
  const server = app.listen(process.env.PORT || 3000, '0.0.0.0', () => console.log('Sign Language AI server started.'));
  const shutdown = () => { server.close(() => process.exit(0)); setTimeout(() => process.exit(1), 10000).unref(); };
  process.on('SIGTERM', shutdown); process.on('SIGINT', shutdown);
}
module.exports = app;
