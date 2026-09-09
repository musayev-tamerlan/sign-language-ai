const worker = new Worker('/detector-worker.js');
const timeout = setTimeout(() => { document.getElementById('result').textContent = 'FAIL: timeout'; worker.terminate(); },45000);
worker.onerror = e => { clearTimeout(timeout); document.getElementById('result').textContent = 'FAIL: ' + e.message; worker.terminate(); };
worker.onmessage = async ({data}) => {
  if (data.type === 'ready') {
    const canvas = document.createElement('canvas'); canvas.width=640; canvas.height=480;
    canvas.getContext('2d').fillRect(0,0,640,480);
    const bitmap = await createImageBitmap(canvas);
    worker.postMessage({type:'frame',bitmap,timestamp:performance.now()},[bitmap]);
  } else {
    clearTimeout(timeout);
    document.getElementById('result').textContent = data.type === 'result' && data.hands === 0 ? 'PASS: real MediaPipe worker initialized and processed a blank frame' : 'FAIL: '+JSON.stringify(data);
    worker.terminate();
  }
};
worker.postMessage({type:'init'});
