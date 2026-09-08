const video = document.getElementById("camera");
const button = document.getElementById("start");
const result = document.getElementById("result");

let handLandmarker = null;
let lastVideoTime = -1;
let running = false;

async function initAI() {
  const vision = await import(
    "/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs"
  );

  const { HandLandmarker, FilesetResolver } = vision;

  const fileset = await FilesetResolver.forVisionTasks(
    "/node_modules/@mediapipe/tasks-vision/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: 1280,
      height: 720,
      facingMode: "user"
    },
    audio: false
  });

  video.srcObject = stream;

  await video.play();

  running = true;
  requestAnimationFrame(detectHands);
}

function detectHands() {
  if (!running || !handLandmarker) {
    requestAnimationFrame(detectHands);
    return;
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const detection = handLandmarker.detectForVideo(
      video,
      performance.now()
    );

    const hands = detection.landmarks.length;

    if (hands > 0) {
      result.textContent = `Hand detected • ${hands}`;
    } else {
      result.textContent = "Show your hand";
    }
  }

  requestAnimationFrame(detectHands);
}

button.addEventListener("click", async () => {
  button.disabled = true;

  try {
    result.textContent = "Loading AI...";

    await initAI();

    result.textContent = "Starting camera...";

    await startCamera();
  } catch {
    result.textContent = "Could not start camera";
    button.disabled = false;
  }
});