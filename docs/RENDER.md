# Render deployment

The UI is Azerbaijani-only. This release detects hands; it does not translate signs
into words until an evaluated AzSL classifier is integrated.

## Local

~~~powershell
npm ci
npm run build
npm test
npm start
~~~

Open http://localhost:3000. Node 22.x is declared in package.json.
The build prepares the 7.8 MB MediaPipe hand detector, verifies its SHA-256,
and downloads the pinned official version if no valid local copy exists.
No Python environment, dataset or training run is required by the deployed app.

## Render Web Service

Use the repository root. For an existing service set:

- Runtime: Node
- Build command: npm ci --omit=dev && npm run build
- Start command: npm start
- Environment: NODE_ENV=production
- Health check: /api/health

Alternatively use render.yaml as a Blueprint for a new service. The Blueprint selects
the free plan. The server listens on 0.0.0.0 and Render's PORT. Use the HTTPS Render URL
for camera access. No credentials are required. Do not create a duplicate service if
one is already connected. This change prepares deployment; it does not publish or push.

Official configuration: https://render.com/docs/blueprint-spec
Health checks: https://render.com/docs/health-checks

## Runtime behavior

Hand detection runs in a CPU Web Worker, with at most one frame in flight and about
12 requests/sec maximum. The UI changes hand status only after 250 ms of consistency.
Stop, tab hiding and page exit release camera tracks and terminate the worker.
Late camera permission responses are discarded safely after cancellation.
Camera errors have Azerbaijani guidance. Model initialization and processing timeouts
allow retry. Browser features are checked before starting; unsupported browsers get
an upgrade message. Actual phone/camera compatibility still requires device testing.

Runtime and WASM are served locally under /vendor/vision; the model is served locally
under /assets. No CDN is needed during browser use. Security headers restrict resources
to the same origin, disable framing and microphone access. Raw video is not uploaded.
The old general /node_modules route is removed. Health returns 503 if the model is absent.

## Checks

npm test covers server routes/assets and camera lifecycle with simulated permissions.
For a real worker smoke test without a webcam:

~~~powershell
node tests/browser-server.cjs
~~~

Open http://localhost:3101/smoke. It loads the actual model and processes a blank frame.
These test routes are not part of the production server.
Before launch, verify camera permission, both hands, Stop/retry and mobile performance
on target devices. Word recognition remains a separate unfinished ML milestone.
