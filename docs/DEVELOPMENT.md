# Development

## Local run

From project root:

```powershell
npm install
npm start
```

Open:

```text
http://localhost:3000
```

## Development mode

If supported by current package.json:

```powershell
npm run dev
```

## Expected structure

```text
sign-language-ai/
├── public/
│   ├── index.html
│   └── app.js
├── training/
│   ├── extract_landmarks.py
│   ├── train.py
│   └── evaluate.py
├── models/
├── docs/
├── server.js
├── package.json
├── .gitignore
└── AzSLD_Words_100.zip   # local only, NEVER commit
```

## Git

Before committing:

```powershell
git status
```

Make sure the dataset ZIP is ignored.

Then:

```powershell
git add .
git commit -m "..."
git push
```

## Render

Current deployment target: Render.

Official Render behavior:
- connected Git branch can auto-deploy on push
- Node Web Service uses build/start commands
- service must listen on `0.0.0.0`
- use `process.env.PORT`

Suggested:

```text
Build Command:
npm install

Start Command:
npm start
```

## Dataset

Local dataset:

```text
AzSLD_Words_100.zip
```

Approx size:

```text
1.1 GB
```

Do not commit it.

After extraction, inspect before writing training code.

## Logs

Browser:
- no per-frame console logs

Server:
- startup log is fine
- avoid request logging unless debugging

When debugging, temporarily enable logs and remove them afterward.
