const express = require("express");
const path = require("path");

const app = express();

const PORT = process.env.PORT || 3000;
const HOST = "0.0.0.0";

app.use(
  "/node_modules",
  express.static(
    path.join(__dirname, "node_modules")
  )
);

app.use(
  express.static(
    path.join(__dirname, "public")
  )
);

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    service: "sign-language-ai"
  });
});

app.listen(PORT, HOST, () => {
  // Только один startup log.
  console.log(`Server started on port ${PORT}`);
});