const queryInput = document.querySelector("#query");
const topKInput = document.querySelector("#topK");
const runBtn = document.querySelector("#runBtn");
const sampleBtn = document.querySelector("#sampleBtn");
const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");
const metaEl = document.querySelector("#meta");
const jsonOutput = document.querySelector("#jsonOutput");

const samples = [
  "We are a small enterprise manufacturing 33 Grade Ordinary Portland Cement. Which BIS standard covers the chemical and physical requirements for our product?",
  "Looking for the standard detailing corrugated and semi-corrugated asbestos cement sheets used for roofing and cladding.",
  "Which standard applies to masonry cement used for general purposes where mortars for masonry are required, but not intended for structural concrete?",
  "I need regulations for coarse and fine aggregates derived from natural sources intended for use in structural concrete."
];

let sampleIndex = 0;

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    statusEl.textContent = data.gemini_enabled ? "Gemini enabled" : "Local ranking";
    statusEl.className = "status ready";
  } catch (error) {
    statusEl.textContent = "Server issue";
    statusEl.className = "status error";
  }
}

function setBusy(isBusy) {
  runBtn.disabled = isBusy;
  sampleBtn.disabled = isBusy;
  topKInput.disabled = isBusy;
  runBtn.textContent = isBusy ? "Running" : "Recommend";
}

function renderResult(data) {
  resultsEl.replaceChildren();
  if (data.out_of_scope) {
    const item = document.createElement("li");
    item.textContent = data.message || "No relevant BIS building-material standard was found for this query.";
    resultsEl.appendChild(item);
    metaEl.textContent = `Out of scope · ${data.latency_seconds}s`;
    jsonOutput.textContent = JSON.stringify(data, null, 2);
    return;
  }
  data.retrieved_standards.forEach((standard) => {
    const item = document.createElement("li");
    item.textContent = standard;
    resultsEl.appendChild(item);
  });
  metaEl.textContent = `${data.mode} · ${data.latency_seconds}s`;
  jsonOutput.textContent = JSON.stringify(data, null, 2);
}

async function recommend() {
  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }

  setBusy(true);
  metaEl.textContent = "Retrieving standards";
  resultsEl.replaceChildren();
  jsonOutput.textContent = "Running LangGraph workflow...";

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: "UI-QUERY",
        query,
        top_k: Number(topKInput.value)
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Recommendation failed.");
    }

    renderResult(await response.json());
  } catch (error) {
    metaEl.textContent = "Request failed";
    jsonOutput.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

sampleBtn.addEventListener("click", () => {
  sampleIndex = (sampleIndex + 1) % samples.length;
  queryInput.value = samples[sampleIndex];
  queryInput.focus();
});

runBtn.addEventListener("click", recommend);

queryInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    recommend();
  }
});

loadHealth();
