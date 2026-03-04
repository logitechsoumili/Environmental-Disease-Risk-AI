let currentEnvironmentClass = null;
let latestReportData = null;

const labels = {
  stagnant_water: "Stagnant Water",
  garbage_dirty: "Garbage / Dirty Area",
  air_pollution: "Air Pollution",
  hygienic_environment: "Hygienic Environment"
};

const badgeColors = {
  stagnant_water: "blue",
  garbage_dirty: "red",
  air_pollution: "lightorange",
  hygienic_environment: "green"
};

function byId(id) {
  return document.getElementById(id);
}

function ensureToastWrap() {
  let wrap = byId("toastWrap");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.id = "toastWrap";
    wrap.className = "toast-wrap";
    document.body.appendChild(wrap);
  }
  return wrap;
}

function showToast(message, type = "info", duration = 3000) {
  const wrap = ensureToastWrap();
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  wrap.appendChild(toast);

  if (duration > 0) {
    setTimeout(() => {
      toast.remove();
    }, duration);
  }

  return toast;
}

function setActiveNav() {
  const page = document.body.dataset.page || "";
  const map = {
    home: "/",
    report: "/report",
    about: "/about"
  };
  const targetHref = map[page];
  if (!targetHref) return;

  document.querySelectorAll(".navlinks a").forEach((link) => {
    if (link.getAttribute("href") === targetHref) {
      link.classList.add("active");
    }
  });
}

function renderList(elementId, items) {
  const list = byId(elementId);
  if (!list) return;

  list.innerHTML = "";
  const safeItems = Array.isArray(items) ? items : [];

  if (!safeItems.length) {
    const li = document.createElement("li");
    li.textContent = "No data available.";
    list.appendChild(li);
    return;
  }

  safeItems.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}
async function analyze() {
  const input = byId("imageInput");
  // Clear previous outputs when a new image is selected
input.addEventListener("change", () => {
  if (preview) preview.src = "";
  if (predictionBadge) predictionBadge.innerText = "Pending";
  if (confidence) confidence.innerText = "0%";
  if (confidenceBar) confidenceBar.style.width = "0%";
  if (diseasesList) diseasesList.innerHTML = "";
  if (preventionList) preventionList.innerHTML = "";
  if (guidelinesList) guidelinesList.innerHTML = "";
  if (resultsSection) resultsSection.hidden = true;
  if (followupSection) followupSection.hidden = true;
});
  const analyzeBtn = byId("analyzeBtn");
  const preview = byId("preview");

  if (!input) return;

  const file = input.files[0];
  if (!file) {
    showToast("Please choose an image first.", "error");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";
  // CLEAR PREVIOUS RESULTS
if (preview) preview.src = "";
if (predictionBadge) predictionBadge.innerText = "Pending";
if (confidence) confidence.innerText = "0%";
if (confidenceBar) confidenceBar.style.width = "0%";
if (diseasesList) diseasesList.innerHTML = "";
if (preventionList) preventionList.innerHTML = "";
if (guidelinesList) guidelinesList.innerHTML = "";
if (resultsSection) resultsSection.hidden = true;
if (followupSection) followupSection.hidden = true;

  const formData = new FormData();
  formData.append("image", file);

  let localPreviewUrl = null;
  if (preview) {
    localPreviewUrl = URL.createObjectURL(file);
    preview.src = localPreviewUrl;
  }

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      showToast(data.error || "Failed to analyze image.", "error");
      return;
    }

    currentEnvironmentClass = data.prediction;
    latestReportData = data;
    // UPDATE NEW RESULTS
if (preview && localPreviewUrl) preview.src = localPreviewUrl;

if (predictionBadge) predictionBadge.innerText = data.prediction || "Unknown";

if (confidence && confidenceBar) {
  const confPercent = Math.round(data.confidence || 0);
  confidence.innerText = confPercent + "%";
  confidenceBar.style.width = confPercent + "%";
}

if (diseasesList) diseasesList.innerHTML = (data.diseases || []).map(d => `<li>${d}</li>`).join("");
if (preventionList) preventionList.innerHTML = (data.prevention || []).map(p => `<li>${p}</li>`).join("");
if (guidelinesList) guidelinesList.innerHTML = (data.guidelines || []).map(g => `<li>${g}</li>`).join("");

// Show the results and follow-up sections
if (resultsSection) resultsSection.hidden = false;
if (followupSection) followupSection.hidden = false;

// Re-enable the button
analyzeBtn.disabled = false;
analyzeBtn.textContent = "Analyze Environment";

    const results = byId("results");
    const followup = byId("followupSection");
    if (results) results.hidden = false;
    if (followup) followup.hidden = false;

    if (preview) {
      preview.src = `/uploads/${data.image}?t=${Date.now()}`;
    }

    const badge = byId("predictionBadge");
    if (badge) {
      const label = labels[data.prediction] || data.prediction;
      const color = badgeColors[data.prediction] || "blue";
      badge.className = `badge ${color}`;
      badge.textContent = label;
    }

    const confidence = byId("confidence");
    const confidenceBar = byId("confidenceBar");

    const percent = data.confidence > 1
      ? data.confidence
      : Math.round(data.confidence * 100);

    if (confidence) confidence.textContent = `${percent}%`;
    if (confidenceBar) confidenceBar.style.width = `${percent}%`;

    renderList("diseases", data.diseases || []);
    renderList("prevention", data.preventive_measures || []);
    renderList("guidelines", data.health_guidelines || []);

    const answer = byId("answer");
    if (answer) {
      answer.textContent =
        data.rag_answer || "Analysis complete. Ask a follow-up question below.";
    }

    showToast("Analysis completed successfully.", "success");

  } catch (error) {
    console.error(error);
    showToast("Unexpected error during analysis.", "error");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Environment";

    if (localPreviewUrl) {
      setTimeout(() => URL.revokeObjectURL(localPreviewUrl), 1000);
    }
  }
}

  
async function ask() {
  const questionInput = byId("question");
  const answerBox = byId("answer");
  if (!questionInput || !answerBox) return;

  const question = questionInput.value.trim();
  if (!question) {
    showToast("Please enter a question.", "error");
    return;
  }

  if (!currentEnvironmentClass) {
    showToast("Analyze an image first.", "error");
    return;
  }

  byId("askBtn").disabled = true;
  byId("askBtn").textContent = "Thinking...";
  const loadingToast = showToast("Generating follow-up answer...", "info", 0);

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        environment_class: currentEnvironmentClass
      })
    });

    const data = await response.json();
    if (!response.ok) {
      answerBox.textContent = data.error || "Could not generate answer.";
      showToast(data.error || "Could not generate answer.", "error");
      return;
    }

    answerBox.textContent = data.answer || "No answer generated.";
    showToast("Answer generated.", "success");
  } catch (error) {
    answerBox.textContent = "An unexpected error occurred while fetching the answer.";
    showToast("An unexpected error occurred while fetching the answer.", "error");
  } finally {
    loadingToast.remove();
    byId("askBtn").disabled = false;
    byId("askBtn").textContent = "Ask";
  }
}

async function downloadReport() {
  if (!latestReportData) {
    showToast("Analyze an image first.", "error");
    return;
  }

  const payload = {
    prediction: labels[latestReportData.prediction] || latestReportData.prediction,
    confidence: latestReportData.confidence,
    diseases: latestReportData.diseases || [],
    preventive_measures: latestReportData.preventive_measures || [],
    health_guidelines: latestReportData.health_guidelines || [],
    image: latestReportData.image || ""
  };

  const btn = byId("downloadBtn");
  btn.disabled = true;
  btn.textContent = "Preparing...";
  const loadingToast = showToast("Preparing PDF report...", "info", 0);

  try {
    const response = await fetch("/download_report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      showToast("Failed to download report.", "error");
      return;
    }

    const blob = await response.blob();
    if (blob.size === 0) {
      showToast("Received an empty PDF file from server.", "error");
      return;
    }
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "environmental_health_report.pdf";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("Report downloaded.", "success");
  } catch (error) {
    showToast("An unexpected error occurred while downloading the report.", "error");
  } finally {
    loadingToast.remove();
    btn.disabled = false;
    btn.textContent = "Download Report";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setActiveNav();
  const analyzeBtn = byId("analyzeBtn");
  const askBtn = byId("askBtn");
  const downloadBtn = byId("downloadBtn");

  if (analyzeBtn) analyzeBtn.addEventListener("click", analyze);
  if (askBtn) askBtn.addEventListener("click", ask);
  if (downloadBtn) downloadBtn.addEventListener("click", downloadReport);
});
