/* ═══════════════════════════════════════════════════════════
   main.js — AI-Powered Agile Coaching System
   Handles: Task Priority API calls, Chart.js rendering,
            UI interactions
   ═══════════════════════════════════════════════════════════ */

// ── Task Priority Prediction ──────────────────────────────────
async function predictTaskPriority() {
  const btn = document.getElementById("predict-task-btn");
  const resultDiv = document.getElementById("task-result");

  // Gather form values
  const complexity   = document.getElementById("complexity").value;
  const deadline     = document.getElementById("deadline_days").value;
  const bizValue     = document.getElementById("business_value").value;
  const dependencies = document.getElementById("dependencies").value;

  if (!complexity || !deadline || !bizValue || !dependencies) {
    showAlert("Please fill in all fields.", "warning");
    return;
  }

  // Show loading state
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span> Predicting…';
  resultDiv.innerHTML = "";

  try {
    const response = await fetch("/api/task_priority", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({
        complexity  : parseFloat(complexity),
        deadline_days   : parseFloat(deadline),
        business_value  : parseFloat(bizValue),
        dependencies: parseFloat(dependencies),
      }),
    });

    const data = await response.json();

    if (data.error) {
      resultDiv.innerHTML = `
        <div class="result-card danger fade-up">
          <div class="result-label">Error</div>
          <p class="text-red mono">${data.error}</p>
        </div>`;
      return;
    }

    const colorMap = { danger: "#ef4444", warning: "#f59e0b", success: "#22c55e" };
    const badgeMap = { danger: "badge-red", warning: "badge-amber", success: "badge-green" };
    const iconMap  = { danger: "🔴", warning: "🟡", success: "🟢" };

    resultDiv.innerHTML = `
      <div class="result-card ${data.color} fade-up">
        <div class="result-label">Predicted Priority</div>
        <div class="result-value" style="color:${colorMap[data.color]}">
          ${iconMap[data.color]} ${data.priority}
        </div>
        <div class="mt-3">
          <span class="badge-teal">Confidence: ${data.confidence}</span>
        </div>
        <div class="divider"></div>
        <div class="card-title">Probability Breakdown</div>
        ${buildProbBars(data.probabilities)}
      </div>`;
  } catch (err) {
    resultDiv.innerHTML = `<div class="result-card danger fade-up"><p class="text-red">Network error: ${err.message}</p></div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = "🔮 Predict Priority";
  }
}

function buildProbBars(probs) {
  const colors = { High: "#ef4444", Medium: "#f59e0b", Low: "#22c55e" };
  return Object.entries(probs).map(([label, pct]) => {
    const val = parseFloat(pct);
    return `
      <div class="mb-2">
        <div class="d-flex justify-content-between mb-1">
          <span class="mono" style="font-size:12px">${label}</span>
          <span class="mono text-teal" style="font-size:12px">${pct}</span>
        </div>
        <div class="progress-bar-wrap">
          <div class="progress-bar-fill" style="width:${val}%;background:${colors[label]}"></div>
        </div>
      </div>`;
  }).join("");
}

// ── Alert helper ──────────────────────────────────────────────
function showAlert(msg, type = "info") {
  const colorMap = { info: "teal", warning: "amber", danger: "red", success: "green" };
  const div = document.createElement("div");
  div.className = `result-card ${type === "warning" ? "warning" : "info"} fade-up`;
  div.style.cssText = "position:fixed;top:20px;right:20px;z-index:9999;max-width:320px";
  div.innerHTML = `<p>${msg}</p>`;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), 3500);
}

// ── Charts Page ───────────────────────────────────────────────
async function loadCharts() {
  const canvas = document.getElementById("velocity-chart");
  if (!canvas) return;

  try {
    const res  = await fetch("/api/chart_data");
    const data = await res.json();

    // 1. Velocity Trend (Line)
    new Chart(document.getElementById("velocity-chart"), {
      type: "line",
      data: {
        labels  : data.velocity_trend.labels,
        datasets: [{
          label          : "Team Velocity",
          data           : data.velocity_trend.data,
          borderColor    : "#00d4c8",
          backgroundColor: "rgba(0,212,200,.08)",
          borderWidth    : 2.5,
          pointBackgroundColor: "#00d4c8",
          pointRadius    : 5,
          tension        : 0.35,
          fill           : true,
        }],
      },
      options: chartOptions("Velocity (Story Points)"),
    });

    // 2. Task Status (Doughnut)
    new Chart(document.getElementById("task-status-chart"), {
      type: "doughnut",
      data: {
        labels  : data.task_status.labels,
        datasets: [{
          data           : data.task_status.data,
          backgroundColor: data.task_status.colors,
          borderColor    : "#0a0f1e",
          borderWidth    : 3,
          hoverOffset    : 6,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: "#e2e8f0", font: { family: "JetBrains Mono", size: 12 } } },
        },
        cutout: "68%",
      },
    });

    // 3. Bug Trend (Bar)
    new Chart(document.getElementById("bug-chart"), {
      type: "bar",
      data: {
        labels  : data.bug_trend.labels,
        datasets: [{
          label          : "Open Bugs",
          data           : data.bug_trend.data,
          backgroundColor: "rgba(239,68,68,.5)",
          borderColor    : "#ef4444",
          borderWidth    : 1.5,
          borderRadius   : 5,
        }],
      },
      options: chartOptions("Bug Count"),
    });

    // 4. Sentiment Pie
    new Chart(document.getElementById("sentiment-chart"), {
      type: "pie",
      data: {
        labels  : data.sentiment_dist.labels,
        datasets: [{
          data           : data.sentiment_dist.data,
          backgroundColor: data.sentiment_dist.colors,
          borderColor    : "#0a0f1e",
          borderWidth    : 3,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: "#e2e8f0", font: { family: "JetBrains Mono", size: 12 } } },
        },
      },
    });

  } catch (e) {
    console.error("Chart load error:", e);
  }
}

function chartOptions(yLabel = "") {
  return {
    responsive: true,
    plugins: {
      legend: { labels: { color: "#e2e8f0", font: { family: "JetBrains Mono", size: 12 } } },
    },
    scales: {
      x: {
        ticks : { color: "#64748b", font: { family: "JetBrains Mono", size: 11 } },
        grid  : { color: "rgba(255,255,255,.05)" },
      },
      y: {
        ticks : { color: "#64748b", font: { family: "JetBrains Mono", size: 11 } },
        grid  : { color: "rgba(255,255,255,.05)" },
        title : { display: !!yLabel, text: yLabel, color: "#64748b", font: { size: 11 } },
      },
    },
  };
}

// ── Standup character count ────────────────────────────────────
function updateCharCount() {
  const ta   = document.getElementById("team_updates");
  const cnt  = document.getElementById("char-count");
  if (ta && cnt) cnt.textContent = ta.value.length;
}

// ── Init ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadCharts();

  const ta = document.getElementById("team_updates");
  if (ta) ta.addEventListener("input", updateCharCount);

  // Highlight active nav link
  const path  = window.location.pathname;
  document.querySelectorAll(".nav-link-item").forEach(link => {
    if (link.getAttribute("href") === path) link.classList.add("active");
  });
});
