const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("errorBox");
const resultPanel = document.getElementById("resultPanel");

function safeValue(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }
    if (Array.isArray(value)) {
        return value.join(", ");
    }
    if (typeof value === "object") {
        return JSON.stringify(value);
    }
    return String(value);
}

async function checkBackend() {
    const dot = document.getElementById("backendDot");
    const status = document.getElementById("backendStatus");

    try {
        const response = await fetch("/api/health");
        if (!response.ok) {
            throw new Error("Backend unavailable");
        }

        const data = await response.json();

        if (data.status === "ok") {
            dot.classList.add("online");
            status.textContent = "Backend Online";
        } else {
            dot.classList.add("offline");
            status.textContent = "Backend Error";
        }
    } catch (error) {
        dot.classList.add("offline");
        status.textContent = "Backend Offline";
    }
}

async function askQuestion() {
    const question = questionInput.value.trim();

    if (!question) {
        showError("Please enter a question.");
        return;
    }

    hideError();
    loading.style.display = "block";
    askButton.disabled = true;
    resultPanel.style.display = "none";

    try {
        const response = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Unable to process the question.");
        }

        displayResult(data);
    } catch (error) {
        showError(error.message);
    } finally {
        loading.style.display = "none";
        askButton.disabled = false;
    }
}

function displayResult(data) {
    // backend field names aren't always consistent, so we fall back a lot here
    const evidence = data.evidence || {};

    document.getElementById("answer").textContent =
        safeValue(data.answer ?? data.result);

    document.getElementById("activityEvent").textContent =
        safeValue(data.activity_event ?? data.activity ?? data.event);

    document.getElementById("timestamps").textContent =
        safeValue(evidence.timestamps ?? evidence.timestamp ?? data.timestamps ?? data.timestamp);

    document.getElementById("sensorModality").textContent =
        safeValue(evidence.sensor_modality ?? evidence.modality ?? data.sensor_modality);

    document.getElementById("sensorChannels").textContent =
        safeValue(evidence.sensor_channels ?? evidence.channels ?? data.sensor_channels);

    document.getElementById("explanation").textContent =
        safeValue(data.explanation ?? data.reasoning);

    resultPanel.style.display = "block";
    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadTimeline() {
    const container = document.getElementById("timeline");

    try {
        const response = await fetch("/api/timeline");
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Timeline unavailable");
        }

        let timeline = data.timeline;

        if (!timeline) {
            container.innerHTML = '<div style="color:var(--muted)">No timeline available.</div>';
            return;
        }

        if (!Array.isArray(timeline)) {
            timeline = timeline.intervals || timeline.events || timeline.activities || [];
        }

        if (!timeline.length) {
            container.innerHTML = '<div style="color:var(--muted)">No activity intervals detected.</div>';
            return;
        }

        container.innerHTML = "";

        timeline.slice(0, 40).forEach(item => {
            const row = document.createElement("div");
            row.className = "timeline-item";

            const activity = item.activity ?? item.label ?? item.prediction ?? item.activity_event ?? "Activity";
            const start = item.start ?? item.start_time ?? item.start_sec ?? item.start_s ?? "?";
            const end = item.end ?? item.end_time ?? item.end_sec ?? item.end_s ?? "?";
            const confidence = item.confidence ?? item.probability ?? item.score;

            const confidenceText = confidence !== undefined
                ? "confidence " + (typeof confidence === "number" ? (confidence * 100).toFixed(1) + "%" : confidence)
                : "";

            row.innerHTML = `
                <div class="activity">${escapeHtml(String(activity))}</div>
                <div class="timeline-time">${escapeHtml(String(start))} → ${escapeHtml(String(end))}</div>
                <div class="confidence">${escapeHtml(confidenceText)}</div>
            `;

            container.appendChild(row);
        });

        if (timeline.length > 40) {
            const more = document.createElement("div");
            more.style.color = "var(--muted)";
            more.style.fontSize = "12px";
            more.style.marginTop = "8px";
            more.textContent = `Showing first 40 of ${timeline.length} intervals.`;
            container.appendChild(more);
        }
    } catch (error) {
        container.innerHTML = `<div style="color:#b91c1c;">Could not load timeline: ${escapeHtml(error.message)}</div>`;
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.style.display = "block";
}

function hideError() {
    errorBox.style.display = "none";
}

askButton.addEventListener("click", askQuestion);

questionInput.addEventListener("keydown", event => {
    if (event.key === "Enter") {
        askQuestion();
    }
});

document.querySelectorAll(".example").forEach(button => {
    button.addEventListener("click", () => {
        questionInput.value = button.textContent.trim();
        askQuestion();
    });
});

checkBackend();
loadTimeline();
