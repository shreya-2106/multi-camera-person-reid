"use strict";

/* ============================================================
   Phase 15 — Authenticated Frontend
   ============================================================ */

const API = "/api";

const state = {
    user: null,
    currentSearch: null,
    cameraCount: 2,
};

/* ============================================================
   Helpers
   ============================================================ */

const $ = (id) => document.getElementById(id);

function show(id) {
    const element = $(id);
    if (element) {
        element.classList.remove("hidden");
    }
}

function hide(id) {
    const element = $(id);
    if (element) {
        element.classList.add("hidden");
    }
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function formatPercent(value) {
    const number = Number(value);
    return Number.isFinite(number)
        ? `${(number * 100).toFixed(1)}%`
        : "—";
}

function formatDate(value) {
    if (!value) {
        return "—";
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return value;
    }

    return date.toLocaleString();
}

function showStatus(message, type = "info") {
    // The authentication screen uses dedicated inline error elements rather
    // than the shared global status element.
    if (!$("auth-screen")?.classList.contains("hidden")) {
        const errorId = $("login-view")?.classList.contains("hidden")
            ? "register-error"
            : "login-error";
        const authError = $(errorId);

        if (authError) {
            authError.textContent = message;
            return;
        }
    }

    const status = $("global-status");
    if (status) {
        status.textContent = message;
        status.className = `status ${type}`;
        show("global-status");
        return;
    }

    const searchError = $("search-error");
    if (searchError && !$("page-search")?.classList.contains("hidden")) {
        searchError.textContent = message;
    }
}

function clearStatus() {
    const status = $("global-status");
    if (status) {
        status.textContent = "";
        hide("global-status");
    }

    ["login-error", "register-error", "search-error"].forEach((id) => {
        const element = $(id);
        if (element) {
            element.textContent = "";
        }
    });
}

async function apiFetch(path, options = {}) {
    const response = await fetch(`${API}${path}`, {
        credentials: "include",
        ...options,
    });

    let data = null;

    const contentType =
        response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
        data = await response.json();
    } else {
        data = await response.text();
    }

    if (!response.ok) {
        let message = "Request failed.";

        if (data && typeof data === "object") {
            if (typeof data.detail === "string") {
                message = data.detail;
            } else if (Array.isArray(data.detail)) {
                message = data.detail
                    .map((item) => item.msg || "Invalid request.")
                    .join(" ");
            }
        }

        throw new Error(message);
    }

    return data;
}

/* ============================================================
   Screen navigation
   ============================================================ */

function showAuthScreen() {
    show("auth-screen");
    hide("app-screen");
}

function showAppScreen() {
    hide("auth-screen");
    show("app-screen");
}

function showPage(pageName) {
    document.querySelectorAll(".page").forEach((page) => {
        page.classList.add("hidden");
    });

    const pageId = `page-${pageName}`;
    const page = $(pageId);

    if (page) {
        page.classList.remove("hidden");
    }

    document.querySelectorAll("[data-page]").forEach((button) => {
        button.classList.toggle(
            "active",
            button.dataset.page === pageName
        );
    });
}

/* ============================================================
   Authentication
   ============================================================ */

async function checkSession() {
    try {
        const response = await apiFetch("/auth/me");
        state.user = response.user;

        showAppScreen();
        updateUserDisplay();
        showPage("search");

        return true;
    } catch {
        state.user = null;
        showAuthScreen();
        return false;
    }
}

function updateUserDisplay() {
    if (!state.user) {
        return;
    }

    const userName = $("user-name");

    if (userName) {
        userName.textContent =
            state.user.name || state.user.email || "User";
    }
}

async function login(event) {
    event.preventDefault();

    const email = $("login-email").value.trim();
    const password = $("login-password").value;

    if (!email || !password) {
        showStatus(
            "Enter your email and password.",
            "error"
        );
        return;
    }

    try {
        showStatus("Signing in...", "info");

        await apiFetch("/auth/login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                email,
                password,
            }),
        });

        const response = await apiFetch("/auth/me");
        state.user = response.user;

        showAppScreen();
        updateUserDisplay();
        showPage("search");
        clearStatus();

        $("login-form").reset();
    } catch (error) {
        showStatus(error.message, "error");
    }
}

async function register(event) {
    event.preventDefault();

    const name = $("register-name").value.trim();
    const email = $("register-email").value.trim();
    const password = $("register-password").value;

    if (!name || !email || !password) {
        showStatus(
            "Complete all registration fields.",
            "error"
        );
        return;
    }

    try {
        showStatus("Creating account...", "info");

        await apiFetch("/auth/register", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
    name,
    email,
    password,
    confirm_password: $("register-confirm").value,
}),
        });

        const response = await apiFetch("/auth/me");
        state.user = response.user;

        showAppScreen();
        updateUserDisplay();
        showPage("search");
        clearStatus();

        $("register-form").reset();
    } catch (error) {
        showStatus(error.message, "error");
    }
}

async function logout() {
    try {
        await apiFetch("/auth/logout", {
            method: "POST",
        });
    } catch {
        // The local UI should still return to login if logout
        // fails because the session may already be invalid.
    }

    state.user = null;
    state.currentSearch = null;

    showAuthScreen();
}

/* ============================================================
   Authentication form switching
   ============================================================ */

function showLoginForm() {
    show("login-view");
    hide("register-view");
}

function showRegisterForm() {
    hide("login-view");
    show("register-view");
}

/* ============================================================
   Camera uploads
   ============================================================ */

function renderCameraInputs() {
    const container = $("camera-list");

    if (!container) {
        return;
    }

    container.innerHTML = "";

    for (let index = 1; index <= state.cameraCount; index += 1) {
        const cameraId = `CAM${String(index).padStart(2, "0")}`;

        const wrapper = document.createElement("div");
        wrapper.className = "camera-card";

        wrapper.innerHTML = `
            <div>
                <h4>${cameraId}</h4>
                <input
                    type="file"
                    id="camera-file-${index}"
                    name="videos"
                    accept=".mp4,.avi,.mov,.mkv,.m4v,video/*"
                >
            </div>

            <button
                type="button"
                class="btn danger remove-camera"
                data-camera-index="${index}"
                ${state.cameraCount <= 2 ? "disabled" : ""}
            >
                Remove
            </button>
        `;

        container.appendChild(wrapper);
    }

    container.querySelectorAll(".remove-camera").forEach((button) => {
        button.addEventListener("click", () => {
            if (state.cameraCount <= 2) {
                return;
            }

            state.cameraCount -= 1;
            renderCameraInputs();
        });
    });
}

function addCamera() {
    if (state.cameraCount >= 3) {
        showStatus(
            "A maximum of 3 cameras can be uploaded.",
            "error"
        );
        return;
    }

    state.cameraCount += 1;
    renderCameraInputs();
}

/* ============================================================
   Search submission
   ============================================================ */


async function createSearch(event) {
    event.preventDefault();

    const caseName =
        $("case-name").value.trim() || "Untitled Search";

    const formData = new FormData();

    formData.append("case_name", caseName);

    let selectedFiles = 0;

    for (
        let index = 1;
        index <= state.cameraCount;
        index += 1
    ) {
        const input = $(`camera-file-${index}`);

        if (!input || !input.files.length) {
            continue;
        }

        formData.append("videos", input.files[0]);
        selectedFiles += 1;
    }

    if (selectedFiles < 2) {
        showStatus(
            "Select at least two camera videos.",
            "error"
        );
        return;
    }

    try {
        showPage("processing");
        showStatus(
            "Uploading videos and creating search...",
            "info"
        );

        const search = await apiFetch("/searches", {
            method: "POST",
            body: formData,
        });

        state.currentSearch = search;

        renderSearchResult(search);

        showPage("results");

        showStatus(
            "Search completed successfully.",
            "success"
        );
    } catch (error) {
        showPage("search");

        showStatus(
            error.message,
            "error"
        );
    }
}

/* ============================================================
   Search results
   ============================================================ */
function renderSearchResult(search) {
    state.currentSearch = search;

    const stored =
        search.result ||
        search.result_json ||
        {};

    const result =
        stored.result ||
        stored;

    const identities =
        Array.isArray(result.global_identities)
            ? result.global_identities
            : [];

    const videos =
        Array.isArray(result.videos)
            ? result.videos
            : [];

    const candidates =
        Array.isArray(result.candidate_reports)
            ? result.candidate_reports
            : [];

    const acceptedMatches =
        identities.filter(
            (identity) =>
                identity.decision === "accept"
        );
    const summary = result.summary && typeof result.summary === "object"
        ? result.summary
        : null;
    const matchedPeople = summary && Array.isArray(summary.matched_people)
        ? summary.matched_people
        : [];

    const identityElement = $("identity-label");
    const matchBanner = $("match-banner");

    if (identityElement) {
        if (!summary) {
            identityElement.textContent = "Match summary unavailable";
        } else if (matchedPeople.length) {
            identityElement.textContent =
                matchedPeople.map((person) => `Global Person ${person.global_person_id}`).join(", ");
        } else {
            identityElement.textContent =
                "No cross-camera match";
        }
    }

    const confidenceElement = $("overall-confidence");
    const confidenceFill = $("confidence-fill");

    if (confidenceElement) {
        if (acceptedMatches.length && matchedPeople.length) {
            const score =
                acceptedMatches[0].score;

            const confidence =
                extractIdentityScore(score);

            confidenceElement.textContent =
                confidence !== null
                    ? formatPercent(confidence)
                    : "—";
        } else {
            confidenceElement.textContent =
                "—";
        }
    }

    if (confidenceFill) {
        const score = acceptedMatches[0]
            ? extractIdentityScore(acceptedMatches[0].score)
            : null;
        confidenceFill.style.width = score === null
            ? "0%"
            : `${Math.round(score * 100)}%`;
    }

    const completed = result.status === "completed";
    const failed = search.status === "error" || Boolean(result.error);
    const partial = videos.some((video) => video.status !== "completed");
    if (matchBanner) {
        matchBanner.className = "match-banner";
        if (failed) {
            matchBanner.classList.add("result-failed");
            matchBanner.textContent = result.error || "Analysis failed. No complete result is available.";
        } else if (!completed || partial) {
            matchBanner.classList.add("result-partial");
            matchBanner.textContent = "Analysis completed with partial camera evidence. Review camera status and warnings below.";
        } else if (!summary) {
            matchBanner.classList.add("result-partial");
            matchBanner.textContent = "Analysis result is available, but its match summary is missing. Reload the investigation or contact support.";
        } else if (matchedPeople.length) {
            matchBanner.classList.add("result-match");
            matchBanner.textContent = `${matchedPeople.length} unique matched ${matchedPeople.length === 1 ? "person" : "people"} across ${new Set(matchedPeople.flatMap((person) => person.camera_ids || [])).size} cameras.`;
        } else {
            matchBanner.classList.add("result-no-match");
            const rejectionCounts = {};
            candidates.filter((candidate) => candidate.status === "rejected")
                .forEach((candidate) => {
                    const reason = candidate.reason || "other validation";
                    rejectionCounts[reason] = (rejectionCounts[reason] || 0) + 1;
                });
            const rejectionSummary = Object.entries(rejectionCounts)
                .map(([reason, count]) => `${count} ${reason.replaceAll("_", " ")}`)
                .join(", ");
            matchBanner.textContent = rejectionSummary
                ? `No accepted cross-camera match. Candidate rejections: ${rejectionSummary}.`
                : "No accepted cross-camera match. Available evidence did not meet the configured validation criteria.";
        }
    }

    renderInvestigationSummary(summary);

    renderCameraResults(videos, search.id);

    renderEvidence(
        acceptedMatches.length
            ? acceptedMatches[0].evidence || {}
            : {}
    );

    renderTimeline(buildRealTimeline(summary, candidates));

    if (search.id !== undefined && search.id !== null) {
        void loadSearchEvidence(search.id);
    }

    const flagButton =
        $("flag-result");

    const flagged =
        Boolean(stored.flagged);

    if (flagButton) {
        flagButton.disabled = flagged;

        flagButton.textContent =
            flagged
                ? "Result Flagged"
                : "Flag Incorrect Result";
    }
}

function renderInvestigationSummary(summary) {
    const metrics = $("investigation-summary");
    const people = $("matched-person-list");
    if (!metrics || !people) return;
    metrics.replaceChildren();
    people.replaceChildren();

    const availability = summary?.metric_availability || {};
    const values = [
        ["Matched people", summary?.matched_people_count ?? "Unavailable"],
        ["Total tracks", summary?.total_tracks ?? "Unavailable"],
        ["Person detections", summary?.total_detections ?? "Unavailable"],
        ["Matched camera observations", summary?.camera_observations_count ?? "Unavailable"],
        ["Cameras processed", summary?.cameras_processed_count ?? "Unavailable"],
    ];
    values.forEach(([label, value]) => {
        const card = document.createElement("div");
        card.className = "summary-metric";
        const number = document.createElement("strong");
        number.textContent = typeof value === "number" ? value.toLocaleString() : String(value);
        const caption = document.createElement("span");
        caption.textContent = label;
        card.append(number, caption);
        const reason = availability[label === "Matched people" ? "matched_people_count"
            : label === "Total tracks" ? "total_tracks"
                : label === "Person detections" ? "total_detections"
                    : label === "Matched camera observations" ? "camera_observations_count"
                        : "cameras_processed_count"];
        if (typeof value !== "number" && reason) {
            const explanation = document.createElement("small");
            explanation.textContent = reason;
            card.appendChild(explanation);
        }
        metrics.appendChild(card);
    });

    const matchedPeople = Array.isArray(summary?.matched_people) ? summary.matched_people : [];
    if (!matchedPeople.length) {
        const empty = document.createElement("p");
        empty.className = "empty-state";
        empty.textContent = summary
            ? "No accepted cross-camera matches found. Total tracks are listed separately above."
            : "The investigation summary is unavailable; match counts cannot be confirmed.";
        people.appendChild(empty);
        return;
    }

    matchedPeople.forEach((person) => {
        const card = document.createElement("article");
        card.className = "matched-person-card";
        const heading = document.createElement("div");
        heading.className = "matched-person-heading";
        const title = document.createElement("h3");
        title.textContent = `Global Person ${person.global_person_id}`;
        const cameras = document.createElement("span");
        cameras.textContent = `Matched across ${person.distinct_camera_count} cameras · ${(person.camera_ids || []).join(", ")}`;
        heading.append(title, cameras);
        card.appendChild(heading);

        const table = document.createElement("table");
        table.className = "observation-table";
        const thead = document.createElement("thead");
        const header = document.createElement("tr");
        ["Camera", "Track", "First video time", "Last video time", "Duration", "Frames"].forEach((label) => {
            const cell = document.createElement("th");
            cell.scope = "col";
            cell.textContent = label;
            header.appendChild(cell);
        });
        thead.appendChild(header);
        const tbody = document.createElement("tbody");
        (person.observations || []).forEach((observation) => {
            const row = document.createElement("tr");
            const first = formatVideoTime(observation.first_time_seconds);
            const last = formatVideoTime(observation.last_time_seconds);
            const duration = Number.isFinite(observation.duration_seconds)
                ? `${observation.duration_seconds.toFixed(2)} s`
                : "Unavailable";
            const frames = Number.isInteger(observation.first_frame) && Number.isInteger(observation.last_frame)
                ? `${observation.first_frame}–${observation.last_frame}`
                : "Unavailable";
            [observation.camera_id, observation.track_id, first, last, duration, frames].forEach((value) => {
                const cell = document.createElement("td");
                cell.textContent = value === null || value === undefined || value === ""
                    ? "Capture time unavailable"
                    : String(value);
                row.appendChild(cell);
            });
            tbody.appendChild(row);
        });
        table.append(thead, tbody);
        card.appendChild(table);

        const evidence = (person.matching_evidence || []).find((item) => item.evidence || item.score);
        if (evidence) {
            const evidenceText = document.createElement("p");
            evidenceText.className = "match-evidence-caption";
            const similarity = evidence.evidence?.mean_similarity;
            const score = evidence.score?.total_score;
            const available = [];
            if (Number.isFinite(similarity)) available.push(`appearance ${formatPercent(similarity)}`);
            if (Number.isFinite(score)) available.push(`decision score ${formatPercent(score)}`);
            evidenceText.textContent = available.length ? `Accepted match evidence: ${available.join(" · ")}` : "Accepted match evidence is recorded.";
            card.appendChild(evidenceText);
        }
        people.appendChild(card);
    });
}

function formatVideoTime(seconds) {
    if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
        return "Capture time unavailable";
    }
    const totalMilliseconds = Math.round(seconds * 1000);
    const hours = Math.floor(totalMilliseconds / 3600000);
    const minutes = Math.floor((totalMilliseconds % 3600000) / 60000);
    const wholeSeconds = Math.floor((totalMilliseconds % 60000) / 1000);
    const milliseconds = totalMilliseconds % 1000;
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}.${String(milliseconds).padStart(3, "0")}`;
}


function extractIdentityScore(score) {
    if (!score || typeof score !== "object") {
        return null;
    }

    const value =
        score.total_score ??
        score.score ??
        score.final_score;

    if (
        typeof value !== "number" ||
        !Number.isFinite(value)
    ) {
        return null;
    }

    return Math.max(
        0,
        Math.min(1, value)
    );
}


function renderCameraResults(videos) {
    const body = $("result-cameras");

    if (!body) {
        return;
    }

    body.replaceChildren();

    if (!videos.length) {
        const empty = document.createElement("p");
        empty.className = "empty-state";
        empty.textContent = "No camera results available.";
        body.appendChild(empty);
        return;
    }

    videos.forEach((video) => {
        const card = document.createElement("article");
        card.className = "camera-evidence-card";
        const title = document.createElement("h3");
        title.textContent = video.camera_id || "Unknown camera";
        const status = document.createElement("span");
        status.className = `camera-status ${video.status === "completed" ? "complete" : "incomplete"}`;
        status.textContent = video.status || "unknown";
        const stats = document.createElement("p");
        stats.textContent = `${Number(video.frames_processed || 0).toLocaleString()} frames · ${Number(video.tracks_detected || 0).toLocaleString()} tracks · ${Number(video.crops_generated || 0).toLocaleString()} crops · ${Number(video.embeddings_generated || 0).toLocaleString()} embeddings`;
        card.append(title, status, stats);
        if (video.error) {
            const warning = document.createElement("p");
            warning.className = "camera-warning";
            warning.textContent = video.error;
            card.appendChild(warning);
        }
        const gallery = document.createElement("div");
        gallery.className = "crop-gallery";
        gallery.dataset.cameraId = video.camera_id || "";
        card.appendChild(gallery);
        body.appendChild(card);
    });
}


function renderEvidence(evidence) {
    const container = $("evidence-grid");
    if (!container) return;
    container.replaceChildren();
    const values = [
        ["Appearance similarity", formatEvidenceValue(evidence.mean_similarity)],
        ["Temporal validation", evidence.temporal_valid === true ? "Valid" : evidence.temporal_valid === false ? "Rejected" : "Not available"],
        ["Camera topology", evidence.topology_valid === true ? "Valid" : evidence.topology_valid === false ? "Rejected" : "Not available"],
        ["Travel time", Number.isFinite(evidence.travel_time_seconds) ? `${evidence.travel_time_seconds.toFixed(1)} s` : "Not available"],
        ["Comparisons", Number.isFinite(evidence.comparison_count) ? String(evidence.comparison_count) : "Not available"],
    ];
    values.forEach(([label, value]) => {
        const item = document.createElement("div");
        item.className = "evidence-item";
        const name = document.createElement("span");
        name.textContent = label;
        const detail = document.createElement("strong");
        detail.textContent = value;
        item.append(name, detail);
        container.appendChild(item);
    });
}


async function loadSearchEvidence(searchId) {
    try {
        const response = await apiFetch(`/searches/${searchId}/evidence`);
        if (String(state.currentSearch?.id) !== String(searchId)) return;
        const images = Array.isArray(response.images) ? response.images : [];
        const cameras = $("result-cameras");
        cameras?.querySelectorAll(".crop-gallery").forEach((gallery) => {
            const matching = images.filter((image) => image.camera_id === gallery.dataset.cameraId);
            if (!matching.length) {
                const empty = document.createElement("p");
                empty.className = "empty-state";
                empty.textContent = "No generated crop images are available for this camera.";
                gallery.appendChild(empty);
                return;
            }
            matching.forEach((image, index) => {
                const figure = document.createElement("figure");
                figure.className = "crop-evidence";
                const img = document.createElement("img");
                img.src = image.image_url;
                img.alt = `${image.camera_id}, local track ${image.track_id}, generated person crop ${index + 1}`;
                img.loading = "lazy";
                img.onerror = () => {
                    img.remove();
                    const unavailable = document.createElement("span");
                    unavailable.className = "image-unavailable";
                    unavailable.textContent = "Crop image unavailable";
                    figure.prepend(unavailable);
                };
                const caption = document.createElement("figcaption");
                caption.textContent = `Track ${image.track_id} · crop ${index + 1}`;
                figure.append(img, caption);
                gallery.appendChild(figure);
            });
        });
    } catch (error) {
        const galleries = $("result-cameras")?.querySelectorAll(".crop-gallery");
        galleries?.forEach((gallery) => {
            const message = document.createElement("p");
            message.className = "empty-state";
            message.textContent = "Evidence images could not be loaded.";
            gallery.appendChild(message);
        });
        console.warn("Unable to load search evidence images", error);
    }
}


function formatEvidenceValue(value) {
    if (
        typeof value !== "number" ||
        !Number.isFinite(value)
    ) {
        return "—";
    }

    return formatPercent(value);
}


function buildRealTimeline(summary, candidates) {
    const events = [];
    const matchedPeople = Array.isArray(summary?.matched_people) ? summary.matched_people : [];
    matchedPeople.forEach((person) => {
        (person.observations || []).forEach((observation) => {
            events.push({
                time: formatVideoTime(observation.first_time_seconds),
                camera: observation.camera_id || "Unknown camera",
                event: `Global Person ${person.global_person_id} · Track ${observation.track_id} · observed until ${formatVideoTime(observation.last_time_seconds)}`,
            });
        });
    });
    events.sort((left, right) => left.time.localeCompare(right.time));

    if (!events.length) {
        const rejectionCounts = {};
        (Array.isArray(candidates) ? candidates : [])
            .filter((candidate) => candidate.status === "rejected")
            .forEach((candidate) => {
                const reason = candidate.reason || "other validation";
                rejectionCounts[reason] = (rejectionCounts[reason] || 0) + 1;
            });
        Object.entries(rejectionCounts).forEach(([reason, count]) => {
            events.push({
                time: "",
                camera: "Candidate review",
                event: `${count} candidates rejected: ${reason.replaceAll("_", " ")}`,
            });
        });
        if (!events.length) events.push({ time: "", camera: "Timeline", event: "No accepted cross-camera observations are available." });
    }

    return events;
}


function renderTimeline(events) {
    const container = $("timeline");

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (!events.length) {
        container.innerHTML =
            `<div class="empty">No timeline events.</div>`;

        return;
    }

    events.forEach((event) => {
        const item =
            document.createElement("div");

        item.className =
            "timeline-item";

        item.innerHTML = `
            <time>
                ${escapeHtml(
                    event.time ||
                    event.timestamp ||
                    ""
                )}
            </time>

            <div>
                <strong>
                    ${escapeHtml(
                        event.camera ||
                        ""
                    )}
                </strong>

                <div>
                    ${escapeHtml(
                        event.event ||
                        ""
                    )}
                </div>
            </div>
        `;

        container.appendChild(item);
    });
}

/* ============================================================
   History
   ============================================================ */

async function loadHistory() {
    try {
        const response =
            await apiFetch("/searches");
        const searches = Array.isArray(response)
            ? response
            : response.searches || [];

        const container =
            $("history-list");

        if (!container) {
            return;
        }

        container.innerHTML = "";

        if (!searches.length) {
            container.innerHTML =
                `<div class="empty">
                    No searches have been created yet.
                </div>`;
            return;
        }

        searches.forEach((search) => {
            const item =
                document.createElement("div");

            item.className = "history-item";

            item.innerHTML = `
                <div>
                    <h4>
                        ${escapeHtml(
                            search.case_name
                        )}
                    </h4>

                    <p>
                        ${search.camera_count} cameras
                        · ${escapeHtml(
                            formatDate(search.created_at)
                        )}
                    </p>
                </div>

                <div class="actions">
                    <span class="badge ${
                        search.status === "error"
                            ? "danger"
                            : search.status === "processing"
                                ? "warning"
                                : "success"
                    }">
                        ${escapeHtml(search.status)}
                    </span>

                    <button
                        type="button"
                        class="btn view-search"
                        data-search-id="${search.id}"
                    >
                        View
                    </button>

                    <button
                        type="button"
                        class="btn danger delete-search"
                        data-search-id="${search.id}"
                    >
                        Delete
                    </button>
                </div>
            `;

            container.appendChild(item);
        });

        container
            .querySelectorAll(".view-search")
            .forEach((button) => {
                button.addEventListener(
                    "click",
                    () =>
                        openSearch(
                            button.dataset.searchId
                        )
                );
            });

        container
            .querySelectorAll(".delete-search")
            .forEach((button) => {
                button.addEventListener(
                    "click",
                    () =>
                        deleteSearch(
                            button.dataset.searchId
                        )
                );
            });
    } catch (error) {
        showStatus(
            error.message,
            "error"
        );
    }
}

async function openSearch(searchId) {
    try {
        showStatus(
            "Loading saved search...",
            "info"
        );

        const search =
            await apiFetch(
                `/searches/${searchId}`
            );

        renderSearchResult(search);
        showPage("results");

        clearStatus();
    } catch (error) {
        showStatus(
            error.message,
            "error"
        );
    }
}

async function deleteSearch(searchId) {
    const confirmed = window.confirm(
        "Delete this search, its uploaded recordings, and its saved result?"
    );

    if (!confirmed) {
        return;
    }

    try {
        await apiFetch(
            `/searches/${searchId}`,
            {
                method: "DELETE",
            }
        );

        if (
            state.currentSearch &&
            String(state.currentSearch.id) ===
                String(searchId)
        ) {
            state.currentSearch = null;
        }

        await loadHistory();

        showStatus(
            "Search and stored files deleted.",
            "success"
        );
    } catch (error) {
        showStatus(
            error.message,
            "error"
        );
    }
}

/* ============================================================
   Flagging
   ============================================================ */

async function flagCurrentResult() {
    if (!state.currentSearch) {
        return;
    }

    const searchId =
        state.currentSearch.id;

    try {
        await apiFetch(
            `/searches/${searchId}/flag`,
            {
                method: "POST",
            }
        );

        if (
            state.currentSearch.result
        ) {
            state.currentSearch.result.flagged =
                true;
        }

        const button =
            $("flag-result");

        if (button) {
            button.disabled = true;
            button.textContent =
                "Result Flagged";
        }

        showStatus(
            "Result flagged successfully.",
            "success"
        );
    } catch (error) {
        showStatus(
            error.message,
            "error"
        );
    }
}

/* ============================================================
   Export
   ============================================================ */

function exportCurrentResult() {
    if (!state.currentSearch) {
        showStatus(
            "No result is currently open.",
            "error"
        );
        return;
    }

    const payload = {
        search: state.currentSearch,
        exported_at: new Date().toISOString(),
    };

    const blob = new Blob(
        [JSON.stringify(payload, null, 2)],
        {
            type: "application/json",
        }
    );

    const url =
        URL.createObjectURL(blob);

    const link =
        document.createElement("a");

    link.href = url;

    link.download =
        `search_${state.currentSearch.id}_report.json`;

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);

    showStatus(
        "Report exported.",
        "success"
    );
}

/* ============================================================
   Profile
   ============================================================ */

async function loadProfile() {
    try {
        const profile =
            await apiFetch("/profile");

        const details = $("profile-details");
        if (details) {
            const fields = [
                ["Name", profile.name],
                ["Email", profile.email],
                ["Role", profile.role],
                ["Created", formatDate(profile.created_at)],
                ["Last login", formatDate(profile.last_login_at)],
            ];

            details.innerHTML = fields.map(([label, value]) => `
                <div class="profile-detail">
                    <span>${escapeHtml(label)}</span>
                    <strong>${escapeHtml(value || "—")}</strong>
                </div>
            `).join("");
        }
    } catch (error) {
        showStatus(
            error.message,
            "error"
        );
    }
}

async function changePassword(event) {
    event.preventDefault();

    const currentPassword =
        $("current-password").value;

    const newPassword =
        $("new-password").value;

    const confirmPassword =
        $("confirm-new-password").value;

    try {
        await apiFetch(
            "/profile/password",
            {
                method: "PATCH",
                headers: {
                    "Content-Type":
                        "application/json",
                },
                body: JSON.stringify({
                    current_password:
                        currentPassword,
                    new_password:
                        newPassword,
                    confirm_password:
                        confirmPassword,
                }),
            }
        );

        $("password-form").reset();

        showStatus(
            "Password changed successfully.",
            "success"
        );

        await loadAuditLog();
    } catch (error) {
        showStatus(
            error.message,
            "error"
        );
    }
}

/* ============================================================
   Audit log
   ============================================================ */

async function loadAuditLog() {
    try {
        const response =
            await apiFetch("/audit-log");
        const entries = Array.isArray(response)
            ? response
            : response.entries || [];

        const body = $("audit-list");

        if (!body) {
            return;
        }

        body.innerHTML = "";

        if (!entries.length) {
            body.innerHTML = `<div class="empty">No audit entries available.</div>`;
            return;
        }

        entries.forEach((entry) => {
            const item = document.createElement("div");
            item.className = "profile-detail";
            item.innerHTML = `
                <span>${escapeHtml(formatDate(entry.created_at))}</span>
                <strong>${escapeHtml(entry.action)}</strong>
                <span>${escapeHtml(entry.detail || "")}</span>
            `;
            body.appendChild(item);
        });
    } catch (error) {
        showStatus(
            error.message,
            "error"
        );
    }
}

/* ============================================================
   Event binding
   ============================================================ */

function bindEvents() {
    const loginForm =
        $("login-form");

    if (loginForm) {
        loginForm.addEventListener(
            "submit",
            login
        );
    }

    const registerForm =
        $("register-form");

    if (registerForm) {
        registerForm.addEventListener(
            "submit",
            register
        );
    }

    const passwordForm =
        $("password-form");

    if (passwordForm) {
        passwordForm.addEventListener(
            "submit",
            changePassword
        );
    }

    const addCameraButton =
        $("add-camera");

    if (addCameraButton) {
        addCameraButton.addEventListener(
            "click",
            addCamera
        );
    }

    const searchForm =
        $("search-form");

    if (searchForm) {
        searchForm.addEventListener(
            "submit",
            createSearch
        );
    }

    const logoutButton =
        $("logout-button");

    if (logoutButton) {
        logoutButton.addEventListener(
            "click",
            logout
        );
    }

    const flagButton =
        $("flag-result");

    if (flagButton) {
        flagButton.addEventListener(
            "click",
            flagCurrentResult
        );
    }

    const exportButton =
        $("export-result");

    if (exportButton) {
        exportButton.addEventListener(
            "click",
            exportCurrentResult
        );
    }

    const loginSwitch =
        $("show-login");

    if (loginSwitch) {
        loginSwitch.addEventListener(
            "click",
            showLoginForm
        );
    }

    const registerSwitch =
        $("show-register");

    if (registerSwitch) {
        registerSwitch.addEventListener(
            "click",
            showRegisterForm
        );
    }

    document
        .querySelectorAll("[data-page]")
        .forEach((button) => {
            button.addEventListener(
                "click",
                async () => {
                    const page =
                        button.dataset.page;

                    showPage(page);

                    if (page === "history") {
                        await loadHistory();
                    }

                    if (page === "profile") {
                        await loadProfile();
                        await loadAuditLog();
                    }
                }
            );
        });
}

/* ============================================================
   Initialization
   ============================================================ */

async function initialize() {
    bindEvents();
    renderCameraInputs();

    const authenticated =
        await checkSession();

    if (!authenticated) {
        showLoginForm();
    }
}

initialize();
