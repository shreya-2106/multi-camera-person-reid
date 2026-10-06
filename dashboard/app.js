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
    const status = $("global-status");

    if (!status) {
        return;
    }

    status.textContent = message;
    status.className = `status ${type}`;
    show("global-status");
}

function clearStatus() {
    hide("global-status");
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

function showPage(pageId) {
    document.querySelectorAll(".page").forEach((page) => {
        page.classList.add("hidden");
    });

    const page = $(pageId);

    if (page) {
        page.classList.remove("hidden");
    }

    document.querySelectorAll("[data-page]").forEach((button) => {
        button.classList.toggle(
            "active",
            button.dataset.page === pageId
        );
    });
}

/* ============================================================
   Authentication
   ============================================================ */

async function checkSession() {
    try {
        state.user = await apiFetch("/auth/me");

        showAppScreen();
        updateUserDisplay();
        showPage("search-page");

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

        state.user = await apiFetch("/auth/me");

        showAppScreen();
        updateUserDisplay();
        showPage("search-page");
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
            }),
        });

        state.user = await apiFetch("/auth/me");

        showAppScreen();
        updateUserDisplay();
        showPage("search-page");
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
    show("login-panel");
    hide("register-panel");
}

function showRegisterForm() {
    hide("login-panel");
    show("register-panel");
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
    if (state.cameraCount >= 8) {
        showStatus(
            "A maximum of 8 cameras can be uploaded.",
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
        showPage("processing-page");
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

        showPage("results-page");

        showStatus(
            "Search completed successfully.",
            "success"
        );
    } catch (error) {
        showPage("search-page");

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

    const result =
        search.result ||
        search.result_json ||
        {};

    const identity =
        result.identity_label ||
        "No global identity";

    const confidence =
        result.overall_confidence;

    const identityElement =
        $("result-identity");

    if (identityElement) {
        identityElement.textContent = identity;
    }

    const confidenceElement =
        $("result-confidence");

    if (confidenceElement) {
        confidenceElement.textContent =
            formatPercent(confidence);
    }

    const caseElement =
        $("result-case-name");

    if (caseElement) {
        caseElement.textContent =
            search.case_name || "Search";
    }

    renderCameraResults(result.cameras || []);
    renderEvidence(result.evidence || {});
    renderTimeline(result.timeline || []);

    const flagButton = $("flag-result");

    if (flagButton) {
        flagButton.disabled = Boolean(result.flagged);
        flagButton.textContent = result.flagged
            ? "Result Flagged"
            : "Flag Incorrect Result";
    }
}

function renderCameraResults(cameras) {
    const body = $("camera-results-body");

    if (!body) {
        return;
    }

    body.innerHTML = "";

    if (!cameras.length) {
        body.innerHTML = `
            <tr>
                <td colspan="3">No camera results available.</td>
            </tr>
        `;
        return;
    }

    cameras.forEach((camera) => {
        const row = document.createElement("tr");

        row.innerHTML = `
            <td>${escapeHtml(camera.camera)}</td>
            <td>
                <span class="badge success">
                    ${escapeHtml(camera.status)}
                </span>
            </td>
            <td>${formatPercent(camera.confidence)}</td>
        `;

        body.appendChild(row);
    });
}

function renderEvidence(evidence) {
    const appearance =
        $("evidence-appearance");

    const temporal =
        $("evidence-temporal");

    const topology =
        $("evidence-topology");

    if (appearance) {
        appearance.textContent =
            formatPercent(evidence.appearance_similarity);
    }

    if (temporal) {
        temporal.textContent =
            formatPercent(evidence.temporal_consistency);
    }

    if (topology) {
        topology.textContent =
            evidence.topology_valid === true
                ? "Valid"
                : evidence.topology_valid === false
                    ? "Invalid"
                    : "—";
    }
}

function renderTimeline(events) {
    const container = $("result-timeline");

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

        item.className = "timeline-item";

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
        const searches =
            await apiFetch("/searches");

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
        showPage("results-page");

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

        const name =
            $("profile-name");

        const email =
            $("profile-email");

        const role =
            $("profile-role");

        const created =
            $("profile-created");

        const lastLogin =
            $("profile-last-login");

        if (name) {
            name.textContent =
                profile.name || "—";
        }

        if (email) {
            email.textContent =
                profile.email || "—";
        }

        if (role) {
            role.textContent =
                profile.role || "—";
        }

        if (created) {
            created.textContent =
                formatDate(profile.created_at);
        }

        if (lastLogin) {
            lastLogin.textContent =
                formatDate(profile.last_login_at);
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
        $("confirm-password").value;

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
        const entries =
            await apiFetch("/audit-log");

        const body =
            $("audit-log-body");

        if (!body) {
            return;
        }

        body.innerHTML = "";

        if (!entries.length) {
            body.innerHTML = `
                <tr>
                    <td colspan="3">
                        No audit entries available.
                    </td>
                </tr>
            `;
            return;
        }

        entries.forEach((entry) => {
            const row =
                document.createElement("tr");

            row.innerHTML = `
                <td>
                    ${escapeHtml(
                        formatDate(
                            entry.created_at
                        )
                    )}
                </td>

                <td>
                    ${escapeHtml(
                        entry.action
                    )}
                </td>

                <td>
                    ${escapeHtml(
                        entry.detail
                    )}
                </td>
            `;

            body.appendChild(row);
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

                    if (
                        page ===
                        "history-page"
                    ) {
                        await loadHistory();
                    }

                    if (
                        page ===
                        "profile-page"
                    ) {
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
