(() => {
  "use strict";

  const tokenFragment = new URLSearchParams(window.location.hash.slice(1));
  const suppliedToken = tokenFragment.get("token");
  if (suppliedToken) {
    sessionStorage.setItem("grip-lan-token", suppliedToken);
    history.replaceState({}, "", window.location.pathname + window.location.search);
  }

  const token = sessionStorage.getItem("grip-lan-token") || "";
  const state = { canEdit: null, logFloor: 0, comparisonLoaded: false };
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (token) headers.set("X-GRIP-Token", token);
    if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    const response = await fetch(path, { ...options, headers });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
    return payload;
  }

  function toast(message, error = false) {
    const item = document.createElement("div");
    item.className = `toast${error ? " error" : ""}`;
    item.textContent = message;
    $("#toast-region").append(item);
    window.setTimeout(() => item.remove(), 4200);
  }

  function setBadge(element, text, mode) {
    element.textContent = text;
    element.className = `badge ${mode}`;
  }

  function applyAccess(canEdit) {
    if (state.canEdit === canEdit) return;
    state.canEdit = canEdit;
    document.body.dataset.local = String(canEdit);
    $$(".host-only").forEach((element) => { element.hidden = !canEdit; });
    $("#remote-notice").hidden = canEdit;
    setBadge($("#access-badge"), canEdit ? "Host controls" : "LAN viewer", canEdit ? "local" : "viewer");
  }

  function renderState(payload) {
    applyAccess(Boolean(payload.can_edit));
    const pipeline = Boolean(payload.running.pipeline);
    const tensorboard = Boolean(payload.running.tensorboard);
    const pipelineJob = payload.jobs?.pipeline;
    const tensorboardJob = payload.jobs?.tensorboard;
    const jobText = (job, fallback) => {
      if (!job) return fallback;
      const action = job.action.replaceAll("_", " ");
      return `${action} · ${job.status}`;
    };
    $("#pipeline-state").textContent = jobText(pipelineJob, "Idle");
    $("#tensorboard-state").textContent = jobText(tensorboardJob, "Idle");
    setBadge($("#pipeline-badge"), jobText(pipelineJob, "Pipeline idle"), pipeline ? "running" : "neutral");
    setBadge($("#tensorboard-badge"), jobText(tensorboardJob, "TensorBoard idle"), tensorboard ? "running" : "neutral");
    $$('[data-stop="pipeline"]').forEach((button) => { button.disabled = !pipeline; });
    $$('[data-stop="tensorboard"]').forEach((button) => { button.disabled = !tensorboard; });

    const sharePanel = $("#lan-share-panel");
    const viewerUrl = payload.can_edit ? payload.lan_viewer_url : null;
    sharePanel.hidden = !viewerUrl;
    if (viewerUrl) {
      $("#lan-viewer-url").textContent = viewerUrl;
      const qr = $("#lan-viewer-qr");
      if (!qr.getAttribute("src")) qr.src = "/api/lan/qr.png";
    }

    if (payload.can_edit) {
      const lines = payload.logs
        .filter((entry) => entry.sequence > state.logFloor)
        .map((entry) => `[${entry.slot === "tensorboard" ? "TensorBoard" : "Pipeline"}] ${entry.text}`);
      const log = $("#run-log");
      const rendered = lines.length ? lines.join("\n") : "Waiting for a local command…";
      if (log.textContent !== rendered) {
        const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
        log.textContent = rendered;
        if (atBottom) log.scrollTop = log.scrollHeight;
      }
      if (payload.comparison_root) $("#comparison-root").value = payload.comparison_root;
    }
  }

  async function pollState() {
    try {
      renderState(await api("/api/state"));
      if (!state.comparisonLoaded) await loadComparison();
    } catch (error) {
      if (state.canEdit === null) {
        setBadge($("#access-badge"), token ? "Access denied" : "Token required", "viewer");
        $("#remote-notice").hidden = false;
        $("#remote-notice").innerHTML = `<strong>LAN authentication required.</strong> Open the complete viewer URL printed on the host machine.`;
      }
    }
  }

  function formPayload(form) {
    const payload = { action: form.dataset.action };
    [...form.elements].forEach((control) => {
      if (!control.name || control.disabled) return;
      payload[control.name] = control.type === "checkbox" ? control.checked : control.value;
    });
    return payload;
  }

  $$(".action-form").forEach((form) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const button = $("button[type='submit']", form);
      button.disabled = true;
      try {
        const result = await api("/api/run", {
          method: "POST",
          body: JSON.stringify(formPayload(form)),
        });
        toast(`${result.slot === "tensorboard" ? "TensorBoard" : "Pipeline"} started.`);
        location.hash = "#logs";
        await pollState();
      } catch (error) {
        toast(error.message, true);
      } finally {
        button.disabled = false;
      }
    });
  });

  $$('[data-stop]').forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        const result = await api(`/api/stop/${button.dataset.stop}`, { method: "POST" });
        toast(result.stopped ? "Stop requested." : "Process is not running.");
        await pollState();
      } catch (error) { toast(error.message, true); }
    });
  });

  $("#open-tensorboard").addEventListener("click", () => {
    const port = $("[name='port']", $("[data-action='tensorboard']")).value;
    window.open(`http://127.0.0.1:${port}`, "_blank", "noopener");
  });

  $("#clear-log").addEventListener("click", async () => {
    try {
      const payload = await api("/api/state");
      state.logFloor = payload.last_sequence;
      $("#run-log").textContent = "Waiting for new local output…";
    } catch (error) { toast(error.message, true); }
  });

  async function loadConfig() {
    const path = $("#config-path").value.trim();
    try {
      const payload = await api(`/api/config?path=${encodeURIComponent(path)}`);
      $("#config-path").value = payload.path;
      $("#config-editor").value = payload.text;
      toast("Configuration loaded and validated.");
    } catch (error) { toast(error.message, true); }
  }

  $("#load-config").addEventListener("click", loadConfig);
  $("#save-config").addEventListener("click", async () => {
    try {
      const payload = await api("/api/config", {
        method: "PUT",
        body: JSON.stringify({ path: $("#config-path").value, text: $("#config-editor").value }),
      });
      $("#config-path").value = payload.path;
      toast("Configuration validated and saved.");
    } catch (error) { toast(error.message, true); }
  });

  $$('[data-config-preset]').forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        const payload = await api("/api/config/preset", {
          method: "POST",
          body: JSON.stringify({
            path: $("#config-path").value,
            preset: button.dataset.configPreset,
            model: $("#preset-yolo-model").value,
          }),
        });
        $("#config-editor").value = payload.text;
        toast("Preset applied in the editor. Review, then validate and save.");
      } catch (error) { toast(error.message, true); }
    });
  });

  function metric(value) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : "—";
  }

  async function loadComparison() {
    try {
      const selected = $("#comparison-metric").value;
      if (state.canEdit) {
        await api("/api/comparison/root", {
          method: "PUT",
          body: JSON.stringify({ path: $("#comparison-root").value }),
        });
      }
      const payload = await api(`/api/comparison?metric=${encodeURIComponent(selected)}`);
      const body = $("#comparison-body");
      body.replaceChildren();
      if (!payload.runs.length) {
        const row = body.insertRow();
        const cell = row.insertCell();
        cell.colSpan = 10;
        cell.className = "empty";
        cell.textContent = "No compatible training summaries found.";
      }
      payload.runs.forEach((run) => {
        const row = body.insertRow();
        [
          run.model,
          run.augmentation,
          run.split_id,
          metric(run.accuracy),
          metric(run.macro_recall),
          metric(run.macro_f1),
          metric(run.recall_left),
          metric(run.recall_right),
          run.validation_samples ?? "—",
          run.source,
        ].forEach((value) => {
          const cell = row.insertCell();
          cell.textContent = value;
        });
      });
      $("#run-count").textContent = String(payload.runs.length);
      state.comparisonLoaded = true;
    } catch (error) {
      if (state.canEdit !== null) toast(error.message, true);
    }
  }

  $("#refresh-comparison").addEventListener("click", loadComparison);
  $("#comparison-metric").addEventListener("change", loadComparison);
  $("#copy-lan-url").addEventListener("click", async () => {
    const viewerUrl = $("#lan-viewer-url").textContent;
    try {
      await navigator.clipboard.writeText(viewerUrl);
      toast("Viewer link copied.");
    } catch (_error) {
      toast("Could not copy automatically. Select the displayed link instead.", true);
    }
  });

  const pathDialog = $("#path-dialog");
  let pathTarget = null;
  let pathKind = "file";
  let currentPath = "";

  async function browsePath(path = "") {
    try {
      const payload = await api(`/api/paths?path=${encodeURIComponent(path)}`);
      currentPath = payload.path;
      $("#path-current").textContent = payload.path;
      $("#path-parent").disabled = !payload.parent;
      $("#path-parent").dataset.path = payload.parent || "";
      const entries = $("#path-entries");
      entries.replaceChildren();
      payload.entries.forEach((entry) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `path-entry${entry.is_directory ? " directory" : ""}`;
        button.textContent = entry.name;
        button.addEventListener("click", () => {
          if (entry.is_directory) browsePath(entry.path);
          else if (pathKind === "file") {
            pathTarget.value = entry.path;
            pathDialog.close();
          }
        });
        entries.append(button);
      });
    } catch (error) { toast(error.message, true); }
  }

  $$(".browse").forEach((button) => {
    button.addEventListener("click", () => {
      pathTarget = document.getElementById(button.dataset.target);
      pathKind = button.dataset.kind || "file";
      pathDialog.showModal();
      browsePath(pathTarget.value);
    });
  });
  $("#path-parent").addEventListener("click", (event) => browsePath(event.currentTarget.dataset.path));
  $("#path-use-folder").addEventListener("click", () => {
    if (pathTarget) pathTarget.value = currentPath;
    pathDialog.close();
  });
  $("#path-close").addEventListener("click", () => pathDialog.close());

  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    $$(".nav-link").forEach((link) => {
      const active = link.hash === `#${visible.target.id}`;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
  }, { rootMargin: "-20% 0px -65%", threshold: [0, 0.2, 0.5] });
  $$(".page-section").forEach((section) => observer.observe(section));

  pollState().then(() => {
    if (state.canEdit) loadConfig();
  });
  window.setInterval(pollState, 1200);
})();
