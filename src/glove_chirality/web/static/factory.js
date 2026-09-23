(() => {
  "use strict";

  const token = sessionStorage.getItem("grip-lan-token") || "";
  const $ = (selector) => document.querySelector(selector);
  const seenRejects = new Set();
  let audioArmed = false;
  let lastCropEvent = "";
  let previewTimer = 0;

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (token) headers.set("X-GRIP-Token", token);
    if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    const response = await fetch(path, { ...options, headers });
    const type = response.headers.get("content-type") || "";
    const payload = type.includes("json") ? await response.json().catch(() => ({})) : {};
    if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
    return payload;
  }

  function beep() {
    const context = new AudioContext();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.frequency.value = 880;
    gain.gain.value = 0.08;
    oscillator.connect(gain).connect(context.destination);
    oscillator.start();
    gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.22);
    oscillator.stop(context.currentTime + 0.24);
  }

  function selectedValue(select, customInput) {
    if (!select) return "";
    if (select.value === "__custom__") return customInput ? customInput.value.trim() : "";
    return select.value;
  }

  function fillSelect(select, items, placeholder, customLabel) {
    const current = select.value;
    select.replaceChildren();
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = placeholder;
    select.append(blank);
    items.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.value;
      option.textContent = item.label;
      if (item.valid === false) option.dataset.invalid = "true";
      if (item.error) option.dataset.error = item.error;
      select.append(option);
    });
    if (customLabel) {
      const custom = document.createElement("option");
      custom.value = "__custom__";
      custom.textContent = customLabel;
      select.append(custom);
    }
    if ([...select.options].some((option) => option.value === current)) select.value = current;
  }

  async function loadCatalog() {
    const [configs, devices, status] = await Promise.all([
      api("/api/factory/configs"),
      api("/api/factory/devices"),
      api("/api/factory/status"),
    ]);
    fillSelect($("#factory-config"), configs.configs.map((item) => ({
      value: item.path,
      label: item.name,
      valid: item.valid,
      error: item.error,
    })), "Choose a YAML file", "Custom path...");
    fillSelect($("#factory-device"), devices.choices.map((item) => ({ value: item, label: item })), "Device");
    $("#factory-cuda-status").textContent = devices.label;
    const settings = await api(`/api/factory/checkpoints?root=${encodeURIComponent($("#factory-models-root").value || "")}`);
    if (settings.root && !$("#factory-models-root").value) $("#factory-models-root").value = settings.root;
    fillCheckpoints(settings.checkpoints || []);
    render(status);
  }

  function fillCheckpoints(checkpoints) {
    fillSelect($("#factory-checkpoint"), checkpoints.map((item) => ({
      value: item.path,
      label: item.label,
    })), "Choose a checkpoint", "Custom checkpoint...");
  }

  function render(status) {
    $("#factory-stage").textContent = status.fault ? `Fault: ${status.fault}` : (status.status || "idle");
    const running = Boolean(status.running);
    ["factory-camera", "factory-checkpoint", "factory-config", "factory-device"].forEach((id) => {
      const element = document.getElementById(id);
      if (element) element.disabled = running;
    });
    const counters = status.counters || {};
    $("#count-accepted").textContent = counters.total_accepted ?? 0;
    $("#count-left").textContent = counters.left_pass ?? 0;
    $("#count-right").textContent = counters.right_reject ?? 0;
    $("#count-pipeline").textContent = counters.pipeline_rejected ?? 0;
    $("#count-multiple").textContent = counters.multiple_candidates ?? 0;
    $("#count-partial").textContent = counters.partial ?? 0;
    $("#count-commands").textContent = counters.commands_sent ?? 0;
    const metrics = status.metrics || {};
    const num = (value) => value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(1);
    $("#metric-capture-fps").textContent = num(metrics.capture_fps);
    $("#metric-processed-fps").textContent = num(metrics.processed_fps);
    $("#metric-yolo").textContent = num(metrics.yolo_ms);
    $("#metric-classifier").textContent = num(metrics.classifier_ms);
    $("#metric-event").textContent = num(metrics.event_ms);
    $("#metric-accepted-latency").textContent = num(metrics.accepted_latency_ms);
    $("#metric-dropped").textContent = metrics.dropped_frames ?? "—";
    const serial = status.serial || {};
    $("#factory-serial-status").textContent = serial.connected
      ? `Connected ${serial.port || ""} · last ${serial.last_command || "—"} · ACK ${serial.last_ack || "—"}`
      : `Disconnected${serial.fault ? ` · ${serial.fault}` : ""}`;
    const camera = status.camera_actual || {};
    const request = status.camera_request || {};
    const requested = request.width && request.height ? `${request.width}x${request.height}` : "auto";
    const actual = camera.width && camera.height ? `${camera.width}x${camera.height}` : "—";
    $("#factory-capture-info").textContent = `Capture: ${actual} · Requested: ${requested} · FPS requested ${request.fps || "auto"} · actual ${camera.fps ?? "—"} · Backend: ${camera.backend || "—"} · FourCC: ${camera.fourcc || request.fourcc || "auto"} · Inference frame unchanged · Preview encode: ${status.preview_encode_ms == null ? "—" : Number(status.preview_encode_ms).toFixed(1)} ms · YOLO imgsz: ${status.yolo_imgsz || 640}`;
    const warning = $("#factory-geometry-warning");
    if (status.geometry_warning) {
      warning.hidden = false;
      warning.textContent = status.geometry_warning;
    } else {
      warning.hidden = true;
    }
    const counts = status.yolo_counts || {};
    $("#factory-yolo-counts").textContent = `YOLO raw: ${counts.raw ?? "—"} · size rejected: ${counts.size_rejected ?? "—"} · kept: ${counts.kept ?? "—"} · eligible: ${counts.eligible ?? "—"}`;
    const positions = status.positions || [];
    $("#factory-position").textContent = positions.length
      ? positions.map((item) => `${item.state} X=${Number(item.center_px[0]).toFixed(0)} px, Y=${Number(item.center_px[1]).toFixed(0)} px · ${Number(item.center_norm[0]).toFixed(3)}, ${Number(item.center_norm[1]).toFixed(3)}`).join(" | ")
      : "Position: waiting for a detection";
    renderDiagnostics(status.latest, status.reject_class || "right");
    renderEvents(status.events || []);
    if (document.body.dataset.local === "true" && running) startPreview();
    else stopPreview();
    maybeBeep(status.latest, status.reject_class || "right");
  }

  const decisionClass = {
    WAITING: "decision-waiting",
    INSPECTING: "decision-inspecting",
    CLASSIFYING: "decision-classifying",
    PASS: "decision-pass",
    REJECT: "decision-reject",
    "NO DECISION": "decision-none",
  };

  function renderDecision(decision) {
    const card = $("#factory-decision");
    const label = $("#factory-decision-label");
    if (!card || !label) return;
    const phase = decision && decision.phase ? decision.phase : "WAITING";
    label.textContent = phase;
    card.className = `decision-card ${decisionClass[phase] || "decision-waiting"}`;
  }

  function renderDiagnostics(latest, rejectClass) {
    const detail = $("#factory-latest-detail");
    const crop = $("#factory-crop");
    if (!detail || !crop) return;
    if (!latest) {
      detail.textContent = "Confidence, crop, timestamps, and detector details stay here and in the session log.";
      crop.hidden = true;
      return;
    }
    const accepted = latest.status === "accepted";
    detail.textContent = [
      accepted ? `Layer-2 ${latest.prediction || "—"} ${latest.confidence == null ? "" : Number(latest.confidence).toFixed(3)}` : "Classifier not run",
      `Detector ${latest.detector_confidence ?? "—"}`,
      latest.event_id || "",
      latest.wall_time_iso || "",
      latest.center_px ? `X=${Number(latest.center_px[0]).toFixed(0)} Y=${Number(latest.center_px[1]).toFixed(0)}` : "",
      `crossing ${latest.trigger_crossing_wall_time_iso || "not observed"}`,
      `classifier ${latest.classifier_ms ?? "—"} ms`,
      `reject class ${rejectClass}`,
    ].filter(Boolean).join(" · ");
    if (document.body.dataset.local === "true" && latest.event_id && latest.event_id !== lastCropEvent && accepted) {
      lastCropEvent = latest.event_id;
      crop.hidden = false;
      crop.src = `/api/factory/crop.jpg?t=${Date.now()}`;
    }
  }

  function renderEvents(events) {
    const body = $("#factory-events");
    body.replaceChildren();
    const rows = events.slice(-40).reverse();
    if (!rows.length) {
      const row = body.insertRow();
      const cell = row.insertCell();
      cell.colSpan = 10;
      cell.className = "empty";
      cell.textContent = "No passages yet.";
      return;
    }
    rows.forEach((event) => {
      const row = body.insertRow();
      const position = event.center_px ? `X=${Number(event.center_px[0]).toFixed(0)},Y=${Number(event.center_px[1]).toFixed(0)}` : "—";
      [
        event.wall_time_iso || "—",
        event.event_id || "—",
        event.status || "—",
        event.prediction || "—",
        event.prediction == null ? "not run" : Number(event.confidence).toFixed(3),
        event.detector_confidence ?? "—",
        position,
        event.trigger_crossing_wall_time_iso || "—",
        event.result || (event.status === "accepted" ? "—" : "pipeline"),
        event.actuator_command || (event.would_reject ? "would reject" : "—"),
      ].forEach((value) => {
        row.insertCell().textContent = value;
      });
    });
  }

  function maybeBeep(latest, rejectClass) {
    if (!latest || !$("#factory-audio").checked || !audioArmed) return;
    if (latest.status !== "accepted" || latest.prediction !== rejectClass) return;
    if (seenRejects.has(latest.event_id)) return;
    seenRejects.add(latest.event_id);
    if (seenRejects.size > 500) seenRejects.delete(seenRejects.values().next().value);
    beep();
  }

  function startPreview() {
    if (previewTimer) return;
    let busy = false;
    const tick = () => {
      const image = $("#factory-preview");
      if (!image || busy) return;
      busy = true;
      const done = () => { busy = false; };
      image.onload = done;
      image.onerror = done;
      image.src = `/api/factory/frame.jpg?t=${Date.now()}`;
    };
    tick();
    previewTimer = window.setInterval(tick, 120);
  }

  function stopPreview() {
    if (previewTimer) window.clearInterval(previewTimer);
    previewTimer = 0;
  }

  async function startSession(confirmArmed) {
    const config = $("#factory-config");
    const selected = config.selectedOptions[0];
    if (selected && selected.dataset.invalid === "true") {
      $("#factory-config-warning").hidden = false;
      $("#factory-config-warning").textContent = selected.dataset.error || "Selected config is invalid.";
      return;
    }
    audioArmed = true;
    await api("/api/factory/start", {
      method: "POST",
      body: JSON.stringify({
        camera_preset: $("#factory-camera-preset").value,
        camera_backend: $("#factory-camera-backend").value,
        camera_width: $("#factory-camera-width").value,
        camera_height: $("#factory-camera-height").value,
        camera_fps: $("#factory-camera-fps").value,
        camera_fourcc: $("#factory-camera-fourcc").value,
        geometry: window.factoryGeometry || "yaml",
        source: selectedValue($("#factory-camera"), $("#factory-custom-source")),
        checkpoint: selectedValue($("#factory-checkpoint"), $("#factory-custom-checkpoint")),
        config: selectedValue($("#factory-config"), $("#factory-custom-config")),
        device: $("#factory-device").value || "auto",
        amp: $("#factory-amp").checked,
        mode: $("#factory-mode").value,
        reject_class: $("#factory-reject-class").value,
        decision_class: $("#factory-decision-class").value,
        decision_threshold: Number($("#factory-decision-threshold").value),
        belt_direction: $("#factory-belt-direction").value,
        trigger_line_enabled: $("#factory-trigger-enabled").checked,
        trigger_line_fraction: Number($("#factory-trigger-fraction").value),
        delay_ms: Number($("#factory-delay").value),
        continue_counters: $("#factory-continue-counters").checked,
        confirm_armed: confirmArmed,
      }),
    });
  }

  function bind() {
    if (!$("#factory")) return;
    $("#factory-camera-preset").addEventListener("change", () => {
      $("#factory-camera-custom").hidden = $("#factory-camera-preset").value !== "custom";
    });
    $("#factory-grip-geometry").addEventListener("click", () => {
      window.factoryGeometry = "grip";
      $("#factory-belt-direction").value = "bottom_to_top";
      $("#factory-trigger-enabled").checked = true;
      $("#factory-trigger-fraction").value = "0.5";
      $("#factory-geometry-note").textContent = "Runtime geometry override active";
    });
    $("#factory-reset-geometry").addEventListener("click", () => {
      window.factoryGeometry = "yaml";
      $("#factory-geometry-note").textContent = "Using the selected YAML geometry.";
    });
    $("#factory-show-rejected").addEventListener("change", async () => {
      await api("/api/factory/display", { method: "POST", body: JSON.stringify({ show_size_rejected: $("#factory-show-rejected").checked }) });
    });
    $("#factory-expand").addEventListener("click", () => {
      const panel = $("#factory-camera-panel");
      if (panel.requestFullscreen) panel.requestFullscreen();
    });
    $("#factory-scan").addEventListener("click", async () => {
      const payload = await api("/api/factory/cameras");
      fillSelect($("#factory-camera"), payload.cameras.map((item) => ({ value: String(item.index), label: item.label })), "Choose a camera", "Custom source...");
    });
    $("#factory-camera").addEventListener("change", () => {
      $("#factory-custom-source-wrap").hidden = $("#factory-camera").value !== "__custom__";
    });
    $("#factory-checkpoint").addEventListener("change", async () => {
      $("#factory-custom-checkpoint-wrap").hidden = $("#factory-checkpoint").value !== "__custom__";
      if (!$("#factory-checkpoint").value || $("#factory-checkpoint").value === "__custom__") return;
      const meta = await api(`/api/factory/checkpoint?path=${encodeURIComponent($("#factory-checkpoint").value)}`);
      $("#factory-checkpoint-meta").textContent = `${meta.model_name || "unknown model"} · image ${meta.image_size || "—"} · classes ${(meta.classes || []).join(", ") || "—"} · ${meta.modified || ""}`;
    });
    $("#factory-sha256").addEventListener("click", async () => {
      const path = selectedValue($("#factory-checkpoint"), $("#factory-custom-checkpoint"));
      const meta = await api(`/api/factory/checkpoint?path=${encodeURIComponent(path)}&sha256=1`);
      $("#factory-checkpoint-meta").textContent = `${$("#factory-checkpoint-meta").textContent} · ${meta.sha256 || "no hash"}`;
    });
    $("#factory-models-save").addEventListener("click", async () => {
      const payload = await api("/api/factory/models-root", { method: "POST", body: JSON.stringify({ path: $("#factory-models-root").value }) });
      fillCheckpoints(payload.checkpoints || []);
    });
    $("#factory-config").addEventListener("change", () => {
      const selected = $("#factory-config").selectedOptions[0];
      $("#factory-custom-config-wrap").hidden = $("#factory-config").value !== "__custom__";
      $("#factory-config-path").textContent = $("#factory-config").value && $("#factory-config").value !== "__custom__" ? $("#factory-config").value : "";
      const invalid = selected && selected.dataset.invalid === "true";
      $("#factory-config-warning").hidden = !invalid;
      $("#factory-config-warning").textContent = invalid ? (selected.dataset.error || "This config failed validation.") : "";
    });
    $("#factory-reject-class").addEventListener("change", () => {
      $("#factory-reject-warning").hidden = $("#factory-reject-class").value === "right";
    });
    $("#factory-refresh-ports").addEventListener("click", async () => {
      const payload = await api("/api/factory/ports");
      fillSelect($("#factory-serial-port"), payload.ports.map((item) => ({ value: item.device, label: item.label })), "Choose a port");
    });
    $("#factory-connect").addEventListener("click", async () => {
      await api("/api/factory/serial/connect", { method: "POST", body: JSON.stringify({ port: $("#factory-serial-port").value, baud: Number($("#factory-baud").value) }) });
    });
    $("#factory-disconnect").addEventListener("click", async () => {
      await api("/api/factory/serial/disconnect", { method: "POST", body: "{}" });
    });
    $("#factory-start").addEventListener("click", async () => {
      if ($("#factory-mode").value === "armed") {
        $("#arm-dialog").showModal();
        return;
      }
      await startSession(false);
    });
    $("#factory-mode").addEventListener("change", async () => {
      if ($("#factory-mode").value === "armed") {
        $("#arm-dialog").returnValue = "";
        $("#arm-dialog").showModal();
        return;
      }
      try {
        await api("/api/factory/mode", { method: "POST", body: JSON.stringify({ mode: $("#factory-mode").value, confirm: false }) });
      } catch (error) {
        window.alert(error.message);
      }
    });
    $("#arm-cancel").addEventListener("click", () => {
      $("#factory-mode").value = "shadow";
    });
    $("#arm-confirm").addEventListener("click", async () => {
      try {
        const status = await api("/api/factory/status");
        if (status.running) {
          await api("/api/factory/mode", { method: "POST", body: JSON.stringify({ mode: "armed", confirm: true }) });
          return;
        }
        await startSession(true);
      } catch (error) {
        window.alert(error.message);
      }
    });
    $("#factory-stop").addEventListener("click", async () => {
      await api("/api/factory/stop", { method: "POST", body: "{}" });
      stopPreview();
    });
    $("#factory-reset-counters").addEventListener("click", async () => {
      if (!window.confirm("Reset Factory Live counters?")) return;
      await api("/api/factory/counters/reset", { method: "POST", body: JSON.stringify({ confirm: true }) });
    });
    document.querySelectorAll("#factory button, #factory-start").forEach((button) => {
      button.addEventListener("click", async (event) => {
        try {
          if (event.currentTarget.id === "factory-start" || event.currentTarget.id === "arm-confirm") return;
        } catch (error) {
          window.alert(error.message);
        }
      });
    });
    window.setInterval(async () => {
      try { renderDecision(await api("/api/factory/decision")); } catch (_error) { /* viewer may be unauthenticated */ }
    }, 100);
    window.setInterval(async () => {
      try { render(await api("/api/factory/status")); } catch (_error) { /* viewer may be unauthenticated */ }
    }, 400);
    loadCatalog().catch((error) => {
      const stage = $("#factory-stage");
      if (stage && document.body.dataset.local === "true") stage.textContent = error.message;
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bind);
  else bind();
})();
