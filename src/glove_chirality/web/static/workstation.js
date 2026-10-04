/* Supplemental workstation presentation. Existing controllers own all process/I/O actions. */
(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const text = (value) => value === undefined || value === null ? '—' : String(value);
  const metric = (value) => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '—';
  const node = (tag, content, className) => { const element = document.createElement(tag); element.textContent = content; if (className) element.className = className; return element; };
  function cards(container, values) {
    container.replaceChildren();
    values.forEach(([label, value]) => { const item = node('div', '', 'telemetry-card'); item.append(node('span', label), node('strong', text(value))); container.append(item); });
  }
  let summary;
  const training = $('form[data-action="train"]');
  if (training) {
    summary = node('section', '', 'dataset-summary host-only'); summary.id = 'training-dataset-summary';
    summary.hidden = document.body.dataset.local !== 'true';
    summary.append(node('h3', 'Dataset summary'), node('p', 'Validate the selected manifest before training. Counts and split are calculated by the server.'));
    const details = node('div', '', 'telemetry-grid'); details.id = 'dataset-summary-cards'; summary.append(details);
    const validate = node('button', 'Audit dataset / validate settings'); validate.type = 'button'; summary.append(validate);
    training.before(summary);
    let version = 0;
    training.elements.manifest.addEventListener('input', () => { version += 1; details.replaceChildren(node('p', 'Manifest changed — validate again.')); });
    validate.addEventListener('click', async () => {
      const ticket = ++version; validate.disabled = true; details.replaceChildren(node('p', 'PREFLIGHT — checking actual images, class counts and source-group split…'));
      const payload = Object.fromEntries(new FormData(training));
      training.querySelectorAll('input[type="checkbox"]').forEach((input) => { payload[input.name] = input.checked; });
      try {
        if (document.body.dataset.local !== 'true') return;
        const response = await fetch('/api/jobs/preflight', {signal: AbortSignal.timeout(15000), method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({action:'train', ...payload})});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `Preflight returned ${response.status}`);
        if (ticket !== version) return;
        showDataset(result.dataset);
        details.append(node('p', result.ok ? 'DATASET READY — required settings validated.' : 'NOT READY — correct the highlighted fields.'));
        if (!result.ok) window.GripJobs?.fail(result, training, result.summary || 'Settings need attention');
        if (result.warnings?.length) details.append(node('p', result.warnings.map(warning => warning.message || warning).join('\n'), 'warning'));
      } catch (error) { if (ticket === version) { details.replaceChildren(node('p', `Validation failed: ${error.message}`)); window.GripJobs?.fail({error:error.message}, training); } }
      finally { validate.disabled = false; }
    });
  }
  function showDataset(report) {
    const container = $('#dataset-summary-cards'); if (!container || !report) return;
    const counts = report.counts || {}; const split = report.grouped_split || {}; const duplicate = report.duplicates || {};
    cards(container, [['LEFT', counts.classes?.left ?? 0], ['RIGHT', counts.classes?.right ?? 0], ['Total', counts.total],
      ['Train', split.train_count], ['Validation', split.validation_count], ['Source groups', counts.sources],
      ['Invalid', report.validity?.invalid_rows?.length ?? 0], ['Duplicate content groups', duplicate.content?.length ?? 0]]);
  }
  const monitor = node('details', '', 'workstation-monitor'); monitor.id = 'workstation-monitor';
  monitor.append(node('summary', 'Real-time monitoring · expand for metrics'));
  const monitorCards = node('div', '', 'telemetry-grid'); monitor.append(monitorCards);
  $('main')?.prepend(monitor);
  function render(state) {
    const pipeline = state.jobs?.pipeline;
    if (pipeline?.action === 'train' && pipeline.preflight?.dataset) showDataset(pipeline.preflight.dataset);
    const active = state.running?.pipeline ? pipeline : state.inference?.running ? state.inference : state.factory?.running ? state.factory : pipeline;
    const progress = active?.progress || {};
    const values = progress.validation || active?.result?.best_validation || {};
    const device = progress.device || state.inference?.actual_device || state.factory?.actual_device;
    const gpu = state.gpu?.gpus?.[0];
    const latest = state.inference?.running ? state.inference.latest : state.factory?.latest;
    cards(monitorCards, [['Workflow', active?.action || active?.workflow || 'Idle'], ['Stage', active?.current_stage || active?.stage],
      ['Epoch', progress.epoch ? `${progress.epoch} / ${progress.epochs}` : '—'], ['Train loss', progress.train_loss],
      ['Validation loss', progress.validation_loss], ['RIGHT recall', metric(values.recall_per_class?.[1])],
      ['LEFT recall', metric(values.recall_per_class?.[0])], ['Macro recall', metric(values.macro_recall)],
      ['Macro F1', metric(values.macro_f1)], ['Device', device],
      ['GPU utilization (system)', gpu ? `${gpu.utilization_percent}%` : 'Unavailable'],
      ['VRAM', gpu ? `${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB` : 'Unavailable'],
      ['Latest event', latest ? `${latest.prediction || latest.status} · ${latest.event_id}` : '—']]);
  }
  window.GripWorkstation = {render, showDataset};
})();
