(() => {
  'use strict';
  const $ = s => document.querySelector(s);
  const bar = document.createElement('div'); bar.id = 'global-status'; bar.setAttribute('role', 'status'); bar.textContent = 'Connecting to backend…'; document.querySelector('header').after(bar);
  const dialog = document.createElement('dialog'); dialog.id = 'job-error';
  dialog.innerHTML = '<h2>Workflow needs attention</h2><p id="job-error-message"></p><ul id="job-checks"></ul><p>Fix the highlighted fields or resource paths, review the run log, then retry. No work has been started for failed preflight checks.</p><div class="form-actions"><button type="button" class="button" id="job-fix">Fix all fields</button><button type="button" class="button" id="job-retry">Retry</button><button type="button" class="button" id="job-log">View log</button><button type="button" class="button" id="job-cancel">Close</button></div>';
  document.body.append(dialog);
  let failedForm, retryAction, highlighted = [], lastJobs = new Map(), bootstrapped = false, renderedResult = '';
  const reported = new Set();
  function checks(report) { const value = report?.checks || []; return Array.isArray(value) ? value : Object.entries(value).map(([field, check]) => typeof check === 'object' ? {field, ...check} : {field, message:String(check)}); }
  function fail(report, form, message, retry) {
    if (document.body.dataset.local !== 'true') return;
    failedForm = form; retryAction = retry; highlighted.forEach(field => { field.classList.remove('invalid-field'); field.removeAttribute('aria-invalid'); }); highlighted = [];
    const detail = message || report?.error || 'Preflight failed';
    $('#job-error-message').textContent = typeof detail === 'object' ? detail.message || JSON.stringify(detail) : detail;
    const list = $('#job-checks'); list.replaceChildren();
    checks(report).forEach(check => {
      const li = document.createElement('li'); li.textContent = `${check.label || check.field || 'Check'} · ${check.status || 'unavailable'}: ${check.message || ''}`; list.append(li);
      if (check.status === 'failed' && form) {
        const field = [...form.querySelectorAll('input,select,textarea')].find(el => el.name === check.field || el.id === check.field || el.id === `factory-${check.field?.replaceAll('_','-')}`);
        if (field) { field.classList.add('invalid-field'); field.setAttribute('aria-invalid','true'); highlighted.push(field); }
      }
    });
    (report?.errors || []).forEach(error => { const li = document.createElement('li'); li.textContent = typeof error === 'string' ? error : error.message || JSON.stringify(error); list.append(li); });
    if (!dialog.open) dialog.showModal();
  }
  $('#job-cancel').onclick = () => dialog.close();
  $('#job-fix').onclick = () => { dialog.close(); const field = highlighted[0]; if (field) { field.closest('details')?.setAttribute('open',''); field.focus(); field.scrollIntoView({block:'center'}); } else failedForm?.scrollIntoView({block:'center'}); };
  $('#job-retry').onclick = () => { dialog.close(); if (retryAction) retryAction(); else if (failedForm?.id === 'factory') $('#factory-start').click(); else failedForm?.requestSubmit?.(); };
  $('#job-log').onclick = () => { dialog.close(); location.hash = '#logs'; };
  const output = document.createElement('article'); output.id = 'job-result'; output.className = 'panel host-only'; output.hidden = true; $('#logs').prepend(output);
  function result(job) {
    if (!job?.result && !job?.artifacts?.length) return;
    const key = JSON.stringify([job.job_id, job.result, job.artifacts]); if (key === renderedResult) return; renderedResult = key;
    output.hidden = false; output.replaceChildren();
    const heading = document.createElement('h2'); heading.textContent = 'Workflow results & artifacts'; output.append(heading);
    const pre = document.createElement('pre'); pre.textContent = JSON.stringify(job.result || {}, null, 2); output.append(pre);
    if (job.output) { const path = document.createElement('input'); path.value = job.output; path.readOnly = true; path.setAttribute('aria-label','Job output location'); const browse = document.createElement('button'); browse.className = 'button'; browse.textContent = 'Browse output location'; browse.onclick = () => window.GripBrowse(path); output.append(path,browse); }
    const checkpoint = job.result?.best_checkpoint || job.result?.checkpoint || (job.artifacts || []).find(a => a.kind === 'checkpoint' || /\.(pt|pth)$/i.test(a.name || ''))?.path;
    if (checkpoint) { const button = document.createElement('button'); button.className = 'button'; button.textContent = 'Run on test video'; button.onclick = () => { document.querySelectorAll('#infer [name="checkpoint"]').forEach(field => {field.value = checkpoint;}); $('#visual-checkpoint').value = checkpoint; $('#visual-form').elements.source_type.value = 'video'; if ($('#visual-source').value === '0') $('#visual-source').value = ''; location.hash = '#infer'; $('#visual-source').focus(); }; output.append(button); }
    (job.artifacts || []).forEach((artifact,index) => {
      const artifactUrl = artifact.url || (job.job_id ? `/api/jobs/${encodeURIComponent(job.job_id)}/artifacts/${index}` : null);
      if (!artifactUrl) { const p = document.createElement('p'); p.textContent = `${artifact.name || artifact.path}: host download URL unavailable`; output.append(p); return; }
      const url = new URL(artifactUrl, location.origin); if (url.origin !== location.origin || !url.pathname.startsWith('/api/')) return;
      const link = document.createElement('a'); link.href = url.href; link.textContent = artifact.name || artifact.kind || 'Artifact'; link.className = 'button'; output.append(link);
      if (artifact.kind === 'image' || /\.(png|jpe?g)$/i.test(artifact.name || artifact.path || '')) { const image = document.createElement('img'); image.src = url.href; image.dataset.workflow = job.action; image.dataset.jobId = job.job_id; image.alt = 'Generated calibration / diagnostic artifact'; image.className = 'artifact-preview'; output.append(image); }
    });
  }
  function render(payload) {
    const jobs = Object.values(payload.jobs || {}).filter(Boolean);
    const statusItems = [...jobs,...[payload.factory,payload.inference].filter(Boolean)];
    bar.classList.remove('connection-lost');
    bar.textContent = statusItems.map(job => `${job.job_id || 'ID unavailable'} · ${(job.action || job.workflow || 'job').replaceAll('_',' ')} · ${(job.status || 'unavailable').toUpperCase()} · ${job.current_stage || job.stage || 'stage unavailable'} · elapsed ${job.elapsed_time ?? job.elapsed ?? job.elapsed_s ?? 'unavailable'} · progress ${job.progress == null ? 'unavailable' : typeof job.progress === 'object' ? JSON.stringify(job.progress) : job.progress}`).join(' | ') || 'No active job';
    bar.textContent += ` | GPU ${payload.gpu?.name || payload.gpu?.status || 'unavailable'} | Factory ${(payload.factory?.status || 'unavailable').toUpperCase()} | Alerts ${$('#enable-alerts').textContent}`;
    const inference = payload.inference;
    (inference?.events || []).forEach(event => { if (!event.event_id) return; const key = `inference:${inference.session_id || inference.job_id || inference.started_at || ''}:${event.event_id}`; if (!bootstrapped) window.GripAlerts.remember(key); else if (event.status === 'accepted' && event.prediction === (inference.reject_class || 'right')) window.GripAlerts.tone('reject',key); });
    jobs.forEach(job => {
      const id = job.job_id || `${job.action}:${job.started_at}`; const status = String(job.status).toLowerCase();
      if (bootstrapped && job.preflight?.warnings?.length) window.GripAlerts?.tone('warning',`job:${id}:warnings`);
      if (bootstrapped && lastJobs.has(id) && lastJobs.get(id) !== status && ['completed','succeeded','failed'].includes(status)) window.GripAlerts?.tone(status === 'failed' ? 'error' : 'success',`job:${id}:${status}`);
      lastJobs.set(id,status); if (lastJobs.size > 100) lastJobs.delete(lastJobs.keys().next().value);
      if (payload.can_edit && status === 'failed' && !reported.has(id)) { reported.add(id); if (reported.size > 100) reported.delete(reported.values().next().value); fail(job.preflight, [...document.querySelectorAll('.action-form')].find(form => form.dataset.action === job.action) || (job.action === 'audit_dataset' ? document.querySelector('[data-action="train"]') : null), job.error, job.action === 'audit_dataset' ? () => $('#audit-dataset').click() : null); }
      if (payload.can_edit) result(job);
    }); bootstrapped = true;
  }
  window.GripJobs = { render, fail, checks, disconnected(error) { bar.classList.add('connection-lost'); bar.textContent = `BACKEND CONNECTION LOST · ${error.message} · retrying; displayed state may be stale`; }, pending(id) {bar.textContent = `${id || 'Job queued'} · PREFLIGHT · awaiting resource validation`; } };
})();
