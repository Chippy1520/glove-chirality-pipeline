(() => {
  'use strict';
  const $ = s => document.querySelector(s);
  async function api(path, body) {
    if (path.endsWith('/playback') && body?.paused !== undefined) body = {...body,pause:body.paused};
    const token = sessionStorage.getItem('grip-lan-token');
    const response = await fetch(path, {signal:AbortSignal.timeout(15000),method: body === undefined ? 'GET' : 'POST', headers: {'Content-Type':'application/json', ...(token ? {'X-GRIP-Token':token} : {})}, ...(body === undefined ? {} : {body:JSON.stringify(body)})});
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) { const error = new Error(payload.error || `${response.status} ${response.statusText}`); error.report = payload.preflight || payload; throw error; } return payload;
  }
  window.GripAPI = api;
  const panel = document.createElement('article'); panel.className = 'panel host-only'; panel.id = 'visualization';
  panel.innerHTML = '<h2>Visual inference · SHADOW ONLY</h2><p>No serial commands or physical actuation. Simulated triggers are logged alarms only.</p><form id="visual-form" class="form-grid" novalidate><label class="field">Source type<select name="source_type"><option value="camera">Camera</option><option value="video">Video file</option><option value="stream">Stream</option></select></label><label class="field">Source<input id="visual-source" name="source" value="0"></label><label class="field">Config<input name="config" value="configs/default.yaml"></label><label class="field">Checkpoint<input id="visual-checkpoint" name="checkpoint"></label><label class="field">Device<select name="device"><option>auto</option><option>cpu</option><option>cuda</option><option>cuda:0</option></select></label><label class="field">Decision policy<select name="decision_class"><option>argmax</option><option>right</option><option>left</option></select></label><label class="field">Threshold<input name="decision_threshold" value="0.5" type="number" step="0.01"></label><label class="field">CUDA AMP<input type="checkbox" name="amp"></label><div class="form-actions"><button class="button primary" type="submit">Start visualization</button><button class="button danger" type="button" id="visual-stop">Stop</button><button class="button" type="button" id="visual-simulate">Simulate alarm (no actuator)</button></div></form><p id="visual-status" role="status">Idle</p><div class="form-actions"><button type="button" class="button" id="visual-pause">Pause</button><label>Playback speed<select id="visual-speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></label></div><img id="visual-frame" class="artifact-preview" alt="Visual inference frame" hidden><img id="visual-crop" class="factory-crop" alt="Latest canonical crop" hidden><pre id="visual-detail"></pre>';
  $('#infer .section-heading').after(panel);
  const decision = document.createElement('div'); decision.id = 'visual-decision'; decision.className = 'decision-card decision-waiting'; decision.setAttribute('aria-live','polite'); decision.textContent = 'WAITING · accepted passages only'; $('#visual-status').after(decision);
  const startup = document.createElement('pre'); startup.id = 'visual-checks'; $('#visual-status').after(startup);
  let paused = false, polling = false, frameBusy = false, lastCrop = '', session = '', initialized = false, lastFault = '';
  function render(status) {
    const running = Boolean(status.running);
    $('#visual-status').textContent = `${(status.status || 'unavailable').toUpperCase()} · ${status.current_stage || 'stage unavailable'} · ${status.fault || ''}`;
    [...$('#visual-form').elements].forEach(el => { if (el.name) el.disabled = running; });
    $('#visual-pause').disabled = !running || status.source_type !== 'video'; $('#visual-speed').disabled = !running || status.source_type !== 'video';
    startup.textContent = window.GripJobs.checks(status.preflight || {checks:status.checks}).map(check => `${check.label || check.field}: ${check.status} · ${check.message || ''}`).join('\n') || 'Startup checks unavailable';
    const latest = status.latest;
    if (latest?.status === 'accepted' && latest.prediction) { const reject = latest.prediction === (status.reject_class || 'right'); decision.className = `decision-card ${reject ? 'decision-reject' : 'decision-pass'}`; decision.textContent = `${latest.prediction.toUpperCase()} / ${reject ? 'REJECT' : 'PASS'} · Confidence ${latest.confidence ?? 'unavailable'} · Threshold ${status.threshold ?? status.decision_threshold ?? 'unavailable'} · ${status.policy || status.decision_class || 'policy unavailable'} · SHADOW · NO PHYSICAL ACTUATION`; }
    $('#visual-detail').textContent = JSON.stringify({mode:status.mode, metrics:status.metrics || 'unavailable', camera_actual:status.camera_actual || 'unavailable', latest:latest || 'unavailable'},null,2);
    if (running && !frameBusy) { frameBusy = true; const image = $('#visual-frame'); image.hidden = false; image.onload = image.onerror = () => {frameBusy = false;}; image.src = `/api/inference/frame.jpg?t=${Date.now()}`; }
    const id = `inference:${status.session_id || status.job_id || status.started_at || session}:${latest?.event_id}`;
    if (latest?.event_id && id !== lastCrop && latest.status === 'accepted') { const image = $('#visual-crop'); image.hidden = false; image.src = `/api/inference/crop.jpg?t=${Date.now()}`; if (initialized && latest.prediction === 'right') window.GripAlerts.tone('reject',id); else window.GripAlerts.remember(id); lastCrop = id; }
    initialized = true;
    const fault = `${status.job_id}:${status.fault || status.error || ''}`;
    if ((status.preflight?.errors?.length || status.fault) && fault !== lastFault) { lastFault = fault; window.GripJobs.fail(status.preflight || {checks:status.checks}, $('#visual-form'), status.fault); }
  }
  async function poll() { if (document.body.dataset.local !== 'true' || polling) return; polling = true; try {render(await api('/api/inference/status'));} catch(error) { $('#visual-status').textContent = `Inference connection unavailable: ${error.message}`; } finally {polling = false;} }
  $('#visual-form').addEventListener('submit',async event => {event.preventDefault(); const form = event.currentTarget; const body = Object.fromEntries(new FormData(form)); body.amp = form.elements.amp.checked; body.mode = 'shadow'; $('#visual-status').textContent = 'PREFLIGHT'; try { const payload = await api('/api/inference/start',body); session = payload.session_id || payload.job_id || ''; initialized = false; await poll(); } catch(error) {window.GripJobs.fail(error.report,form,error.message);} });
  function action(id, path, body) { $(id).addEventListener('click', async () => {try {await api(path, typeof body === 'function' ? body() : body); await poll();} catch(error) {window.GripJobs.fail(error.report,$('#visual-form'),error.message);} }); }
  action('#visual-stop','/api/inference/stop',{}); action('#visual-simulate','/api/inference/simulate-trigger',{});
  action('#visual-pause','/api/inference/playback',() => {paused = !paused; $('#visual-pause').textContent = paused ? 'Resume' : 'Pause'; return {paused,speed:Number($('#visual-speed').value)};});
  $('#visual-speed').addEventListener('change',() => api('/api/inference/playback',{paused,speed:Number($('#visual-speed').value)}).catch(error => window.GripJobs.fail(error.report,$('#visual-form'),error.message)));
  window.setInterval(poll,700); poll();
})();
