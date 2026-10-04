(() => {
  'use strict';
  let context, enabled = false, cooldown = 0;
  const seen = new Set();
  const preferences = (() => { try { return JSON.parse(localStorage.getItem('grip-alerts') || '{}'); } catch (_) { return {}; } })();
  function remember(key) { if (seen.has(key)) return false; seen.add(key); if (seen.size > 600) seen.delete(seen.values().next().value); return true; }
  function tone(kind, key) {
    if (!remember(key) || !enabled || context?.state !== 'running') return;
    if (kind === 'warning' && Date.now() < cooldown) return;
    cooldown = Date.now() + 2500;
    const count = kind === 'error' ? 3 : kind === 'success' ? 2 : 1;
    for (let i = 0; i < count; i++) {
      const oscillator = context.createOscillator(), gain = context.createGain();
      const start = context.currentTime + i * .18;
      oscillator.frequency.value = kind === 'reject' ? 1100 : kind === 'error' ? 330 : 660;
      gain.gain.setValueAtTime(Math.max(.001, Number(preferences.volume ?? .08)), start);
      gain.gain.exponentialRampToValueAtTime(.001, start + .12);
      oscillator.connect(gain).connect(context.destination); oscillator.start(start); oscillator.stop(start + .14);
    }
  }
  async function enable() {
    context ||= new (window.AudioContext || window.webkitAudioContext)();
    await context.resume(); // Must occur directly inside a user gesture, including on mobile.
    enabled = context.state === 'running';
    document.querySelector('#enable-alerts').textContent = enabled ? 'Alerts enabled' : 'Enable alerts';
  }
  const button = document.createElement('button'); button.id = 'enable-alerts'; button.type = 'button'; button.className = 'button'; button.textContent = 'Enable alerts';
  button.addEventListener('click', () => enable().catch(error => { button.textContent = `Audio unavailable: ${error.message}`; }));
  document.querySelector('.top-status').append(button);
  const volume = document.createElement('input'); volume.type = 'range'; volume.min = '0'; volume.max = '.2'; volume.step = '.01'; volume.value = String(preferences.volume ?? .08); volume.setAttribute('aria-label', 'Alert volume'); volume.className = 'alert-volume';
  volume.addEventListener('input', () => { preferences.volume = Number(volume.value); localStorage.setItem('grip-alerts', JSON.stringify(preferences)); }); button.after(volume);
  window.GripAlerts = { tone, remember, enable };
})();
