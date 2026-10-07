/* Presentation only: retain existing controls, values and backend contracts. */
(() => {
  'use strict';
  const $ = selector => document.querySelector(selector);
  const sections = [...document.querySelectorAll('main > .page-section')];
  const links = [...document.querySelectorAll('.nav-link')];

  function refresh() {
    const id = location.hash.slice(1);
    const target = document.getElementById(id);
    let selected = target?.closest('.page-section') || $('#overview');
    if (selected.classList.contains('host-only') && document.body.dataset.local !== 'true') selected = $('#overview');
    sections.forEach(section => { section.hidden = section !== selected; });
    links.forEach(link => {
      const active = link.hash === `#${selected.id}`;
      link.classList.toggle('active', active);
      if (active) {
        link.setAttribute('aria-current', 'page');
      } else link.removeAttribute('aria-current');
    });
    const tools = $('.nav-tools');
    const secondary = Boolean(links.find(link => link.hash === `#${selected.id}`)?.closest('.nav-tools'));
    tools.classList.toggle('active', secondary);
    if (!secondary) tools.open = false;
    $('.skip-link').href = `#${selected.id}`;
  }

  function reveal(element) {
    const section = element?.closest('.page-section');
    if (!section || (section.classList.contains('host-only') && document.body.dataset.local !== 'true')) return;
    location.hash = section.id;
    refresh();
    for (let parent = element.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS') parent.open = true;
    }
  }

  function fold(parent, title, nodes, grid = 'form-grid') {
    const items = [...new Set(nodes.filter(Boolean))];
    if (!items.length) return;
    const details = document.createElement('details');
    details.className = 'options';
    const summary = document.createElement('summary');
    summary.textContent = title;
    const content = document.createElement('div');
    content.className = grid;
    content.append(...items);
    details.append(summary, content);
    parent.append(details);
    return details;
  }

  function fields(parent, title, selectors) {
    return fold(parent, title, selectors.map(selector => {
      const control = parent.querySelector(selector);
      return control?.closest('label, .form-actions') || control;
    }));
  }

  const training = $('[data-action="train"]');
  fields(training, 'Advanced training settings', [
    '[name="head_only_epochs"]', '[name="backbone_learning_rate"]', '[name="image_size"]',
    '[name="learning_rate"]', '[name="validation_fraction"]', '[name="seed"]',
    '[name="workers"]', '[name="loss"]', '[name="recall_target"]', '[name="recall_weight"]',
    '[name="selection_metric"]', '[name="augmentation"]', '[name="tensorboard_logdir"]', '[name="amp"]',
  ]);
  training.append(training.querySelector(':scope > .form-actions'));
  const trainPanel = training.closest('.panel');
  fold(trainPanel, 'Dataset audit details', [$('#audit-dataset').closest('.form-actions'), $('#audit-output').closest('label'), $('#dataset-summary'), trainPanel.querySelector(':scope > .field-help')]);

  const visual = $('#visual-form');
  fields(visual, 'Device and decision settings', ['[name="device"]', '[name="decision_class"]', '[name="decision_threshold"]', '[name="amp"]']);
  visual.append(visual.querySelector(':scope > .form-actions'));
  const inference = $('#infer');
  fold(inference, 'Batch inference and file exports', [inference.querySelector(':scope > .two-column'), inference.querySelector(':scope > article:not(#visualization)')], '');
  fold(inference, 'Inference field guide', [inference.querySelector(':scope > .field-help')], '');
  fold($('#visualization'), 'Session diagnostics', [$('#visual-checks'), $('#visual-detail'), $('#visual-crop')], '');

  const controls = $('.factory-controls');
  fields(controls, 'Camera settings', ['#factory-camera-preset', '#factory-camera-fps', '#factory-camera-custom']);
  fields(controls, 'Model and device details', ['#factory-models-root', '#factory-checkpoint-meta', '#factory-sha256', '#factory-device', '#factory-cuda-status', '#factory-amp']);
  fields(controls, 'Decision and geometry settings', ['#factory-reject-class', '#factory-decision-class', '#factory-decision-threshold', '#factory-direction-options', '#factory-trigger-enabled', '#factory-trigger-fraction', '#factory-grip-geometry', '#factory-geometry-note']);
  fields(controls, 'Actuator setup · ARMED only', ['#factory-serial-port', '#factory-baud', '#factory-delay', '#factory-serial-status', '#factory-connect']);
  fields(controls, 'Session options', ['#factory-show-rejected', '#factory-audio', '#factory-continue-counters']);
  controls.append($('#factory-start').closest('.form-actions'));
  $('#factory-direction-options').replaceWith($('#factory-direction-options label'));
  $('#factory-diagnostics').append($('#factory-startup-checks'), $('#factory-capture-info'), $('#factory-yolo-counts'), $('#factory-position'));
  const metrics = $('#factory > .metric-grid');
  fold($('#factory'), 'Detailed latency and rejection metrics', ['count-pipeline', 'count-multiple', 'count-partial', 'metric-capture-fps', 'metric-yolo', 'metric-classifier', 'metric-event', 'metric-accepted-latency'].map(id => document.getElementById(id).closest('.metric-card')), 'metric-grid');
  // Keep the main counts immediately after the decision, before diagnostics.
  $('#factory-decision').closest('.factory-layout').after(metrics);

  document.addEventListener('invalid', event => reveal(event.target), true);
  window.addEventListener('hashchange', () => {
    refresh();
    if (window.matchMedia?.('(max-width: 760px)')?.matches) $('.nav-tools').open = false;
    const target = document.getElementById(location.hash.slice(1));
    const heading = target?.querySelector('h1');
    // A preflight fix may already have focused an input before hashchange fires.
    if (heading && !target.contains(document.activeElement)) {
      heading.tabIndex = -1;
      heading.focus({preventScroll: true});
      window.scrollTo({top: 0, behavior: 'instant'});
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') $('.nav-tools').open = false;
  });
  window.GripInterface = {refresh, reveal};
  refresh();
})();
