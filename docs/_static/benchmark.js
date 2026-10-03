/* Homepage comparison. Values, options, and labels are in benchmarks.json. */
(() => {
  "use strict";

  const root = document.getElementById("flow-benchmark");
  if (!root) return;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function number(value, decimals) {
    return value.toFixed(decimals);
  }

  function difference(value, reference, decimals) {
    const delta = value - reference;
    const rounded = Number(delta.toFixed(decimals));
    return `${rounded < 0 ? "-" : rounded > 0 ? "+" : ""}${number(Math.abs(rounded), decimals)}`;
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const animations = new WeakMap();

  function animateNumber(node, target, format, duration) {
    const previous = animations.get(node);
    if (previous) cancelAnimationFrame(previous.frame);
    const start = previous?.value ?? (node.classList.contains("flow-stat-value") ? 0 : target);
    const state = { value: start, frame: null };
    animations.set(node, state);
    if (reducedMotion.matches || start === target) {
      state.value = target;
      node.textContent = format(target);
      return;
    }
    const began = performance.now();
    function tick(now) {
      const progress = Math.min(1, (now - began) / duration);
      state.value = start + (target - start) * (1 - (1 - progress) ** 3);
      node.textContent = format(state.value);
      if (progress < 1) state.frame = requestAnimationFrame(tick);
    }
    state.frame = requestAnimationFrame(tick);
  }

  function choiceControl(label, options, selected, id) {
    const wrapper = element("fieldset", "flow-control");
    wrapper.id = id;
    wrapper.append(element("legend", null, label));
    const choices = element("div", "flow-choices");
    if (!options.some(option => option.id === selected)) throw new Error(`Invalid default for ${id}`);
    for (const option of options) {
      const choice = element("label", "flow-choice");
      const input = element("input");
      input.type = "radio";
      input.name = id;
      input.value = option.id;
      input.checked = option.id === selected;
      const caption = element("span", null);
      caption.textContent = option.label;
      choice.append(input, caption);
      choices.append(choice);
    }
    wrapper.append(choices);
    return { wrapper, value: () => wrapper.querySelector("input:checked").value };
  }

  function configurationControl(label, options, selected, models) {
    const wrapper = element("fieldset", "flow-control flow-routing-slider");
    wrapper.id = "flow-configuration";
    wrapper.append(element("legend", null, label));
    const zones = element("div", "flow-model-zones");
    zones.setAttribute("aria-hidden", "true");
    for (const model of models) {
      const zone = element("div", "flow-model-zone");
      zone.dataset.model = model.id;
      zones.append(zone);
    }
    wrapper.append(zones);
    const groups = element("div", "flow-configuration-groups");
    for (const model of models) {
      const group = element("span", null, model.shortLabel || model.label);
      group.dataset.model = model.id;
      groups.append(group);
    }
    const selectedLabel = element("span", "flow-configuration-selected");
    wrapper.append(selectedLabel, groups);
    const range = element("input", "flow-route-range");
    range.type = "range";
    range.min = 0;
    range.max = options.length - 1;
    range.step = 1;
    range.value = options.findIndex(option => option.id === selected);
    range.setAttribute("aria-label", label);
    const rail = element("div", "flow-route-rail");
    const fill = element("div", "flow-route-fill");
    const thumb = element("div", "flow-route-thumb");
    fill.setAttribute("aria-hidden", "true");
    thumb.setAttribute("aria-hidden", "true");
    rail.append(fill, thumb, range);
    const ticks = element("div", "flow-route-ticks");
    const buttons = options.map((option, index) => {
      const button = element("button", null, option.profileLabel);
      button.type = "button";
      button.style.setProperty("--flow-stop-position", `${100 * index / (options.length - 1)}%`);
      button.title = option.label;
      button.setAttribute("aria-label", option.label);
      button.addEventListener("click", () => {
        range.value = index;
        range.dispatchEvent(new Event("input", { bubbles: true }));
      });
      ticks.append(button);
      return button;
    });
    function sync() {
      const index = Number(range.value);
      range.setAttribute("aria-valuetext", options[index].label);
      rail.style.setProperty("--flow-route-progress", `${100 * (index + .5) / options.length}%`);
      buttons.forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
      selectedLabel.textContent = options[index].label;
      for (const group of groups.children) {
        group.dataset.active = String(group.dataset.model === options[index].model);
      }
      for (const zone of zones.children) {
        zone.dataset.active = String(zone.dataset.model === options[index].model);
      }
    }
    range.addEventListener("input", sync);
    sync();
    wrapper.append(rail, ticks);
    return { wrapper, event: "input", value: () => options[Number(range.value)].id };
  }

  function particleLayer(settings, baseline) {
    const layer = element("span", "flow-particles");
    const width = settings.tileWidthPx;
    const color = baseline ? "#ffffff" : "#343080";
    const dots = Array.from({ length: settings.count }, (_, index) => {
      // Place one random dot per horizontal band to limit empty gaps.
      const x = (index + Math.random()) * width / settings.count;
      const y = 3 + Math.random() * 18;
      const radius = 1 + Math.random() * .8;
      const opacity = .55 + Math.random() * .45;
      return `<circle cx="${x}" cy="${y}" r="${radius}" opacity="${opacity}"/>`;
    }).join("");
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="24" viewBox="0 0 ${width} 24"><g fill="${color}">${dots}</g></svg>`;
    layer.style.backgroundImage = `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
    layer.style.setProperty("--flow-particle-tile", `${width}px`);
    return layer;
  }

  function metricRow(metric, baseline, animation) {
    const row = element("div", "flow-metric");
    row.dataset.metric = metric.id;
    const heading = element("div", "flow-metric-heading");
    const label = element("span", "flow-metric-label", metric.label);
    label.title = metric.description;
    const value = element("span", "flow-metric-value");
    heading.append(label, value);
    const track = element("div", "flow-bar-track");
    track.setAttribute("aria-hidden", "true");
    const bar = element("div", "flow-bar");
    if (metric.id === "speed") {
      bar.classList.add("flow-speed-flow");
      bar.append(particleLayer(animation.particles, baseline));
    }
    track.append(bar);
    const delta = element("div", "flow-delta", baseline ? "1x baseline" : "");
    if (baseline) delta.classList.add("flow-baseline-delta");
    row.append(heading, track, delta);
    return { row, value, bar, delta };
  }

  function panel(title, description, metrics, baseline, animation) {
    const node = element("div", baseline ? "flow-panel flow-baseline" : "flow-panel flow-candidate");
    const heading = element("h3", null, title);
    const subtitle = element("p", "flow-panel-description", description);
    const header = element("div", "flow-panel-header");
    header.append(heading, subtitle);
    node.append(header);
    node.style.gridRow = `span ${metrics.length + 1}`;
    const rows = new Map();
    for (const metric of metrics) {
      const row = metricRow(metric, baseline, animation);
      rows.set(metric.id, row);
      node.append(row.row);
    }
    return { node, subtitle, rows };
  }

  function render(config) {
    const { copy, defaults, metrics, datasets, models, routing } = config;
    if (!metrics.length || !datasets.length || !models.length || !routing.length) {
      throw new Error("Empty benchmark configuration");
    }
    // Check all results before rendering so every slider position can be displayed.
    for (const dataset of datasets) {
      if (!dataset.results[dataset.baseline.model]?.[dataset.baseline.routing]) {
        throw new Error(`Missing baseline for ${dataset.id}`);
      }
      for (const model of models) {
        for (const profile of routing) {
          for (const metric of metrics) {
            const value = dataset.results[model.id]?.[profile.id]?.[metric.id];
            if (!Number.isFinite(value) || value < 0 ||
                (metric.scale && value > metric.scale) ||
                (metric.comparison === "ratio" && value <= 0)) {
              throw new Error(`Invalid ${dataset.id}/${model.id}/${profile.id}/${metric.id}`);
            }
          }
        }
      }
    }

    const header = element("div", "flow-benchmark-header");
    header.append(element("h2", null, copy.title), element("p", null, copy.description));
    const controls = element("div", "flow-controls");
    const datasetControl = choiceControl(copy.datasetLabel, datasets, defaults.dataset, "flow-dataset");
    const configurations = config.configurations.map(pair => {
      const model = models.find(item => item.id === pair.model);
      const profile = routing.find(item => item.id === pair.routing);
      if (!model || !profile) throw new Error("Invalid configuration pair");
      return { ...pair, id: `${pair.model}/${pair.routing}`,
        label: `${model.label} , ${profile.label}`, profileLabel: profile.label };
    });
    const configurationChoice = configurationControl(copy.configurationLabel, configurations,
      `${defaults.model}/${defaults.routing}`, models);
    header.append(datasetControl.wrapper);
    controls.append(configurationChoice.wrapper);
    const summary = element("div", "flow-summary");
    const speedStat = element("div", "flow-stat");
    const speedValue = element("span", "flow-stat-value");
    speedStat.append(speedValue, element("span", "flow-stat-label", copy.speedCaption));
    const recallStat = element("div", "flow-stat");
    const recallValue = element("span", "flow-stat-value flow-stat-recall");
    recallStat.append(recallValue, element("span", "flow-stat-label", copy.recallCaption));
    const precisionStat = element("div", "flow-stat");
    const precisionValue = element("span", "flow-stat-value flow-stat-precision");
    precisionStat.append(precisionValue, element("span", "flow-stat-label", copy.precisionCaption));
    summary.append(speedStat, recallStat, precisionStat);
    const announcement = element("span", "flow-sr-only");
    announcement.setAttribute("role", "status");
    announcement.setAttribute("aria-live", "polite");
    announcement.setAttribute("aria-atomic", "true");
    summary.append(announcement);
    // Screen readers receive final values instead of every animation frame.
    speedStat.setAttribute("aria-hidden", "true");
    recallStat.setAttribute("aria-hidden", "true");
    precisionStat.setAttribute("aria-hidden", "true");
    const panels = element("div", "flow-panels");
    const baseline = panel(copy.baselineTitle, "", metrics, true, config.animation);
    const candidate = panel(copy.candidateTitle, "", metrics, false, config.animation);
    candidate.subtitle.before(element("span", "flow-pipeline-label", copy.routedModelLabel));
    panels.append(baseline.node, candidate.node);
    const note = element("p", "flow-benchmark-note", copy.deltaNote);
    const method = element("details", "flow-method");
    const methodCaption = element("p", "flow-method-caption");
    method.append(element("summary", null, copy.methodTitle), methodCaption);
    root.replaceChildren(header, summary, controls, panels, note, method);
    const duration = config.animation.durationMs;
    root.style.setProperty("--flow-duration", `${duration}ms`);
    root.style.setProperty("--flow-routing-duration", `${config.animation.routingDurationMs}ms`);
    // Apply the initial bar widths after layout so their transitions can run.
    root.getBoundingClientRect();

    function update() {
      const dataset = datasets.find(item => item.id === datasetControl.value());
      const configuration = configurations.find(item => item.id === configurationChoice.value());
      const model = models.find(item => item.id === configuration.model);
      const profile = routing.find(item => item.id === configuration.routing);
      const reference = dataset.results[dataset.baseline.model][dataset.baseline.routing];
      const selected = dataset.results[model.id][profile.id];
      methodCaption.textContent = `${dataset.description} ${copy.methodText} ${profile.description} ${copy.comparisonNote}`;
      candidate.subtitle.replaceChildren(
        element("span", "flow-model-name", model.label),
        element("span", "flow-routing-badge", copy.routingBadgeTemplate.replace("{profile}", profile.label))
      );
      const baselineModel = models.find(item => item.id === dataset.baseline.model);
      baseline.subtitle.textContent = `${baselineModel.label} , ${copy.baselineRoutingLabel}`;

      for (const metric of metrics) {
        const maximum = metric.scale || Math.max(...models.flatMap(m =>
          routing.map(r => dataset.results[m.id][r.id][metric.id])));
        for (const [target, values] of [[baseline, reference], [candidate, selected]]) {
          const row = target.rows.get(metric.id);
          animateNumber(row.value, values[metric.id], value => `${number(value, metric.decimals)}${metric.unit}`, duration);
          row.bar.style.width = `${100 * values[metric.id] / maximum}%`;
          if (metric.id === "speed") {
            // Adjust travel time for tile width to keep speed tied to the multiplier.
            const multiplier = values.speed / reference.speed;
            const particles = config.animation.particles;
            row.bar.style.setProperty("--flow-flow-duration",
              `${config.animation.flowBaselineMs * particles.tileWidthPx / 32 / multiplier}ms`);
            const gain = Math.max(0, Math.min(1,
              (multiplier - 1) / (particles.fullOpacityAt - 1)));
            row.bar.style.setProperty("--flow-particle-opacity",
              particles.minOpacity + gain * (particles.maxOpacity - particles.minOpacity));
          }
        }
        const delta = candidate.rows.get(metric.id).delta;
        delta.textContent = metric.comparison === "ratio"
          ? `${number(selected[metric.id] / reference[metric.id], metric.deltaDecimals)}x`
          : `${difference(selected[metric.id], reference[metric.id], metric.deltaDecimals)} pp`;
        delta.dataset.direction = selected[metric.id] < reference[metric.id] ? "lower" : "higher";
      }
      const recall = metrics.find(metric => metric.id === "recall");
      const speed = metrics.find(metric => metric.id === "speed");
      const precision = metrics.find(metric => metric.id === "precision");
      const ratio = selected.speed / reference.speed;
      animateNumber(speedValue, ratio, value => `${number(value, speed.deltaDecimals)}x`, duration);
      animateNumber(recallValue, selected.recall - reference.recall,
        value => `${difference(value, 0, recall.deltaDecimals)} pp`, duration);
      animateNumber(precisionValue, selected.precision - reference.precision,
        value => `${difference(value, 0, precision.deltaDecimals)} pp`, duration);
      announcement.textContent = copy.summaryTemplate
        .replace("{speed}", number(selected.speed / reference.speed, speed.deltaDecimals))
        .replace("{recall}", difference(selected.recall, reference.recall, recall.deltaDecimals))
        .replace("{precision}", difference(selected.precision, reference.precision, precision.deltaDecimals));
      if (!reducedMotion.matches) {
        // Cancel the previous flash when choices change rapidly.
        candidate.node.getAnimations().forEach(animation => animation.cancel());
        candidate.node.animate([
          { boxShadow: "0 0 0 0 rgb(150 140 240 / 45%)" },
          { boxShadow: "0 0 0 10px rgb(150 140 240 / 0%)" },
        ], { duration, easing: "ease-out" });
      }
    }

    for (const control of [datasetControl, configurationChoice]) {
      control.wrapper.addEventListener(control.event || "change", update);
    }
    update();
    root.dataset.ready = "true";
  }

  fetch(root.dataset.config)
    .then(response => {
      if (!response.ok) throw new Error(`Benchmark request failed: ${response.status}`);
      return response.json();
    })
    .then(render)
    .catch(error => {
      root.textContent = "The comparison could not be loaded. Please reload this page.";
      console.error("PseudoPath benchmark:", error);
    });
})();
