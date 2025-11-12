// wait for HTML document to be fully parsed;
window.addEventListener("DOMContentLoaded", () => {
  console.log("Script loaded, DOM ready");
  const canvas = document.getElementById("sim");
  const ctx = canvas.getContext("2d");

  // Remove legacy Bus Log UI and stack Stop Means under Bus Stop Progress in its own column
  const busLogBox = document.getElementById('busLogBox');
  if (busLogBox && busLogBox.remove) busLogBox.remove();
  const busStatusBox = document.getElementById('busStatusBox');
  const stopMeansBox = document.getElementById('stopMeansBox');
  const containerEl = document.querySelector('.container');
  if (containerEl && busStatusBox) {
    let col = document.getElementById('progressColumn');
    if (!col) {
      col = document.createElement('div');
      col.id = 'progressColumn';
    }
    // Insert the column before the current busStatusBox position
    containerEl.insertBefore(col, busStatusBox);
    col.appendChild(busStatusBox);
    if (stopMeansBox) col.appendChild(stopMeansBox);
  }

  let lastData = null;
  const timerStatusEl = document.getElementById('timerStatus');
  const timerMinutesInput = document.getElementById('timerMinutes');
  const startTimerBtn = document.getElementById('startTimerBtn');
  const stopTimerBtn = document.getElementById('stopTimerBtn');

  // Drawing loop
  async function drawFrame() {
      try {
          const res = await fetch("/positions");
          const data = await res.json();
          lastData = data;
          render(data);
      } catch (err) {
          console.error("Fetch error:", err);
      }
      // Refresh every 60ms (~16fps)
      setTimeout(drawFrame, 60);
  }

  function render(data) {
      const route = data.path;
      const stops = data.stops;
      const buses = data.buses;
      const norm = data.norm || route.map((_, i) => i / (route.length - 1));
      updateTimerStatus(data.timer);

      // temp console log sanity check
      console.log({
        hasRoute: !!data.path,
        hasNorm: !!data.norm,
        routeLen: data.path ? data.path.length : 0,
        normLen: data.norm ? data.norm.length : 0
      });


      // fall back error
      if (!route.length || !buses.length) return;

    

      ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Route path
      ctx.beginPath();
      ctx.moveTo(route[0][0], route[0][1]);
      for (let i = 1; i < route.length; i++) ctx.lineTo(route[i][0], route[i][1]);
      ctx.strokeStyle = "#0074D9";
      ctx.lineWidth = 3;
      ctx.stroke();

      // Stops
    stops.forEach(stop => {
      ctx.beginPath();
      ctx.arc(stop.x, stop.y, 8, 0, 2 * Math.PI);
      ctx.fillStyle = "orange";
      ctx.fill();
      ctx.stroke();

      ctx.font = "12px sans-serif";
      ctx.fillStyle = "#222";
      const label = stop.name || `Stop ${stop.id}`;
      ctx.fillText(label, stop.x + 12, stop.y + 3);
    });

    // Buses
    buses.forEach((bus, i) => {
      const [x, y] = interpolateByDistance(route, data.norm, bus.pos);
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = `hsl(${i * 70}, 80%, 50%)`;
      ctx.fill();

      ctx.font = "10px sans-serif";
      ctx.fillStyle = "#111";
      ctx.fillText(`Bus ${bus.id}`, x + 10, y);
    });
  }

  async function drawFrame() {
    try {
      const res = await fetch("/positions");
      const data = await res.json();
      render(data);
      renderBusStatus(data.buses);
    } catch (err) {
      console.error("Fetch error:", err);
    }
    setTimeout(drawFrame, 60);
  }

  // progress bar
  function renderBusStatus(buses) {
    const container = document.getElementById("bus-status");
    container.innerHTML = "";

    buses.forEach(bus => {
      const div = document.createElement("div");
      div.className = "bus-entry";

      const status = bus.status;
      let content = `<strong>${bus.id}</strong>: `;

      if (!status) {
        content += `<span>(moving)</span>`;
      } else {
        const barWidth = Math.floor(status.progress * 100);
        content += `<span>${status.stop_name}</span>
          <div class="progress">
            <div class="progress-bar" style="width:${barWidth}%"></div>
          </div>
          <small>${(status.progress * 100).toFixed(0)}% (${status.remaining}s left)</small>`;
      }

      div.innerHTML = content;
      container.appendChild(div);
    });
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds)) return null;
    const total = Math.max(0, Math.floor(seconds));
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  function updateTimerStatus(timer) {
    if (!timerStatusEl) return;
    if (!timer || typeof timer !== 'object') {
      // No new data—preserve whatever we were showing.
      return;
    }
    if (timer.active && Number.isFinite(timer.remaining)) {
      const label = formatDuration(timer.remaining) || '...';
      timerStatusEl.textContent = `Running – ${label} remaining`;
    } else if (timer.finished) {
      timerStatusEl.textContent = 'Timer finished';
    } else {
      timerStatusEl.textContent = 'Timer off';
    }
  }

  function resetVisualizationState() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    lastData = null;
    bunchHistory = [];
    avgBunchHistory = [];
    drawMetricsChart();
  }


  // dynamic route selector!
  const routeSelector = document.getElementById("routeSelector");
  routeSelector.addEventListener("change", async (e) => {
    const newRoute = e.target.value;
    console.log("Switching to route:", newRoute);

    // send to backend
    await fetch("/set_route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ route_id: newRoute }),
    });

    resetVisualizationState();

    // optionally refresh immediately
    const res = await fetch("/meta");
    const meta = await res.json();
    document.getElementById("simTitle").innerText =
      `Bus Bunching Simulation — Route ${meta.route_id}`;
  });

  function interpolateByDistance(route, norm, t) {
      if (!route || !norm || route.length < 2 || norm.length < 2) {
          return [0, 0];
      }

      const target = t % 1.0;
      for (let i = 0; i < norm.length - 1; i++) {
          if (target >= norm[i] && target <= norm[i + 1]) {
          const frac = (target - norm[i]) / (norm[i + 1] - norm[i]);
          const [x1, y1] = route[i];
          const [x2, y2] = route[i + 1];
          return [x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac];
          }
      }
      return route[route.length - 1];
  }

  // meta and events, fetching.
  async function fetchMeta() {
      const res = await fetch('/meta');
      const meta = await res.json();
      document.getElementById('simTitle').innerText =
          `Bus Bunching Simulation for ${meta.route_id}`;
      document.getElementById('speedLabel').innerText =
          meta.speed_factor.toFixed(1) + "x";
      const busCountEl = document.getElementById('busCount');
      if (busCountEl && typeof meta.num_buses === 'number') {
        busCountEl.value = meta.num_buses;
      }
      updateTimerStatus(meta.timer);
  }

  // === METRICS VISUALIZATION ===
  let bunchHistory = [];
  let avgBunchHistory = [];
  const mCanvas = document.getElementById('metricsChart');
  const mCtx = mCanvas.getContext('2d');

  async function fetchMetrics() {
    try {
      const res = await fetch('/metrics');
      const m = await res.json();

      // textual display
      const bunchScore = Number.isFinite(m.bunch_score) ? m.bunch_score : 0;
      const avgBunchScore = Number.isFinite(m.bunch_score_avg) ? m.bunch_score_avg : bunchScore;
      const headwayScore = Number.isFinite(m.headway_score) ? m.headway_score : 0;
      const clusterScore = Number.isFinite(m.cluster_score) ? m.cluster_score : 0;
      const bunchCount = Number.isFinite(m.bunch_count) ? m.bunch_count : 0;
      const cavCount = Number.isFinite(m.cac_count) ? m.cac_count : 0;
      const livCount = Number.isFinite(m.liv_count) ? m.liv_count : 0;
      const bunchRatio = Number.isFinite(m.bunch_ratio) ? m.bunch_ratio : 0;
      const clusterRatio = Number.isFinite(m.cluster_ratio) ? m.cluster_ratio : 0;

      document.getElementById('metricsData').innerHTML = `
        <b>Bunch Score:</b> ${bunchScore.toFixed(2)}<br>
        Avg Bunch Score: ${avgBunchScore.toFixed(2)}<br>
        Headway Score: ${headwayScore.toFixed(2)}<br>
        Cluster Score: ${clusterScore.toFixed(2)}<br>
        Bunched Buses: ${bunchCount}<br>
        <small>Bunch Ratio: ${(bunchRatio * 100).toFixed(1)}% • Cluster Ratio: ${(clusterRatio * 100).toFixed(1)}%</small>
      `;

      // update chart history
      bunchHistory.push(bunchScore);
      avgBunchHistory.push(avgBunchScore);
      if (bunchHistory.length > 100) bunchHistory.shift();
      if (avgBunchHistory.length > 100) avgBunchHistory.shift();

      drawMetricsChart();

      // update long stop box
      const lsb = document.getElementById('longStopContent');
      if (lsb && typeof m.long_stop_count !== 'undefined') {
        lsb.textContent = `Total: ${m.long_stop_count}`;
      }
    } catch (err) {
      console.error('Metrics fetch failed:', err);
    }
  }

  function drawMetricsChart() {
    const w = mCanvas.width;
    const h = mCanvas.height;
    mCtx.clearRect(0, 0, w, h);

    if (bunchHistory.length < 2 && avgBunchHistory.length < 2) return;

    const padding = { top: 15, right: 20, bottom: 40, left: 50 };
    const axisLeft = padding.left;
    const axisTop = padding.top;
    const axisBottom = h - padding.bottom;
    const axisRight = w - padding.right;
    const plotWidth = axisRight - axisLeft;
    const plotHeight = axisBottom - axisTop;

    // axes
    mCtx.strokeStyle = '#ccc';
    mCtx.lineWidth = 1;
    mCtx.beginPath();
    mCtx.moveTo(axisLeft, axisTop);
    mCtx.lineTo(axisLeft, axisBottom);
    mCtx.lineTo(axisRight, axisBottom);
    mCtx.stroke();

    // axis ticks
    mCtx.fillStyle = '#555';
    mCtx.font = '11px sans-serif';
    const yTicks = [0, 0.5, 1];
    yTicks.forEach((val) => {
      const y = axisBottom - (val * plotHeight);
      mCtx.beginPath();
      mCtx.moveTo(axisLeft - 5, y);
      mCtx.lineTo(axisLeft, y);
      mCtx.stroke();
      mCtx.fillText(val.toFixed(1), axisLeft - 35, y + 4);
    });

    const maxPoints = Math.max(bunchHistory.length, avgBunchHistory.length);
    const xTicks = [0, 0.5, 1];
    xTicks.forEach((frac) => {
      const x = axisLeft + frac * plotWidth;
      mCtx.beginPath();
      mCtx.moveTo(x, axisBottom);
      mCtx.lineTo(x, axisBottom + 5);
      mCtx.stroke();
      const idx = Math.round(frac * Math.max(maxPoints - 1, 0));
      mCtx.fillText(idx.toString(), x - 5, axisBottom + 18);
    });

    const max = 1.0;
    const series = [
      { data: bunchHistory, color: '#0074D9', label: 'Bunch Score' },
      { data: avgBunchHistory, color: '#FF4136', label: 'Avg Bunch Score' },
    ];

    series.forEach(({ data, color }) => {
      if (data.length < 2) return;
      mCtx.strokeStyle = color;
      mCtx.lineWidth = 2;
      mCtx.beginPath();
      const denom = Math.max(data.length - 1, 1);
      data.forEach((val, i) => {
        const clamped = Math.max(0, Math.min(max, val));
        const x = axisLeft + (i / denom) * plotWidth;
        const y = axisBottom - (clamped / max) * plotHeight;
        if (i === 0) mCtx.moveTo(x, y);
        else mCtx.lineTo(x, y);
      });
      mCtx.stroke();
    });

    // labels
    mCtx.fillStyle = '#222';
    mCtx.font = '12px sans-serif';
    mCtx.fillText('Bunch Score (0–1)', axisLeft, axisTop - 5);
    mCtx.textAlign = 'center';
    mCtx.fillText('Time (oldest → newest)', (axisLeft + axisRight) / 2, h - 5);
    mCtx.save();
    mCtx.translate(15, h / 2);
    mCtx.rotate(-Math.PI / 2);
    mCtx.fillText('Score', 0, 0);
    mCtx.restore();
    mCtx.textAlign = 'left';

    // legend
    series.forEach(({ color, label }, idx) => {
      const lx = axisRight - 120;
      const ly = axisTop + idx * 16;
      mCtx.fillStyle = color;
      mCtx.fillRect(lx, ly - 8, 12, 12);
      mCtx.fillStyle = '#222';
      mCtx.fillText(label, lx + 18, ly + 2);
    });
  }

  // === SPEED CONTROL ===
  const slider = document.getElementById('speedSlider');
  const speedVal = document.getElementById('speedLabel');

  slider.addEventListener('input', async () => {
    const newSpeed = parseFloat(slider.value);
    speedVal.textContent = newSpeed.toFixed(1) + "×";

    try {
      await fetch('/speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ factor: newSpeed })
      });
    } catch (err) {
      console.error('Failed to update speed:', err);
    }
  });

  // initial simulation load up

  async function startup() {
    await fetchMeta();
    drawFrame();
    setInterval(fetchMetrics, 1000);
    setInterval(fetchStopStats, 1500);
  }
  startup();

  // === PDF TYPE SELECTOR ===
  const pdfSelector = document.getElementById('pdfSelector');
  if (pdfSelector) {
    pdfSelector.addEventListener('change', async (e) => {
      const type = e.target.value;
      try {
        await fetch('/pdf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ type })
        });
        resetVisualizationState();
      } catch (err) {
        console.error('Failed to set PDF type:', err);
      }
    });
  }

  async function updateTimer(minutes) {
    try {
      const res = await fetch('/timer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ minutes })
      });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || 'Timer request failed');
      updateTimerStatus(payload.timer);
    } catch (err) {
      console.error('Failed to update timer:', err);
      if (timerStatusEl) timerStatusEl.textContent = 'Timer error';
    }
  }

  if (startTimerBtn) {
    startTimerBtn.addEventListener('click', async () => {
      const val = timerMinutesInput ? parseFloat(timerMinutesInput.value || '0') : 0;
      if (!Number.isFinite(val) || val <= 0) {
        if (timerStatusEl) timerStatusEl.textContent = 'Enter minutes > 0';
        return;
      }
      await updateTimer(val);
    });
  }

  if (stopTimerBtn) {
    stopTimerBtn.addEventListener('click', async () => {
      await updateTimer(0);
    });
  }

  // === BUS COUNT CONTROL ===
  const busCountEl = document.getElementById('busCount');
  if (busCountEl) {
    busCountEl.addEventListener('change', async () => {
      const count = Math.max(1, Math.min(30, parseInt(busCountEl.value || '1', 10)));
      busCountEl.value = count;
      try {
        await fetch('/buses', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ count })
        });
      } catch (err) {
        console.error('Failed to set bus count:', err);
      }
    });
  }

  // === STOP MEANS ===
  async function fetchStopStats() {
    try {
      const res = await fetch('/stop_stats');
      const items = await res.json();
      const cont = document.getElementById('stopMeans');
      if (!cont) return;
      cont.innerHTML = '';
      items.forEach(st => {
        const div = document.createElement('div');
        div.className = 'stop-mean';
        const meanTheory = st.mean_theory != null ? st.mean_theory.toFixed(1) : '—';
        const meanSim = st.mean_sim != null ? st.mean_sim.toFixed(1) : '—';
        let diff = '—';
        if (st.diff != null) {
          const d = st.diff;
          const sign = d > 0 ? '+' : '';
          diff = `${sign}${d.toFixed(1)}`;
        }
        div.innerHTML = `
          <span class="label">${st.name}</span>
          <span class="vals">μ(theory): ${meanTheory} • μ(sim): ${meanSim} • Δ: ${diff}</span>
        `;
        cont.appendChild(div);
      });
    } catch (err) {
      console.error('Failed to fetch stop stats:', err);
    }
  }
});
