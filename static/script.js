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

    // clear old route visuals
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    lastData = null;


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
  }

  // === METRICS VISUALIZATION ===
  let bunchHistory = [];
  const mCanvas = document.getElementById('metricsChart');
  const mCtx = mCanvas.getContext('2d');

  async function fetchMetrics() {
    try {
      const res = await fetch('/metrics');
      const m = await res.json();

      // textual display
      document.getElementById('metricsData').innerHTML = `
        <b>Bunch Score:</b> ${m.bunch_score.toFixed(2)}<br>
        Headway Score: ${m.headway_score.toFixed(2)}<br>
        Cluster Score: ${m.cluster_score.toFixed(2)}<br>
        Bunched Buses: ${m.bunch_count}<br>
        CAV: ${m.cac_count} | BUSCH: ${m.liv_count}<br>
        <small>Bunch Ratio: ${(m.bunch_ratio * 100).toFixed(1)}% • Cluster Ratio: ${(m.cluster_ratio * 100).toFixed(1)}%</small>
      `;

      // update chart history
      bunchHistory.push(m.bunch_score);
      if (bunchHistory.length > 100) bunchHistory.shift();

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

    if (bunchHistory.length < 2) return;

    // axes
    mCtx.strokeStyle = '#ccc';
    mCtx.beginPath();
    mCtx.moveTo(0, h - 20);
    mCtx.lineTo(w, h - 20);
    mCtx.moveTo(0, 0);
    mCtx.lineTo(0, h);
    mCtx.stroke();

    // line plot
    mCtx.strokeStyle = '#0074D9';
    mCtx.lineWidth = 2;
    mCtx.beginPath();

    const max = 1.0;
    bunchHistory.forEach((val, i) => {
      const x = (i / (bunchHistory.length - 1)) * w;
      const y = h - (val / max) * (h - 20);
      if (i === 0) mCtx.moveTo(x, y);
      else mCtx.lineTo(x, y);
    });
    mCtx.stroke();

    // title
    mCtx.fillStyle = '#222';
    mCtx.font = '12px sans-serif';
    mCtx.fillText('Bunch Score (0–1)', 5, 12);
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
      } catch (err) {
        console.error('Failed to set PDF type:', err);
      }
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
