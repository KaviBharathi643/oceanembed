/**
 * OceanEmbed (SIH 2026 Problem Statement 26066)
 * Interactive Client Application Logic
 */

let map = null;
let currentMarker = null;
let profileChart = null;
let embeddingChart = null;
let currentPrediction = null;

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initMap();
  initControls();
  initGrid5x5();
  // Auto-run primary showcase location (Central Bay - 14.00N, 87.00E on 2024-04-06)
  const initLat = parseFloat(document.getElementById("input-lat")?.value) || 14.00;
  const initLon = parseFloat(document.getElementById("input-lon")?.value) || 87.00;
  const initDate = document.getElementById("explore-date-input")?.value || "2024-04-06";
  runPrediction(initLat, initLon, initDate);
});

/* ==========================================================================
   1. NAVIGATION & TAB SWITCHING
   ========================================================================== */
function initNavigation() {
  const navButtons = document.querySelectorAll(".nav-item");
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tabName = btn.getAttribute("data-tab");
      switchTab(tabName);
    });
  });
}

function switchTab(tabName) {
  // Update sidebar buttons
  document.querySelectorAll(".nav-item").forEach((btn) => {
    if (btn.getAttribute("data-tab") === tabName) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Update view sections
  document.querySelectorAll(".view-section").forEach((sec) => {
    sec.classList.remove("active");
  });

  const targetSec = document.getElementById(`view-${tabName}`);
  if (targetSec) {
    targetSec.classList.add("active");
  }

  // Handle tab-specific data loading
  if (tabName === "explore" && map) {
    setTimeout(() => map.invalidateSize(), 150);
  } else if (tabName === "predict") {
    updateStudioView();
  } else if (tabName === "datasources") {
    loadDataCatalog();
  } else if (tabName === "model") {
    loadModelSpecs();
  } else if (tabName === "validation") {
    loadArgoValidation();
  }
}

/* ==========================================================================
   2. LEAFLET INTERACTIVE OCEAN MAP
   ========================================================================== */
function initMap() {
  const mapElement = document.getElementById("map-container");
  if (!mapElement) return;

  // Bay of Bengal center and bounds
  const bobCenter = [14.5, 88.5];
  map = L.map("map-container", {
    center: bobCenter,
    zoom: 5,
    minZoom: 4,
    maxZoom: 9,
    maxBounds: [[4.0, 78.0], [26.0, 102.0]],
  });

  // Base marine tile layer (ESRI Ocean / CartoDB Voyager blend)
  const tileLayer = L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap',
    subdomains: "abcd",
    maxZoom: 19,
  });
  tileLayer.on("tileerror", () => {
    const hint = document.getElementById("map-tile-offline-hint");
    if (hint) hint.style.display = "block";
  });
  tileLayer.addTo(map);

  // Add subtle Bay of Bengal PoC region bounding box
  const pocBounds = [[5.0, 80.0], [25.0, 100.0]];
  L.rectangle(pocBounds, {
    color: "#0F5B78",
    weight: 1.5,
    dashArray: "4, 6",
    fillColor: "#0F5B78",
    fillOpacity: 0.04,
  }).addTo(map);

  // Custom marker pin
  const customIcon = L.divIcon({
    className: "custom-ocean-marker",
    html: `
      <div style="
        width: 26px; height: 26px;
        background: #E53935;
        border: 3px solid #FFFFFF;
        border-radius: 50% 50% 50% 0;
        transform: rotate(-45deg);
        box-shadow: 0 3px 10px rgba(0,0,0,0.35);
        display: flex; align-items: center; justify-content: center;
      ">
        <div style="width: 8px; height: 8px; background: #FFFFFF; border-radius: 50%;"></div>
      </div>
    `,
    iconSize: [26, 26],
    iconAnchor: [13, 26],
  });

  currentMarker = L.marker([14.00, 87.00], { icon: customIcon }).addTo(map);

  // Click on map to select coordinates
  map.on("click", (e) => {
    const lat = Math.round(e.latlng.lat * 4.0) / 4.0;
    const lon = Math.round(e.latlng.lng * 4.0) / 4.0;

    if (lat < 5.0 || lat > 25.0 || lon < 80.0 || lon > 100.0) {
      showToast(`Selected point (${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E) is outside the Bay of Bengal domain (5°N–25°N, 80°E–100°E).`, "warning");
      return;
    }

    document.getElementById("input-lat").value = lat.toFixed(2);
    document.getElementById("input-lon").value = lon.toFixed(2);

    updateMarker(lat, lon);

    // Deselect quick buttons
    document.querySelectorAll(".btn-quick-loc").forEach((b) => b.classList.remove("active"));

    const dateVal = (document.getElementById("explore-date-input")?.value || "").trim();
    if (dateVal) {
      runPrediction(lat, lon, dateVal);
    }
  });
}

function updateMarker(lat, lon) {
  if (currentMarker) {
    currentMarker.setLatLng([lat, lon]);
  }
}

/* ==========================================================================
   3. CONTROLS & EVENT LISTENERS
   ========================================================================== */
function initControls() {
  const dateInput = document.getElementById("explore-date-input");
  const displayDate = document.getElementById("display-date");

  dateInput.addEventListener("change", (e) => {
    const dVal = (e.target.value || "").trim();
    if (!dVal) {
      showToast("Please select a valid date in 2024.", "warning");
      return;
    }
    if (displayDate) displayDate.value = dVal;
    const rawLat = (document.getElementById("input-lat")?.value || "").trim();
    const rawLon = (document.getElementById("input-lon")?.value || "").trim();
    const lat = parseFloat(rawLat);
    const lon = parseFloat(rawLon);
    if (!isNaN(lat) && lat >= 5.0 && lat <= 25.0 && !isNaN(lon) && lon >= 80.0 && lon <= 100.0) {
      runPrediction(lat, lon, dVal);
    }
  });

  const predictBtn = document.getElementById("btn-run-prediction");
  predictBtn.addEventListener("click", () => {
    const rawLat = (document.getElementById("input-lat")?.value || "").trim();
    const rawLon = (document.getElementById("input-lon")?.value || "").trim();
    const dateVal = (dateInput?.value || "").trim();

    if (!rawLat || isNaN(parseFloat(rawLat))) {
      showToast("Please enter a valid numeric latitude (5.00°N to 25.00°N).", "warning");
      document.getElementById("input-lat")?.focus();
      return;
    }
    const lat = parseFloat(rawLat);
    if (lat < 5.0 || lat > 25.0) {
      showToast(`Latitude (${lat.toFixed(2)}°N) must be between 5.00°N and 25.00°N within the Bay of Bengal domain.`, "warning");
      document.getElementById("input-lat")?.focus();
      return;
    }

    if (!rawLon || isNaN(parseFloat(rawLon))) {
      showToast("Please enter a valid numeric longitude (80.00°E to 100.00°E).", "warning");
      document.getElementById("input-lon")?.focus();
      return;
    }
    const lon = parseFloat(rawLon);
    if (lon < 80.0 || lon > 100.0) {
      showToast(`Longitude (${lon.toFixed(2)}°E) must be between 80.00°E and 100.00°E within the Bay of Bengal domain.`, "warning");
      document.getElementById("input-lon")?.focus();
      return;
    }

    if (!dateVal) {
      showToast("Please select a valid calendar date in 2024.", "warning");
      dateInput?.focus();
      return;
    }

    updateMarker(lat, lon);
    runPrediction(lat, lon, dateVal);
  });

  // Quick Locations buttons
  const quickBtns = document.querySelectorAll(".btn-quick-loc");
  quickBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      quickBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const locId = btn.getAttribute("data-id");
      handleQuickLocation(locId);
    });
  });
}

function handleQuickLocation(locId) {
  const presets = {
    central_bay: { lat: 14.00, lon: 87.00, date: "2024-04-06" },
    northern_bay: { lat: 17.50, lon: 88.50, date: "2024-05-24" },
    southern_bay: { lat: 7.50, lon: 85.50, date: "2024-07-15" },
    andaman_sea: { lat: 10.50, lon: 91.25, date: "2024-08-27" },
    sri_lanka_east: { lat: 8.25, lon: 83.50, date: "2024-09-15" },
  };

  const p = presets[locId] || presets.central_bay;

  document.getElementById("input-lat").value = p.lat.toFixed(2);
  document.getElementById("input-lon").value = p.lon.toFixed(2);
  document.getElementById("explore-date-input").value = p.date;
  document.getElementById("display-date").value = p.date;

  updateMarker(p.lat, p.lon);
  map.panTo([p.lat, p.lon]);

  runPrediction(p.lat, p.lon, p.date);
}

/* ==========================================================================
   4. 5x5 SPATIAL MATRIX DIAGRAM
   ========================================================================== */
function initGrid5x5() {
  const container = document.getElementById("matrix-5x5");
  if (!container) return;
  container.innerHTML = "";

  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      const cell = document.createElement("div");
      cell.className = "grid-cell-square";
      if (r === 2 && c === 2) {
        cell.classList.add("center-cell");
        cell.title = "Center: Target Prediction Cell";
      } else {
        const dy = 2 - r;
        const dx = c - 2;
        cell.title = `Neighbor offset: (${dx * 0.25 >= 0 ? "+" : ""}${(dx * 0.25).toFixed(2)}°, ${dy * 0.25 >= 0 ? "+" : ""}${(dy * 0.25).toFixed(2)}°)`;
      }
      container.appendChild(cell);
    }
  }
}

/* ==========================================================================
   5. LIVE INFERENCE EXECUTION
   ========================================================================== */
let isPredictionRunning = false;

async function runPrediction(lat, lon, date) {
  if (isPredictionRunning) {
    return;
  }

  // Defensive validation guard
  if (typeof lat !== "number" || isNaN(lat) || typeof lon !== "number" || isNaN(lon)) {
    showToast("Please enter valid numeric coordinates.", "warning");
    return;
  }
  if (lat < 5.0 || lat > 25.0 || lon < 80.0 || lon > 100.0) {
    showToast(`Coordinates (${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E) are outside the Bay of Bengal domain (5°N–25°N, 80°E–100°E).`, "warning");
    return;
  }
  const cleanDate = String(date || "").trim();
  if (!cleanDate) {
    showToast("Please select a valid calendar date in 2024.", "warning");
    return;
  }

  isPredictionRunning = true;
  const btn = document.getElementById("btn-run-prediction");
  const originalBtnText = btn ? btn.innerHTML : "";
  if (btn) {
    btn.innerHTML = `<span class="spinner"></span> <span>Running Inference...</span>`;
    btn.disabled = true;
  }

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, date: cleanDate }),
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      let errorMsg = data.error;
      if (!errorMsg && data.detail && Array.isArray(data.detail)) {
        errorMsg = data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
      }
      showToast(errorMsg || "That location does not have a valid 5x5 ocean context for this prototype. Try a verified demo location.", "warning");
      clearPredictionDisplay("Unavailable");
      return;
    }

    currentPrediction = data;

    // 1. Update Surface Conditions Cards
    updateSurfaceCards(data.surface_conditions);

    // 2. Render Profile Chart, Table & In-Situ Argo Banner
    renderProfileChart(data.profile, data.argo_observation);
    renderProfileTable(data.profile, data.argo_observation);
    updateArgoBanner(data.argo_observation);

    // 3. Update Studio info if active
    updateStudioView();

    // 4. Update coordinates if snapped
    if (data.location.was_snapped) {
      document.getElementById("input-lat").value = data.location.lat.toFixed(2);
      document.getElementById("input-lon").value = data.location.lon.toFixed(2);
      updateMarker(data.location.lat, data.location.lon);
      showToast(`Adjusted to closest valid ocean cell (${data.location.distance_from_query_km} km)`, "info");
    }

  } catch (err) {
    console.error("Inference request failed:", err);
    showToast("Unable to reach inference engine. Try a verified demo location.", "warning");
    clearPredictionDisplay("Unavailable");
  } finally {
    isPredictionRunning = false;
    if (btn) {
      btn.innerHTML = originalBtnText;
      btn.disabled = false;
    }
  }
}

function clearPredictionDisplay(statusText = "—") {
  const ids = ["val-sst", "val-sss", "val-sla", "val-cu", "val-cv", "val-wu", "val-wv"];
  ids.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.innerText = statusText;
  });
  const studioIds = ["studio-sst", "studio-mld", "studio-tc", "studio-grad", "studio-deep"];
  studioIds.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.innerText = statusText;
  });
  ["mini-sst", "mini-sss", "mini-sla", "mini-cu", "mini-cv", "mini-wu", "mini-wv"].forEach((id) => {
    const canvas = document.getElementById(id);
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  });
  const banner = document.getElementById("argo-observation-banner");
  if (banner) {
    banner.className = "argo-obs-banner unmatched";
    banner.innerHTML = `
      <i data-lucide="info" style="width: 14px; height: 14px; flex-shrink: 0; color: #64748B;"></i>
      <span>No nearby Argo observation available for this location/date</span>
    `;
    if (window.lucide && typeof lucide.createIcons === "function") lucide.createIcons();
  }
}

/* ==========================================================================
   6. SURFACE CONDITIONS DISPLAY & MINI HEATMAPS
   ========================================================================== */
function updateSurfaceCards(sc) {
  if (!sc) return;

  const mapping = {
    sst: { idVal: "val-sst", idCanvas: "mini-sst", color: ["#1E88E5", "#FDD835", "#E53935"] },
    sss: { idVal: "val-sss", idCanvas: "mini-sss", color: ["#00ACC1", "#43A047", "#1E88E5"] },
    sla: { idVal: "val-sla", idCanvas: "mini-sla", color: ["#1E88E5", "#FFFFFF", "#E53935"] },
    current_u: { idVal: "val-cu", idCanvas: "mini-cu", color: ["#42A5F5", "#BBDEFB", "#1565C0"] },
    current_v: { idVal: "val-cv", idCanvas: "mini-cv", color: ["#64B5F6", "#E3F2FD", "#0D47A1"] },
    wind_u: { idVal: "val-wu", idCanvas: "mini-wu", color: ["#81D4FA", "#E1F5FE", "#0288D1"] },
    wind_v: { idVal: "val-wv", idCanvas: "mini-wv", color: ["#EF9A9A", "#FFEBEE", "#C62828"] },
  };

  for (const [key, cfg] of Object.entries(mapping)) {
    if (sc[key]) {
      const elVal = document.getElementById(cfg.idVal);
      if (elVal) elVal.innerText = sc[key].display;

      // Draw mini 5x5 heatmap
      drawMiniHeatmap(cfg.idCanvas, sc[key].grid_5x5, cfg.color);
    }
  }
}

function drawMiniHeatmap(canvasId, grid5x5, colors) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !grid5x5) return;
  const ctx = canvas.getContext("2d");
  const w = (canvas.width = canvas.clientWidth);
  const h = (canvas.height = canvas.clientHeight);

  const flat = grid5x5.flat();
  const minVal = Math.min(...flat);
  const maxVal = Math.max(...flat);
  const range = maxVal - minVal || 1.0;

  const cellW = w / 5.0;
  const cellH = h / 5.0;

  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      const v = (grid5x5[r][c] - minVal) / range;
      // Simple gradient interpolation
      ctx.fillStyle = interpolateColor(colors[0], colors[colors.length - 1], v);
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }
  }

  // Highlight center cell border
  ctx.strokeStyle = "#FFFFFF";
  ctx.lineWidth = 2;
  ctx.strokeRect(2 * cellW, 2 * cellH, cellW, cellH);
}

function interpolateColor(color1, color2, factor) {
  // Simple hex color interpolation
  const c1 = parseInt(color1.slice(1), 16);
  const c2 = parseInt(color2.slice(1), 16);

  const r1 = (c1 >> 16) & 255, g1 = (c1 >> 8) & 255, b1 = c1 & 255;
  const r2 = (c2 >> 16) & 255, g2 = (c2 >> 8) & 255, b2 = c2 & 255;

  const r = Math.round(r1 + factor * (r2 - r1));
  const g = Math.round(g1 + factor * (g2 - g1));
  const b = Math.round(b1 + factor * (b2 - b1));

  return `rgb(${r},${g},${b})`;
}

/* ==========================================================================
   7. VERTICAL TEMPERATURE PROFILE CHART & TABLE
   ========================================================================== */
function renderProfileChart(profile, argoObs) {
  const ctx = document.getElementById("profile-chart");
  if (!ctx || !profile) return;

  const depths = profile.map((p) => p.depth_m);
  const hasArgo = argoObs && argoObs.available && profile.some((p) => p.argo_temp_c !== null && p.argo_temp_c !== undefined);

  if (profileChart) {
    profileChart.destroy();
  }

  const datasets = [
    {
      label: "OceanEmbed Prediction",
      data: profile.map((p) => ({ x: p.predicted_temp_c, y: p.depth_m })),
      borderColor: "#0F5B78",
      backgroundColor: "#0F5B78",
      borderWidth: 2.5,
      pointRadius: 4.5,
      pointHoverRadius: 6.5,
      pointBackgroundColor: "#0F5B78",
      pointBorderColor: "#FFFFFF",
      pointBorderWidth: 1.5,
      tension: 0.35,
      fill: false,
    },
  ];

  if (hasArgo) {
    // Only include depths where Argo observations actually exist (strictly no extrapolation)
    const argoPoints = profile
      .filter((p) => p.argo_temp_c !== null && p.argo_temp_c !== undefined)
      .map((p) => ({ x: p.argo_temp_c, y: p.depth_m }));

    datasets.push({
      label: "Argo Observation",
      data: argoPoints,
      borderColor: "#E65100",
      backgroundColor: "#E65100",
      borderWidth: 2.0,
      borderDash: [5, 4],
      pointRadius: 4.5,
      pointHoverRadius: 6.5,
      pointBackgroundColor: "#E65100",
      pointBorderColor: "#FFFFFF",
      pointBorderWidth: 1.5,
      tension: 0.35,
      fill: false,
    });
  }

  profileChart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "nearest",
        intersect: false,
      },
      plugins: {
        legend: {
          display: true,
          position: "top",
          align: "end",
          labels: {
            boxWidth: 12,
            font: { size: 10, family: "'Plus Jakarta Sans', sans-serif", weight: "bold" },
            color: "#475569",
          },
        },
        tooltip: {
          backgroundColor: "#1E293B",
          titleFont: { size: 11, weight: "bold" },
          bodyFont: { size: 11 },
          padding: 8,
          callbacks: {
            title: (items) => `Depth: ${items[0].parsed.y} m`,
            label: (item) => ` ${item.dataset.label}: ${item.parsed.x.toFixed(2)} °C`,
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          position: "bottom",
          title: {
            display: true,
            text: "Temperature (°C)",
            font: { size: 10, weight: "bold" },
            color: "#64748B",
          },
          min: 4,
          max: 32,
          grid: { color: "#F1F5F9" },
          ticks: { font: { size: 9 }, color: "#64748B" },
        },
        y: {
          type: "linear",
          reverse: true, // Inverted: 0m at top, 1000m at bottom!
          title: {
            display: true,
            text: "Depth (m)",
            font: { size: 10, weight: "bold" },
            color: "#64748B",
          },
          min: 0,
          max: 1000,
          grid: { color: "#F1F5F9" },
          ticks: {
            stepSize: 200,
            font: { size: 9 },
            color: "#64748B",
          },
        },
      },
    },
  });
}

function renderProfileTable(profile, argoObs) {
  const tbody = document.getElementById("profile-table-body");
  if (!tbody || !profile) return;
  tbody.innerHTML = "";

  profile.forEach((row) => {
    const tr = document.createElement("tr");
    const hasArgoVal = row.argo_temp_c !== null && row.argo_temp_c !== undefined;
    tr.innerHTML = `
      <td>${row.depth_m} m</td>
      <td style="font-weight: 700; color: #0F5B78;">${row.predicted_temp_c.toFixed(1)}</td>
      <td style="color: ${hasArgoVal ? "#E65100" : "#94A3B8"}; font-weight: ${hasArgoVal ? "600" : "normal"};">${hasArgoVal ? row.argo_temp_c.toFixed(1) : "&mdash;"}</td>
    `;
    tbody.appendChild(tr);
  });
}

function updateArgoBanner(argoObs) {
  const banner = document.getElementById("argo-observation-banner");
  const badge = document.getElementById("argo-match-badge");
  if (!banner) return;

  if (argoObs && argoObs.available) {
    banner.className = "argo-obs-banner matched";
    banner.innerHTML = `
      <i data-lucide="check-circle" style="width: 14px; height: 14px; flex-shrink: 0; color: #16A34A;"></i>
      <span><strong>In-Situ Argo Collocation:</strong> Float #${argoObs.platform} (Cycle ${argoObs.cycle} &bull; ${argoObs.distance_km} km away &bull; Date: ${argoObs.profile_date})</span>
    `;
    if (badge) {
      badge.innerText = `Argo Collocated (${argoObs.distance_km} km)`;
      badge.style.color = "#166534";
      badge.style.backgroundColor = "#DCFCE7";
    }
  } else {
    banner.className = "argo-obs-banner unmatched";
    banner.innerHTML = `
      <i data-lucide="info" style="width: 14px; height: 14px; flex-shrink: 0; color: #64748B;"></i>
      <span>No nearby Argo observation available for this location/date</span>
    `;
    if (badge) {
      badge.innerText = "OceanEmbed CNN (15 Depths)";
      badge.style.color = "var(--primary-ocean)";
      badge.style.backgroundColor = "var(--accent-light-blue)";
    }
  }

  if (window.lucide && typeof lucide.createIcons === "function") {
    lucide.createIcons();
  }
}

/* ==========================================================================
   8. STUDIO DIAGNOSTICS & EMBEDDING CHART
   ========================================================================== */
function updateStudioView() {
  if (!currentPrediction) return;

  const diag = currentPrediction.diagnostics;
  const emb = currentPrediction.embedding;

  document.getElementById("studio-sst").innerText = `${diag.surface_temperature_c.toFixed(1)} °C`;
  document.getElementById("studio-mld").innerText = `${diag.mixed_layer_depth_m} m`;
  document.getElementById("studio-tc").innerText = `${diag.thermocline_core_depth_m} m`;
  document.getElementById("studio-grad").innerText = `${diag.max_vertical_gradient_c_per_m.toFixed(3)} °C/m`;
  document.getElementById("studio-deep").innerText = `${diag.deep_temperature_1000m_c.toFixed(1)} °C`;

  // Draw 64-D Embedding Bar Chart
  const ctx = document.getElementById("embedding-chart");
  if (ctx && emb) {
    if (embeddingChart) embeddingChart.destroy();

    embeddingChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: emb.values.map((_, i) => `#${i + 1}`),
        datasets: [
          {
            label: "Latent Dimension Value",
            data: emb.values,
            backgroundColor: emb.values.map((v) => (v >= 0 ? "#1B4D47" : "#0F5B78")),
            borderRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: { grid: { color: "#F1F5F9" }, ticks: { font: { size: 9 } } },
        },
      },
    });
  }
}

/* ==========================================================================
   9. DATA SOURCES TAB (SCIENTIFIC CATALOG)
   ========================================================================== */
async function loadDataCatalog() {
  const container = document.getElementById("catalog-container");
  if (!container || container.children.length > 0) return;

  try {
    const res = await fetch("/api/data-catalog");
    const data = await res.json();

    container.innerHTML = "";
    data.catalog.forEach((item) => {
      const card = document.createElement("div");
      card.className = "catalog-card";
      card.innerHTML = `
        <div>
          <div class="catalog-card-head">
            <h4>${item.variable}</h4>
            <span class="catalog-role-pill">${item.dataset_id}</span>
          </div>
          <p style="font-size: 0.8rem; color: #475569; margin-bottom: 8px;">${item.importance}</p>
          <table class="catalog-spec-table">
            <tr><td>Role:</td><td>${item.role}</td></tr>
            <tr><td>Sensor Platform:</td><td>${item.sensor_platform}</td></tr>
            <tr><td>Spatial Resolution:</td><td>${item.spatial_res}</td></tr>
            <tr><td>Temporal Resolution:</td><td>${item.temporal_res}</td></tr>
            <tr><td>Provider:</td><td>${item.provider}</td></tr>
            <tr><td>Units:</td><td>${item.units}</td></tr>
          </table>
        </div>
        <div style="font-size: 0.72rem; font-weight: 700; color: #166534; background: #DCFCE7; padding: 4px 10px; border-radius: 6px; text-align: center; margin-top: 8px;">
          ✓ ${item.status}
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to load data catalog:", err);
  }
}

/* ==========================================================================
   10. MODEL & METHOD TAB
   ========================================================================== */
async function loadModelSpecs() {
  const tbody = document.getElementById("model-specs-table");
  if (!tbody || tbody.children.length > 0) return;

  try {
    const res = await fetch("/api/model-specs");
    const data = await res.json();

    tbody.innerHTML = "";
    data.architecture_components.forEach((c) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 700; color: #0F5B78;">${c.name}</td>
        <td style="font-size: 0.75rem; color: #475569;">${c.description} ${c.operation ? `<br><code>${c.operation}</code>` : ""}</td>
        <td style="text-align: center; font-family: 'JetBrains Mono', monospace; font-weight: 700;">${c.shape}</td>
        <td style="text-align: right; font-family: 'JetBrains Mono', monospace;">${c.parameters ? c.parameters.toLocaleString() : "—"}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load model specs:", err);
  }
}

/* ==========================================================================
   11. VALIDATION (ARGO) TAB
   ========================================================================== */
async function loadArgoValidation() {
  const tbody = document.getElementById("argo-table-body");
  if (!tbody || tbody.children.length > 0) return;

  try {
    const res = await fetch("/api/validation-argo");
    const data = await res.json();

    document.getElementById("argo-num-floats").innerText = data.collocation.matched_profiles.toLocaleString();
    document.getElementById("argo-num-points").innerText = data.overall_metrics.total_temperature_comparisons.toLocaleString();
    document.getElementById("argo-rmse").innerText = `${data.overall_metrics.oceanembed_rmse.toFixed(4)} °C`;
    document.getElementById("argo-imp").innerText = `+${data.overall_metrics.improvement_pct.toFixed(2)}%`;
    document.getElementById("argo-bias").innerText = `${data.overall_metrics.bias > 0 ? "+" : ""}${data.overall_metrics.bias.toFixed(4)} °C`;

    tbody.innerHTML = "";
    for (const [depthKey, m] of Object.entries(data.per_depth_metrics)) {
      const tr = document.createElement("tr");
      const isPositive = m.improvement_pct > 0;
      tr.innerHTML = `
        <td style="font-weight: 700;">${m.depth_m} m</td>
        <td style="text-align: center;">${m.n_matched}</td>
        <td style="text-align: right; font-weight: 700; color: #0F5B78;">${m.oceanembed_rmse.toFixed(4)} °C</td>
        <td style="text-align: right; color: #64748B;">${m.baseline_rmse.toFixed(4)} °C</td>
        <td style="text-align: right; font-weight: 700; color: ${isPositive ? "#166534" : "#991B1B"};">
          ${isPositive ? "+" : ""}${m.improvement_pct.toFixed(2)}%
        </td>
        <td style="text-align: right;">${m.bias > 0 ? "+" : ""}${m.bias.toFixed(4)} °C</td>
        <td style="text-align: right; font-weight: 600;">${m.pearson_r.toFixed(4)}</td>
        <td style="text-align: right;">${m.r2.toFixed(4)}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (err) {
    console.error("Failed to load ARGO validation metrics:", err);
  }
}

/* ==========================================================================
   12. TOAST NOTIFICATION UTILITY
   ========================================================================== */
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <i data-lucide="${type === "warning" ? "alert-triangle" : "info"}" style="width: 16px; height: 16px;"></i>
    <span>${message}</span>
  `;

  container.appendChild(toast);
  if (window.lucide) lucide.createIcons();

  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 250);
  }, 4000);
}
