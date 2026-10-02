/**
 * Campus IQ - Frontend Application Controller
 * Connects UI interactions to backend statistical ML REST API
 */

let systemSummary = {};
let allDatasetRecords = [];

document.addEventListener('DOMContentLoaded', async () => {
  setupNavigation();
  setupRangeDisplays();
  await loadSummaryData();
  setupEventListeners();
  await triggerInitialSimulations();
});

function setupNavigation() {
  const navBtns = document.querySelectorAll('.nav-btn');
  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      navBtns.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      
      btn.classList.add('active');
      const targetTab = btn.getAttribute('data-tab');
      const targetPane = document.getElementById(targetTab);
      if (targetPane) targetPane.classList.add('active');

      // Trigger chart resize & re-population for newly visible tab
      onTabActivated(targetTab);
    });
  });
}

async function onTabActivated(targetTab) {
  if (!systemSummary || Object.keys(systemSummary).length === 0) {
    await loadSummaryData();
  }
  setTimeout(() => {
    if (!systemSummary) return;

    if (targetTab === 'tab-clustering') {
      if (systemSummary.clustering_and_pca) {
        populateClustering(systemSummary.clustering_and_pca);
      }
      if (chartRegistry.pcaScatter) {
        chartRegistry.pcaScatter.resize();
        chartRegistry.pcaScatter.update('none');
      }
      if (chartRegistry.silhouette) {
        chartRegistry.silhouette.resize();
        chartRegistry.silhouette.update('none');
      }
    } else if (targetTab === 'tab-models') {
      populateModelBenchmarks(systemSummary);
      if (chartRegistry.globalImportance) {
        chartRegistry.globalImportance.resize();
        chartRegistry.globalImportance.update('none');
      }
    } else if (targetTab === 'tab-statistics' || targetTab === 'tab-stats') {
      populateStatisticalEDA(systemSummary);
    } else if (targetTab === 'tab-flow') {
      if (systemSummary.campus_demand_flow) {
        populateFlowMatrix(systemSummary.campus_demand_flow);
      }
    } else if (targetTab === 'tab-data') {
      if (!allDatasetRecords || allDatasetRecords.length === 0) {
        loadDatasetSample();
      } else {
        renderDatasetTable(allDatasetRecords);
      }
    } else if (targetTab === 'tab-sports') {
      if (chartRegistry.sportsExplain) {
        chartRegistry.sportsExplain.resize();
        chartRegistry.sportsExplain.update('none');
      }
    } else if (targetTab === 'tab-food') {
      if (chartRegistry.foodSurplus) {
        chartRegistry.foodSurplus.resize();
        chartRegistry.foodSurplus.update('none');
      }
    } else if (targetTab === 'tab-library') {
      if (chartRegistry.libExamComp) {
        chartRegistry.libExamComp.resize();
        chartRegistry.libExamComp.update('none');
      }
    } else if (targetTab === 'tab-overview') {
      if (chartRegistry.overviewTrends) {
        chartRegistry.overviewTrends.resize();
        chartRegistry.overviewTrends.update('none');
      }
    }

    window.dispatchEvent(new Event('resize'));
  }, 40);
}

function setupRangeDisplays() {
  // Sports hour slider
  const simHour = document.getElementById('sim-hour');
  const valDispHour = document.getElementById('val-disp-hour');
  if (simHour && valDispHour) {
    simHour.addEventListener('input', (e) => {
      const h = parseInt(e.target.value);
      const tag = h < 12 ? 'Morning' : (h < 17 ? 'Afternoon' : (h < 20 ? 'Evening Peak' : 'Night'));
      valDispHour.innerText = `${h.toString().padStart(2, '0')}:00 (${tag})`;
    });
  }

  // Classes ending slider
  const simClasses = document.getElementById('sim-classes');
  const valDispClasses = document.getElementById('val-disp-classes');
  if (simClasses && valDispClasses) {
    simClasses.addEventListener('input', (e) => {
      valDispClasses.innerText = `${e.target.value} Batches`;
    });
  }

  // Prev crowd slider
  const simPrev = document.getElementById('sim-prev-crowd');
  const valDispPrev = document.getElementById('val-disp-prev');
  if (simPrev && valDispPrev) {
    simPrev.addEventListener('input', (e) => {
      valDispPrev.innerText = `${e.target.value}`;
    });
  }

  // Food sports spillover slider
  const foodSpill = document.getElementById('food-sports-spillover');
  const valDispSpill = document.getElementById('val-disp-sports-spill');
  if (foodSpill && valDispSpill) {
    foodSpill.addEventListener('input', (e) => {
      valDispSpill.innerText = `${e.target.value} visitors`;
    });
  }

  // Food planned prep slider
  const foodPrep = document.getElementById('food-planned-prep');
  const valDispPrep = document.getElementById('val-disp-prep');
  if (foodPrep && valDispPrep) {
    foodPrep.addEventListener('input', (e) => {
      valDispPrep.innerText = `${e.target.value} meals`;
    });
  }
}

async function loadSummaryData() {
  try {
    const res = await fetch('/api/summary');
    systemSummary = await res.json();
    console.log('[Campus IQ] Received system summary from API:', systemSummary);
    
    try { populateOverview(systemSummary); } catch (e) { console.error('Error in populateOverview:', e); }
    try { populateFlowMatrix(systemSummary.campus_demand_flow); } catch (e) { console.error('Error in populateFlowMatrix:', e); }
    try { populateClustering(systemSummary.clustering_and_pca); } catch (e) { console.error('Error in populateClustering:', e); }
    try { populateStatisticalEDA(systemSummary); } catch (e) { console.error('Error in populateStatisticalEDA:', e); }
    try { populateModelBenchmarks(systemSummary); } catch (e) { console.error('Error in populateModelBenchmarks:', e); }
    try { await loadDatasetSample(); } catch (e) { console.error('Error in loadDatasetSample:', e); }
  } catch (err) {
    console.error('Failed to load system summary from API:', err);
  }
}

function populateOverview(data) {
  // Update CPI circular gauge
  const cpiVal = data.dataset_metadata?.avg_campus_pulse_index || 52.0;
  const headerVal = document.getElementById('header-cpi-val');
  if (headerVal) headerVal.innerText = cpiVal.toFixed(1);
  const centerVal = document.getElementById('cpi-center-val');
  if (centerVal) centerVal.innerText = cpiVal.toFixed(1);

  const dashVal = Math.round(cpiVal);
  const circlePath = document.getElementById('cpi-gauge-path');
  if (circlePath) {
    circlePath.setAttribute('stroke-dasharray', `${dashVal}, 100`);
    if (dashVal >= 80) circlePath.style.stroke = '#ef4444';
    else if (dashVal >= 60) circlePath.style.stroke = '#f59e0b';
    else circlePath.style.stroke = '#10b981';
  }

  // Initialize overview trend lines safely
  if (data.temporal_trends) {
    try {
      initOverviewTrendsChart(data.temporal_trends);
    } catch (e) {
      console.warn('initOverviewTrendsChart error:', e);
    }
  }
}

function populateFlowMatrix(flowData) {
  if (!flowData || !flowData.cross_correlation_matrix) return;
  const matrix = flowData.cross_correlation_matrix;
  const tbody = document.querySelector('#flowCorrTable tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  const rows = [
    { name: 'Academic Classes Ending', key: 'classes_ending_near' },
    { name: 'Sports Visitor Count', key: 'sports_visitor_count' },
    { name: 'Food Orders Demand', key: 'food_orders_demand' },
    { name: 'Library Occupancy', key: 'library_occupancy' }
  ];

  rows.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${r.name}</strong></td>
      <td>${matrix[r.key]?.['sports_visitor_count'] ?? '1.0'}</td>
      <td>${matrix[r.key]?.['food_orders_demand'] ?? '0.0'}</td>
      <td>${matrix[r.key]?.['library_occupancy'] ?? '0.0'}</td>
      <td><strong>${matrix[r.key]?.['campus_pulse_index'] ?? '0.0'}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function populateClustering(clusterData) {
  if (!clusterData) return;
  const profiles = clusterData.cluster_profiles || {};
  const container = document.getElementById('cluster-cards-container');
  if (container) {
    container.innerHTML = '';
    Object.values(profiles).forEach(p => {
      const card = document.createElement('div');
      card.className = 'persona-card';
      card.innerHTML = `
        <div class="persona-title">Cluster ${p.cluster_id}: ${p.label}</div>
        <div class="persona-pct"><i class="fa-solid fa-users"></i> ${p.percentage_of_data}% of campus states &bull; Peak: ${p.predominant_time_slot}</div>
        <div class="persona-desc">${p.description}</div>
        <div class="text-sm mb-5">
          <strong>Averages:</strong> Sports: ${p.avg_sports_load_pct}% | Food: ${p.avg_food_orders} orders | Lib: ${p.avg_library_load_pct}%
        </div>
        <div class="persona-rec"><i class="fa-solid fa-lightbulb"></i> ${p.recommended_action}</div>
      `;
      container.appendChild(card);
    });
  }

  // Update header badges
  const optPill = document.getElementById('optimal-k-pill');
  if (optPill && clusterData.optimal_k) {
    optPill.innerText = `Optimal K = ${clusterData.optimal_k}`;
  }
  const pcaLabel = document.getElementById('pca-variance-label');
  if (pcaLabel && clusterData.pca_variance_explained) {
    const pc1 = (clusterData.pca_variance_explained[0] * 100).toFixed(1);
    const pc2 = (clusterData.pca_variance_explained[1] * 100).toFixed(1);
    const total = ((clusterData.pca_variance_explained[0] + clusterData.pca_variance_explained[1]) * 100).toFixed(1);
    pcaLabel.innerText = `PCA Variance Explained: PC1 (${pc1}%) + PC2 (${pc2}%) = ${total}%`;
  }

  // Render PCA Scatter & Silhouette Charts
  if (clusterData.pca_points) {
    initPcaScatterChart(clusterData.pca_points, profiles, clusterData.pca_variance_explained);
  }
  if (clusterData.k_evaluation) {
    initSilhouetteChart(clusterData.k_evaluation);
  }
}

function populateStatisticalEDA(data) {
  // Descriptive Stats Table
  const descStats = data.descriptive_stats || [];
  const descTbody = document.querySelector('#descStatsTable tbody');
  if (descTbody) {
    descTbody.innerHTML = '';
    descStats.forEach(s => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${s.metric.replace(/_/g, ' ')}</strong></td>
        <td>${s.count}</td>
        <td>${s.mean}</td>
        <td>${s.std}</td>
        <td>${s.median}</td>
        <td>${s.q25}</td>
        <td>${s.q75}</td>
        <td>${s.iqr}</td>
        <td>${s.min}</td>
        <td>${s.max}</td>
        <td>${s.skewness}</td>
      `;
      descTbody.appendChild(tr);
    });
  }

  // Hypotheses Cards
  const hypotheses = data.hypothesis_tests || [];
  const hypContainer = document.getElementById('hypothesis-cards-container');
  if (hypContainer) {
    hypContainer.innerHTML = '';
    hypotheses.forEach(h => {
      const card = document.createElement('div');
      card.className = 'hypothesis-card';
      card.innerHTML = `
        <div class="hyp-title">${h.hypothesis}</div>
        <div class="hyp-metric">t-stat: <strong>${h.t_statistic}</strong> &bull; p-value: <strong>${h.p_value_ttest}</strong></div>
        <div class="hyp-metric">Mann-Whitney U: <strong>${h.mann_whitney_u}</strong> &bull; Cohen's d: <strong>${h.cohens_d}</strong></div>
        <div class="hyp-conclusion"><i class="fa-solid fa-check"></i> ${h.conclusion}</div>
      `;
      hypContainer.appendChild(card);
    });
  }

  // Outliers Table
  const sportsOutliers = data.outlier_analysis?.sports?.outliers || [];
  const outTbody = document.querySelector('#outlierTable tbody');
  if (outTbody) {
    outTbody.innerHTML = '';
    if (sportsOutliers.length === 0) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td colspan="7" class="text-center text-muted" style="padding: 24px; text-align: center;"><i class="fa-solid fa-circle-check text-success"></i> <strong>Zero Extreme Outliers Detected:</strong> All observations remain strictly within 1.5 &times; IQR operational thresholds.</td>`;
      outTbody.appendChild(tr);
    } else {
      sportsOutliers.slice(0, 15).forEach(o => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><code>${o.record_id}</code></td>
          <td>${o.date}</td>
          <td>${o.time_slot}</td>
          <td><strong class="text-warning">${o.value}</strong></td>
          <td>${o.upper_threshold}</td>
          <td><span class="badge badge-info">${o.event_context}</span></td>
          <td><span class="text-sm">${o.action_taken}</span></td>
        `;
        outTbody.appendChild(tr);
      });
    }
  }
}

function populateModelBenchmarks(data) {
  // Regressors Table
  const regTbody = document.querySelector('#regModelTable tbody');
  if (regTbody) {
    regTbody.innerHTML = '';
    const sportsRegs = data.sports_models?.regressors || {};
    const foodRegs = data.food_models?.regressors || {};
    const libRegs = data.library_models?.regressors || {};

    Object.entries(sportsRegs).forEach(([name, m]) => {
      const tr = document.createElement('tr');
      const isBest = name === data.sports_models.best_regressor;
      tr.innerHTML = `
        <td><span class="badge badge-warning">Sports Visitors</span></td>
        <td><strong>${name}</strong></td>
        <td>${m.MAE}</td>
        <td>${m.RMSE}</td>
        <td><strong>${m.R2}</strong></td>
        <td>${isBest ? '<span class="badge badge-success">Selected Best</span>' : '<span class="badge badge-info">Candidate</span>'}</td>
      `;
      regTbody.appendChild(tr);
    });

    Object.entries(foodRegs).forEach(([name, m]) => {
      const tr = document.createElement('tr');
      const isBest = name === data.food_models.best_regressor;
      tr.innerHTML = `
        <td><span class="badge badge-success">Food Demand</span></td>
        <td><strong>${name}</strong></td>
        <td>${m.MAE}</td>
        <td>${m.RMSE}</td>
        <td><strong>${m.R2}</strong></td>
        <td>${isBest ? '<span class="badge badge-success">Selected Best</span>' : '<span class="badge badge-info">Candidate</span>'}</td>
      `;
      regTbody.appendChild(tr);
    });

    Object.entries(libRegs).forEach(([name, m]) => {
      const tr = document.createElement('tr');
      const isBest = name === data.library_models?.best_regressor;
      tr.innerHTML = `
        <td><span class="badge badge-info">Library Study</span></td>
        <td><strong>${name}</strong></td>
        <td>${m.MAE}</td>
        <td>${m.RMSE}</td>
        <td><strong>${m.R2}</strong></td>
        <td>${isBest ? '<span class="badge badge-success">Selected Best</span>' : '<span class="badge badge-info">Candidate</span>'}</td>
      `;
      regTbody.appendChild(tr);
    });
  }

  // Classifiers Table
  const clfTbody = document.querySelector('#clfModelTable tbody');
  if (clfTbody) {
    clfTbody.innerHTML = '';
    const sportsClfs = data.sports_models?.classifiers || {};
    const foodClfs = data.food_models?.classifiers || {};

    Object.entries(sportsClfs).forEach(([name, m]) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="badge badge-warning">Sports Overcrowding</span></td>
        <td><strong>${name}</strong></td>
        <td>${m.Accuracy}</td>
        <td>${m.Precision}</td>
        <td>${m.Recall}</td>
        <td><strong>${m['F1-Score']}</strong></td>
      `;
      clfTbody.appendChild(tr);
    });

    Object.entries(foodClfs).forEach(([name, m]) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="badge badge-success">Food Surplus Risk</span></td>
        <td><strong>${name}</strong></td>
        <td>${m.Accuracy}</td>
        <td>${m.Precision}</td>
        <td>${m.Recall}</td>
        <td><strong>${m['F1-Score']}</strong></td>
      `;
      clfTbody.appendChild(tr);
    });
  }

  // Global Feature Importance Chart
  const featImportances = data.sports_models?.feature_importances?.random_forest_importance;
  if (featImportances) {
    initGlobalImportanceChart(featImportances);
  }
}

async function loadDatasetSample() {
  try {
    const res = await fetch('/api/dataset');
    const data = await res.json();
    allDatasetRecords = data.records || [];
    document.getElementById('dataset-total-count').innerText = data.total_records || 500;
    renderDatasetTable(allDatasetRecords);
  } catch (err) {
    console.error('Failed to load dataset:', err);
  }
}

function renderDatasetTable(records) {
  const tbody = document.getElementById('datasetTableBody');
  if (!tbody) return;
  tbody.innerHTML = '';

  records.slice(0, 40).forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><code>${r.record_id}</code></td>
      <td><span class="badge badge-info">${r.data_source_label}</span></td>
      <td>${r.date}</td>
      <td>${r.day_of_week}</td>
      <td>${r.time_slot}</td>
      <td>${r.starting_hour}:00</td>
      <td>${r.sports_visitor_count}</td>
      <td><strong>${r.sports_load_pct}%</strong></td>
      <td>${r.food_orders_demand}</td>
      <td>${r.library_occupancy}</td>
      <td><strong>${r.campus_pulse_index}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function setupEventListeners() {
  // Sports simulator button
  const btnSports = document.getElementById('btn-run-sports-sim');
  if (btnSports) {
    btnSports.addEventListener('click', runSportsSimulation);
  }

  // Food simulator button
  const btnFood = document.getElementById('btn-run-food-sim');
  if (btnFood) {
    btnFood.addEventListener('click', runFoodSimulation);
  }

  // Library simulator button
  const btnLib = document.getElementById('btn-run-lib-sim');
  if (btnLib) {
    btnLib.addEventListener('click', runLibrarySimulation);
  }

  // Quick recommender button
  const btnQuickRec = document.getElementById('btn-quick-recommend');
  if (btnQuickRec) {
    btnQuickRec.addEventListener('click', runQuickRecommender);
  }

  // Dataset search box
  const searchInput = document.getElementById('dataset-search');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase();
      const filtered = allDatasetRecords.filter(r => 
        r.record_id.toLowerCase().includes(q) ||
        r.date.toLowerCase().includes(q) ||
        r.time_slot.toLowerCase().includes(q) ||
        r.day_of_week.toLowerCase().includes(q)
      );
      renderDatasetTable(filtered);
    });
  }
}

async function triggerInitialSimulations() {
  try { await runSportsSimulation(); } catch (e) { console.error('Error in runSportsSimulation:', e); }
  try { await runFoodSimulation(); } catch (e) { console.error('Error in runFoodSimulation:', e); }
  try { await runLibrarySimulation(); } catch (e) { console.error('Error in runLibrarySimulation:', e); }
  try { initFoodSurplusChart(); } catch (e) { console.error('Error in initFoodSurplusChart:', e); }
  try { initLibraryExamCompChart(); } catch (e) { console.error('Error in initLibraryExamCompChart:', e); }
}

async function runSportsSimulation() {
  const hour = parseInt(document.getElementById('sim-hour').value);
  const day = parseInt(document.getElementById('sim-day').value);
  const classes = parseInt(document.getElementById('sim-classes').value);
  const prevCrowd = parseInt(document.getElementById('sim-prev-crowd').value);
  const eventVal = document.getElementById('sim-event').value;
  const academicVal = document.getElementById('sim-academic').value;
  const weatherVal = document.getElementById('sim-weather').value;

  const payload = {
    starting_hour: hour,
    day_of_week_num: day,
    is_weekend: day >= 5 ? 1 : 0,
    classes_ending_near: classes,
    sports_prev_crowd: prevCrowd,
    sports_hist_avg_crowd: 85,
    sports_hist_peak_crowd: 180,
    sports_active_courts: 6,
    campus_event_flag: eventVal !== 'none' ? 1 : 0,
    sports_event_flag: eventVal === 'sports_match' || eventVal === 'tournament' ? 1 : 0,
    tournament_flag: eventVal === 'tournament' ? 1 : 0,
    cultural_major_event_flag: eventVal === 'cultural' ? 1 : 0,
    expected_event_attendance: eventVal === 'tournament' ? 220 : (eventVal === 'sports_match' ? 120 : 0),
    is_exam_week: academicVal === 'exam' ? 1 : 0,
    temperature: weatherVal === 'hot' ? 34.0 : 24.0,
    rain_flag: weatherVal === 'rain' ? 1 : 0,
    outdoor_suitability: weatherVal === 'rain' ? 0.2 : (weatherVal === 'hot' ? 0.6 : 1.0)
  };

  try {
    const res = await fetch('/api/predict/sports', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderSportsPredictionResults(data);
  } catch (err) {
    console.error('Error running sports prediction:', err);
  }
}

function renderSportsPredictionResults(data) {
  const pred = data.prediction;
  const rec = data.recommendation;
  const exp = data.explanation;

  document.getElementById('pred-sports-count').innerHTML = `${pred.predicted_visitor_count} <span class="unit">/ ${pred.facility_capacity} visitors</span>`;
  document.getElementById('pred-sports-load').innerText = `${pred.predicted_load_pct}%`;
  document.getElementById('pred-meter-bar').style.width = `${Math.min(100, pred.predicted_load_pct)}%`;
  document.getElementById('pred-overcrowd-prob').innerText = `${(pred.overcrowding_probability * 100).toFixed(1)}%`;
  document.getElementById('pred-model-used').innerText = `${pred.regressor_used} & ${pred.classifier_used}`;

  const badge = document.getElementById('sports-risk-badge');
  badge.innerText = `${pred.risk_label} Overcrowding Risk`;
  badge.className = `badge badge-${pred.risk_label === 'Low' ? 'success' : (pred.risk_label === 'Moderate' ? 'info' : (pred.risk_label === 'High' ? 'warning' : 'danger'))}`;

  // Render Actionable Recommendations
  const recContainer = document.getElementById('sports-recommendation-details');
  recContainer.innerHTML = `
    <div class="action-alert-box alert-${rec.should_shift ? 'warning' : 'success'} mb-15">
      <i class="fa-solid fa-${rec.should_shift ? 'triangle-exclamation' : 'circle-check'}"></i>
      <div>
        <strong>${rec.status_summary}:</strong> ${rec.action_verdict}
      </div>
    </div>
    <div class="text-sm font-semibold mb-10 text-muted">RECOMMENDED LOWER-DEMAND TIME ALTERNATIVES:</div>
  `;

  rec.recommended_lower_demand_slots.forEach(slot => {
    const slotCard = document.createElement('div');
    slotCard.className = 'rec-slot-card';
    slotCard.innerHTML = `
      <div>
        <div class="rec-slot-time"><i class="fa-regular fa-clock"></i> ${slot.time_window}</div>
        <div class="rec-slot-savings">${slot.capacity_reduction}</div>
        <div class="rec-slot-tip">${slot.tip}</div>
      </div>
      <div>
        <span class="rec-slot-badge">${slot.predicted_load_pct}% Load</span>
      </div>
    `;
    recContainer.appendChild(slotCard);
  });

  // Render Explainability chart and factor chips
  const expContainer = document.getElementById('sports-explainability-breakdown');
  expContainer.innerHTML = '';
  exp.contributing_factors.forEach(f => {
    const chip = document.createElement('div');
    chip.className = `action-alert-box alert-${f.type === 'positive' ? 'warning' : 'success'} mt-5`;
    chip.innerHTML = `
      <i class="fa-solid fa-${f.type === 'positive' ? 'arrow-trend-up' : 'arrow-trend-down'}"></i>
      <div><strong>${f.factor}:</strong> Contributes <strong>${f.impact}</strong> to facility load.</div>
    `;
    expContainer.appendChild(chip);
  });

  initSportsExplainChart(exp.contributing_factors);
  if (rec.full_day_schedule) {
    initSportsDayTrajectoryChart(rec.full_day_schedule);
  }
}

async function runFoodSimulation() {
  const outlet = document.getElementById('food-outlet-sel').value;
  const hour = parseInt(document.getElementById('food-slot-sel').value);
  const sportsSpill = parseInt(document.getElementById('food-sports-spillover').value);
  const plannedPrep = parseInt(document.getElementById('food-planned-prep').value);

  const payload = {
    food_outlet_name: outlet,
    starting_hour: hour,
    is_peak_hour: (hour === 12 || hour === 18) ? 1 : 0,
    sports_visitor_count: sportsSpill,
    planned_prep_qty: plannedPrep,
    classes_ending_near: 2,
    food_prev_orders: 85,
    food_hist_demand: 120,
    campus_event_flag: 0,
    expected_event_attendance: 0,
    temperature: 24.0,
    rain_flag: 0
  };

  try {
    const res = await fetch('/api/predict/food', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const pred = data.prediction;
    const rec = data.recommendation;

    document.getElementById('pred-food-demand').innerHTML = `${pred.predicted_orders_demand} <span class="unit">orders</span>`;
    document.getElementById('disp-food-prep').innerText = `${plannedPrep} portions`;
    document.getElementById('disp-food-optimal-prep').innerText = `${pred.recommended_preparation_qty} portions`;

    const pill = document.getElementById('food-risk-pill');
    pill.innerText = pred.predicted_risk_category;
    pill.className = `badge badge-${pred.predicted_risk_category === 'Balanced' ? 'success' : (pred.predicted_risk_category.includes('Surplus') ? 'warning' : 'danger')}`;

    const evalText = document.getElementById('food-eval-text');
    if (evalText) evalText.innerText = rec.evaluation;
  } catch (err) {
    console.error('Error running food simulation:', err);
  }
}

async function runLibrarySimulation() {
  const hour = parseInt(document.getElementById('lib-slot-sel').value);
  const stage = document.getElementById('lib-stage-sel').value;
  const classes = parseInt(document.getElementById('lib-classes-sel').value);

  const payload = {
    starting_hour: hour,
    is_peak_hour: (hour >= 14 && hour <= 20) ? 1 : 0,
    classes_ending_near: classes,
    is_exam_week: stage === 'exam' ? 1 : 0,
    is_assignment_deadline: stage === 'assignment' ? 1 : 0,
    library_prev_occupancy: 140,
    library_hist_avg_occupancy: 160,
    library_hist_peak_occupancy: 300,
    library_capacity: 350
  };

  try {
    const res = await fetch('/api/predict/library', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const pred = data.prediction;

    document.getElementById('pred-lib-count').innerHTML = `${pred.predicted_occupancy} <span class="unit">/ ${pred.library_capacity} seats</span>`;
    document.getElementById('pred-lib-load').innerText = `${pred.predicted_load_pct}%`;
    document.getElementById('pred-lib-meter-bar').style.width = `${Math.min(100, pred.predicted_load_pct)}%`;

    const badge = document.getElementById('lib-avail-badge');
    badge.innerText = pred.study_space_availability;
    badge.className = `badge badge-${pred.predicted_load_pct < 60 ? 'success' : (pred.predicted_load_pct < 80 ? 'info' : 'warning')}`;
  } catch (err) {
    console.error('Error running library simulation:', err);
  }
}

async function runQuickRecommender() {
  const facility = document.getElementById('quick-finder-facility').value;
  const hour = parseInt(document.getElementById('quick-finder-hour').value);
  const exam = parseInt(document.getElementById('quick-finder-exam').value);

  const payload = {
    facility: facility,
    starting_hour: hour,
    is_exam_week: exam,
    classes_ending_near: 2,
    sports_prev_crowd: 80,
    sports_hist_avg_crowd: 90,
    sports_hist_peak_crowd: 180,
    library_prev_occupancy: 120,
    food_prev_orders: 80
  };

  const outputDiv = document.getElementById('quick-recommend-output');
  outputDiv.innerHTML = '<div class="text-sm text-muted"><i class="fa-solid fa-spinner fa-spin"></i> Calculating optimal windows...</div>';

  try {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (facility === 'sports') {
      outputDiv.innerHTML = `
        <div class="action-alert-box alert-${data.should_shift ? 'warning' : 'success'}">
          <i class="fa-solid fa-${data.should_shift ? 'triangle-exclamation' : 'circle-check'}"></i>
          <div>
            <strong>${data.status_summary}:</strong> ${data.action_verdict}
            <div class="mt-5"><strong>Top Recommended Alternative:</strong> ${data.recommended_lower_demand_slots[0]?.time_window} (${data.recommended_lower_demand_slots[0]?.predicted_load_pct}% load &bull; ${data.recommended_lower_demand_slots[0]?.capacity_reduction})</div>
          </div>
        </div>
      `;
    } else if (facility === 'library') {
      outputDiv.innerHTML = `
        <div class="action-alert-box alert-info">
          <i class="fa-solid fa-book-open"></i>
          <div>
            <strong>Study Space Guidance:</strong> Best quiet study windows: <strong>${data.best_study_windows[0]?.time_window} (${data.best_study_windows[0]?.predicted_load_pct}% load)</strong>. Alternative quiet zones: <em>Engineering Hub Quiet Lounge (35% typical load)</em>.
          </div>
        </div>
      `;
    } else {
      outputDiv.innerHTML = `
        <div class="action-alert-box alert-success">
          <i class="fa-solid fa-utensils"></i>
          <div>
            <strong>Food Demand Planning (SDG 12):</strong> Predicted demand is <strong>${data.predicted_demand} orders</strong>. Optimal preparation to avoid waste is <strong>${data.optimal_recommended_prep} meals</strong>.
          </div>
        </div>
      `;
    }
  } catch (err) {
    outputDiv.innerHTML = `<div class="text-sm text-danger">Error fetching recommendation.</div>`;
  }
}

// Excel (.xlsx / .csv) Upload Client Handler
let selectedExcelFile = null;

function setupExcelUpload() {
  const fileInput = document.getElementById('excel-file-input');
  const dropzone = document.getElementById('upload-dropzone');
  const fileInfo = document.getElementById('selected-file-info');
  const fileName = document.getElementById('selected-file-name');
  const fileSize = document.getElementById('selected-file-size');
  const submitBtn = document.getElementById('btn-submit-upload');
  const statusMsg = document.getElementById('upload-status-msg');

  if (!fileInput || !dropzone) return;

  function handleFileSelected(file) {
    if (!file) return;
    selectedExcelFile = file;
    fileName.innerText = file.name;
    fileSize.innerText = `${(file.size / 1024).toFixed(1)} KB`;
    fileInfo.style.display = 'inline-flex';
    submitBtn.disabled = false;
    submitBtn.innerHTML = `<i class="fa-solid fa-cloud-arrow-up"></i> Upload "${file.name}" & Retrain`;
  }

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag and drop events
  ['dragenter', 'dragover'].forEach(ev => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(ev => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  submitBtn.addEventListener('click', async () => {
    if (!selectedExcelFile) return;

    submitBtn.disabled = true;
    submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing & Retraining ML Models...`;
    statusMsg.style.display = 'block';
    statusMsg.className = 'action-alert-box alert-info';
    statusMsg.innerHTML = `<i class="fa-solid fa-gear fa-spin"></i> Reading Excel file, mapping features, and executing 12-step classical ML pipeline...`;

    const reader = new FileReader();
    reader.onload = async (e) => {
      const b64 = e.target.result;
      const mode = document.getElementById('upload-mode-sel').value;

      try {
        const res = await fetch('/api/data/upload_xlsx', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: selectedExcelFile.name,
            file_content_base64: b64,
            mode: mode
          })
        });

        const respData = await res.json();
        if (respData.success) {
          statusMsg.className = 'action-alert-box alert-success';
          statusMsg.innerHTML = `<i class="fa-solid fa-circle-check"></i> <strong>Success!</strong> ${respData.message} Total records: <strong>${respData.total_records}</strong>.`;
          submitBtn.innerHTML = `<i class="fa-solid fa-check"></i> Models Successfully Retrained`;
          
          // Refresh entire dashboard with newly trained models & stats
          await loadSummaryData();
          await triggerInitialSimulations();
        } else {
          statusMsg.className = 'action-alert-box alert-warning';
          statusMsg.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Upload Error: ${respData.error}`;
          submitBtn.disabled = false;
          submitBtn.innerHTML = `<i class="fa-solid fa-rotate-right"></i> Try Again`;
        }
      } catch (err) {
        statusMsg.className = 'action-alert-box alert-warning';
        statusMsg.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Server Error: ${err.message}`;
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i class="fa-solid fa-rotate-right"></i> Try Again`;
      }
    };

    reader.readAsDataURL(selectedExcelFile);
  });
}

// Hook setupExcelUpload into setupEventListeners
const origSetupEventListeners = setupEventListeners;
setupEventListeners = function() {
  origSetupEventListeners();
  setupExcelUpload();
};

