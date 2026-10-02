/**
 * Campus IQ - Visualization & Chart Management Module
 * Uses Chart.js with customized modern dark theme styling
 */

const chartRegistry = {};

const CHART_COLORS = {
  primary: '#6366f1',
  primaryAlpha: 'rgba(99, 102, 241, 0.2)',
  secondary: '#06b6d4',
  secondaryAlpha: 'rgba(6, 182, 212, 0.2)',
  success: '#10b981',
  successAlpha: 'rgba(16, 185, 129, 0.2)',
  warning: '#f59e0b',
  warningAlpha: 'rgba(245, 158, 11, 0.2)',
  danger: '#ef4444',
  dangerAlpha: 'rgba(239, 68, 68, 0.2)',
  grid: 'rgba(255, 255, 255, 0.06)',
  text: '#94a3b8'
};

const defaultOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: { color: CHART_COLORS.text, font: { family: 'Plus Jakarta Sans', size: 11.5 } }
    },
    tooltip: {
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      titleColor: '#f1f5f9',
      bodyColor: '#cbd5e1',
      borderColor: 'rgba(255, 255, 255, 0.1)',
      borderWidth: 1,
      padding: 10,
      cornerRadius: 8
    }
  },
  scales: {
    x: {
      grid: { color: CHART_COLORS.grid },
      ticks: { color: CHART_COLORS.text, font: { family: 'Plus Jakarta Sans', size: 11 } }
    },
    y: {
      grid: { color: CHART_COLORS.grid },
      ticks: { color: CHART_COLORS.text, font: { family: 'Plus Jakarta Sans', size: 11 } }
    }
  }
};

function initOverviewTrendsChart(trendsData) {
  if (typeof Chart === 'undefined') { console.warn('Chart.js is not loaded'); return; }
  const ctx = document.getElementById('overviewTrendsChart');
  if (!ctx) return;
  if (chartRegistry.overviewTrends) chartRegistry.overviewTrends.destroy();

  const slots = ["Morning", "Late Morning", "Noon / Lunch", "Afternoon", "Evening Peak", "Night"];
  const sportsData = slots.map(s => trendsData.by_slot?.[s]?.sports_visitor_count ? (trendsData.by_slot[s].sports_visitor_count / 2) : 35);
  const foodData = slots.map(s => trendsData.by_slot?.[s]?.food_orders_demand || 60);
  const libData = slots.map(s => trendsData.by_slot?.[s]?.library_occupancy ? (trendsData.by_slot[s].library_occupancy / 3.5) : 40);

  chartRegistry.overviewTrends = new Chart(ctx, {
    type: 'line',
    data: {
      labels: slots,
      datasets: [
        {
          label: 'Sports Load (%)',
          data: sportsData,
          borderColor: CHART_COLORS.warning,
          backgroundColor: CHART_COLORS.warningAlpha,
          borderWidth: 2.5,
          tension: 0.35,
          fill: true
        },
        {
          label: 'Food Demand (Orders)',
          data: foodData,
          borderColor: CHART_COLORS.success,
          backgroundColor: 'transparent',
          borderWidth: 2.5,
          tension: 0.35
        },
        {
          label: 'Library Load (%)',
          data: libData,
          borderColor: CHART_COLORS.secondary,
          backgroundColor: CHART_COLORS.secondaryAlpha,
          borderWidth: 2.5,
          tension: 0.35,
          fill: true
        }
      ]
    },
    options: defaultOptions
  });
}

function initSportsDayTrajectoryChart(daySchedule) {
  if (typeof Chart === 'undefined') { console.warn('Chart.js is not loaded'); return; }
  const ctx = document.getElementById('sportsDayTrajectoryChart');
  if (!ctx) return;
  if (chartRegistry.sportsDayTrajectory) chartRegistry.sportsDayTrajectory.destroy();

  const labels = daySchedule.map(s => s.time_window);
  const crowdValues = daySchedule.map(s => s.predicted_crowd);
  const redline = new Array(labels.length).fill(160); // 80% capacity of 200

  chartRegistry.sportsDayTrajectory = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Predicted Visitor Count',
          data: crowdValues,
          borderColor: CHART_COLORS.primary,
          backgroundColor: CHART_COLORS.primaryAlpha,
          fill: true,
          tension: 0.3,
          borderWidth: 3,
          pointRadius: 5,
          pointBackgroundColor: CHART_COLORS.primary
        },
        {
          label: 'Overcrowding Threshold (80% / 160 visitors)',
          data: redline,
          borderColor: CHART_COLORS.danger,
          borderDash: [6, 6],
          borderWidth: 2,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      ...defaultOptions,
      scales: {
        ...defaultOptions.scales,
        y: {
          ...defaultOptions.scales.y,
          min: 0,
          max: 220,
          title: { display: true, text: 'Visitor Count', color: CHART_COLORS.text }
        }
      }
    }
  });
}

function initSportsExplainChart(contributingFactors) {
  if (typeof Chart === 'undefined') { console.warn('Chart.js is not loaded'); return; }
  const ctx = document.getElementById('sportsExplainChart');
  if (!ctx) return;
  if (chartRegistry.sportsExplain) chartRegistry.sportsExplain.destroy();

  const factors = contributingFactors || [];
  const labels = factors.map(f => f.factor.length > 28 ? f.factor.substring(0, 26) + '...' : f.factor);
  const values = factors.map(f => parseFloat(f.impact));
  const bgColors = values.map(v => v >= 0 ? 'rgba(239, 68, 68, 0.85)' : 'rgba(16, 185, 129, 0.85)');
  const borderColors = values.map(v => v >= 0 ? '#ef4444' : '#10b981');

  chartRegistry.sportsExplain = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Impact on Capacity Load (%)',
        data: values,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: (ctx) => `Impact: ${ctx.parsed.x > 0 ? '+' : ''}${ctx.parsed.x.toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          grid: { color: CHART_COLORS.grid },
          ticks: { color: CHART_COLORS.text, font: { family: 'Plus Jakarta Sans', size: 11 }, callback: v => `${v}%` },
          title: { display: true, text: 'Percentage Impact on Capacity', color: CHART_COLORS.text }
        },
        y: {
          type: 'category',
          grid: { color: 'transparent' },
          ticks: { color: '#e2e8f0', font: { family: 'Plus Jakarta Sans', size: 11 } }
        }
      }
    }
  });
}

function initFoodSurplusChart() {
  if (typeof Chart === 'undefined') { console.warn('Chart.js is not loaded'); return; }
  const ctx = document.getElementById('foodSurplusChart');
  if (!ctx) return;
  if (chartRegistry.foodSurplus) chartRegistry.foodSurplus.destroy();

  chartRegistry.foodSurplus = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Central Cafeteria (Lunch)', 'Central Cafeteria (Dinner)', 'Sports Kiosk (Evening)', 'Library Lounge (Afternoon)'],
      datasets: [
        {
          label: 'Predicted Demand',
          data: [210, 185, 140, 95],
          backgroundColor: CHART_COLORS.primary,
          borderRadius: 4
        },
        {
          label: 'Planned Preparation',
          data: [215, 190, 135, 100],
          backgroundColor: CHART_COLORS.secondary,
          borderRadius: 4
        },
        {
          label: 'Surplus (Food Saved Margin)',
          data: [5, 5, 0, 5],
          backgroundColor: CHART_COLORS.success,
          borderRadius: 4
        }
      ]
    },
    options: defaultOptions
  });
}

function initLibraryExamCompChart() {
  if (typeof Chart === 'undefined') { console.warn('Chart.js is not loaded'); return; }
  const ctx = document.getElementById('libraryExamCompChart');
  if (!ctx) return;
  if (chartRegistry.libExamComp) chartRegistry.libExamComp.destroy();

  const slots = ['10:00 AM', '12:00 PM', '02:00 PM', '05:00 PM', '08:00 PM'];
  chartRegistry.libExamComp = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: slots,
      datasets: [
        {
          label: 'Exam Crunch Week (Seats Occupied)',
          data: [265, 290, 335, 340, 310],
          backgroundColor: CHART_COLORS.danger,
          borderRadius: 4
        },
        {
          label: 'Regular Semester (Seats Occupied)',
          data: [120, 145, 180, 195, 140],
          backgroundColor: CHART_COLORS.secondary,
          borderRadius: 4
        }
      ]
    },
    options: defaultOptions
  });
}

function initPcaScatterChart(pcaPoints, profiles, varExplained) {
  if (typeof Chart === 'undefined') { console.warn('Chart.js is not loaded'); return; }
  const ctx = document.getElementById('pcaScatterChart');
  if (!ctx) return;
  if (chartRegistry.pcaScatter) {
    try { chartRegistry.pcaScatter.destroy(); } catch (e) {}
  }

  const clusterColors = ['#f59e0b', '#ef4444', '#06b6d4', '#10b981', '#8b5cf6', '#ec4899'];
  const datasets = [];

  // Group by cluster
  const clusters = {};
  (pcaPoints || []).forEach(p => {
    if (!clusters[p.cluster]) clusters[p.cluster] = [];
    clusters[p.cluster].push({ x: p.pca_x, y: p.pca_y, raw: p });
  });

  Object.keys(clusters).forEach(cid => {
    const color = clusterColors[cid % clusterColors.length];
    const name = profiles[cid]?.label || `Cluster ${cid}`;
    datasets.push({
      label: name,
      data: clusters[cid],
      backgroundColor: color,
      borderColor: color,
      pointRadius: 5,
      pointHoverRadius: 8
    });
  });

  const pc1Pct = varExplained && varExplained[0] ? (varExplained[0] * 100).toFixed(1) : '38.2';
  const pc2Pct = varExplained && varExplained[1] ? (varExplained[1] * 100).toFixed(1) : '24.4';

  chartRegistry.pcaScatter = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: CHART_COLORS.text, font: { family: 'Plus Jakarta Sans', size: 11.5 } }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: function(context) {
              const pt = context.raw.raw;
              return [
                `Cluster: ${pt.cluster_name}`,
                `Sports Load: ${pt.sports_load}% | Library: ${pt.library_load}%`,
                `Food Orders: ${pt.food_orders} | CPI: ${pt.cpi}`
              ];
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: CHART_COLORS.grid },
          ticks: { color: CHART_COLORS.text, font: { family: 'Plus Jakarta Sans', size: 11 } },
          title: { display: true, text: `Principal Component 1 (PC1 - ${pc1Pct}%)`, color: CHART_COLORS.text, font: { family: 'Plus Jakarta Sans', size: 11.5, weight: '600' } }
        },
        y: {
          grid: { color: CHART_COLORS.grid },
          ticks: { color: CHART_COLORS.text, font: { family: 'Plus Jakarta Sans', size: 11 } },
          title: { display: true, text: `Principal Component 2 (PC2 - ${pc2Pct}%)`, color: CHART_COLORS.text, font: { family: 'Plus Jakarta Sans', size: 11.5, weight: '600' } }
        }
      }
    }
  });
}

function initSilhouetteChart(kEval) {
  if (typeof Chart === 'undefined') { console.warn('Chart.js is not loaded'); return; }
  const ctx = document.getElementById('silhouetteChart');
  if (!ctx) return;
  if (chartRegistry.silhouette) {
    try { chartRegistry.silhouette.destroy(); } catch (e) {}
  }

  const list = kEval || [];
  const labels = list.map(e => `K = ${e.k}`);
  const scores = list.map(e => e.silhouette_score);
  const maxScore = scores.length > 0 ? Math.max(...scores) : 0.5;

  chartRegistry.silhouette = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Silhouette Score',
        data: scores,
        backgroundColor: scores.map(s => s === maxScore ? '#10b981' : '#6366f1'),
        borderColor: scores.map(s => s === maxScore ? '#34d399' : '#818cf8'),
        borderWidth: 1.5,
        borderRadius: 5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: (ctx) => `Silhouette Score: ${ctx.parsed.y.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: CHART_COLORS.grid },
          ticks: { color: CHART_COLORS.text, font: { family: 'Plus Jakarta Sans', size: 11 } }
        },
        y: {
          beginAtZero: true,
          suggestedMax: Math.max(0.6, Math.ceil((maxScore + 0.08) * 10) / 10),
          grid: { color: CHART_COLORS.grid },
          ticks: { color: CHART_COLORS.text, font: { family: 'Plus Jakarta Sans', size: 11 } },
          title: { display: true, text: 'Silhouette Coefficient', color: CHART_COLORS.text }
        }
      }
    }
  });
}

function initGlobalImportanceChart(featImportances) {
  if (typeof Chart === 'undefined') { console.warn('Chart.js is not loaded'); return; }
  const ctx = document.getElementById('globalImportanceChart');
  if (!ctx) return;
  if (chartRegistry.globalImportance) {
    try { chartRegistry.globalImportance.destroy(); } catch (e) {}
  }

  const entries = Object.entries(featImportances || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  if (entries.length === 0) return;

  const labels = entries.map(e => e[0].replace(/_/g, ' '));
  const values = entries.map(e => parseFloat(e[1]));

  const barPalette = [
    'rgba(99, 102, 241, 0.9)',
    'rgba(79, 70, 229, 0.85)',
    'rgba(67, 56, 202, 0.85)',
    'rgba(14, 165, 233, 0.85)',
    'rgba(6, 182, 212, 0.85)',
    'rgba(20, 184, 166, 0.85)',
    'rgba(16, 185, 129, 0.85)',
    'rgba(245, 158, 11, 0.85)',
    'rgba(249, 115, 22, 0.85)',
    'rgba(239, 68, 68, 0.85)'
  ];

  chartRegistry.globalImportance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Mean Decrease in Impurity',
        data: values,
        backgroundColor: barPalette.slice(0, values.length),
        borderColor: '#818cf8',
        borderWidth: 1.5,
        borderRadius: 6,
        minBarLength: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: function(context) {
              const val = context.parsed.x;
              return `Importance: ${(val * 100).toFixed(2)}% (MDI: ${val.toFixed(4)})`;
            }
          }
        }
      },
      scales: {
        x: {
          beginAtZero: true,
          grid: { color: CHART_COLORS.grid },
          ticks: {
            color: CHART_COLORS.text,
            font: { family: 'Plus Jakarta Sans', size: 11 },
            callback: (v) => `${(v * 100).toFixed(0)}%`
          },
          title: {
            display: true,
            text: 'Mean Decrease in Impurity (Relative Weight %)',
            color: CHART_COLORS.text,
            font: { family: 'Plus Jakarta Sans', size: 11.5, weight: '600' }
          }
        },
        y: {
          grid: { display: false },
          ticks: {
            color: '#e2e8f0',
            font: { family: 'Plus Jakarta Sans', size: 11, weight: '500' }
          }
        }
      }
    }
  });
}
