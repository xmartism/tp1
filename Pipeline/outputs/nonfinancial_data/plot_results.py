"""
Vygeneruje HTML stránku s grafmi pre každú metriku (MSE, MAE, MAPE, MDA,Train Time, Epochs) naprieč skupinami experimentov.

Použitie (z aktualneho priecinku):
    python3 plot_results.py aggregated_results.csv

Argumenty:
    aggregated_results.csv  výstup zo skriptu aggregate_results.py

Výstup:
    results_charts.html
"""
import pandas as pd
import json
import sys
from pathlib import Path

CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("aggregated_results.csv")
OUT_PATH = CSV_PATH.parent / "results_charts.html"

METRICS = [
    ("Train Time (s)", "Train time (s)", "lower"),
    ("Epochs Trained",  "Epochs trained",  "lower"),
    ("MSE",             "MSE",             "lower"),
    ("MAE",             "MAE",             "lower"),
    ("MAPE (%)",        "MAPE (%)",        "lower"),
    ("MDA (%)",         "MDA (%)",         "higher"),
]

MODELS  = ["deepAR", "tft", "tsmixer", "nbeats", "dlinear"]
COLORS  = {
    "deepAR":  "#378ADD",
    "tft":     "#1D9E75",
    "tsmixer": "#D85A30",
    "nbeats":  "#7F77DD",
    "dlinear": "#888780",
}
DASHES  = {
    "deepAR":  "[]",
    "tft":     "[6,3]",
    "tsmixer": "[2,2]",
    "nbeats":  "[8,4]",
    "dlinear": "[4,4,2,4]",
}

df = pd.read_csv(CSV_PATH)
experiment_labels = [str(d) for d in df["dirs"].unique()]

def build_datasets(col_mean, col_std):
    datasets = []
    for m in MODELS:
        rows = df[df["Model"] == m].set_index("dirs")
        mean_vals = [
            round(float(rows.loc[d, col_mean]), 4) if d in rows.index else None
            for d in df["dirs"].unique()
        ]
        std_vals = [
            round(float(rows.loc[d, col_std]), 4) if d in rows.index else None
            for d in df["dirs"].unique()
        ]
        datasets.append({
            "label":           m,
            "data":            mean_vals,
            "std":             std_vals,
            "borderColor":     COLORS[m],
            "backgroundColor": COLORS[m] + "33",
            "borderDash":      json.loads(DASHES[m]),
            "borderWidth":     2,
            "pointRadius":     4,
            "tension":         0.3,
            "fill":            False,
        })
    return datasets

charts_data = []
for col, label, direction in METRICS:
    col_mean = f"{col}_mean"
    col_std  = f"{col}_std"
    if col_mean not in df.columns:
        continue
    log_scale = col in ("MSE", "MAE", "Train Time (s)", "Epochs Trained")
    charts_data.append({
        "id":        col.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct"),
        "label":     label,
        "direction": direction,
        "logScale":  log_scale,
        "datasets":  build_datasets(col_mean, col_std),
    })

labels_json    = json.dumps(experiment_labels)
charts_json    = json.dumps(charts_data, ensure_ascii=False)
models_json    = json.dumps(MODELS)
colors_json    = json.dumps(COLORS)

HTML = f"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model comparison</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #f8f7f4; color: #2c2c2a; padding: 2rem 1rem; }}
  h1   {{ font-size: 1.3rem; font-weight: 500; margin-bottom: 0.4rem; color: #2c2c2a; }}
  p.sub {{ font-size: 0.85rem; color: #73726c; margin-bottom: 2rem; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 1.5rem; font-size: 12px; color: #5f5e5a; }}
  .legend span {{ display: flex; align-items: center; gap: 5px; }}
  .legend i {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; }}
  .card {{ background: #fff; border: 0.5px solid #d3d1c7; border-radius: 12px; padding: 1.25rem 1.5rem; }}
  .card-title {{ font-size: 0.78rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em;
                 color: #5f5e5a; margin-bottom: 4px; }}
  .card-hint  {{ font-size: 0.75rem; color: #888780; margin-bottom: 1rem; }}
  .chart-wrap {{ position: relative; width: 100%; height: 260px; }}
  canvas {{ display: block; }}
</style>
</head>
<body>

<h1>Model comparison — all metrics</h1>
<p class="sub">Priemer ± std naprieč 3 seedmi. Osa Y je logaritmická tam kde rozsah hodnôt je veľký.</p>

<div class="legend" id="legend"></div>
<div class="grid"   id="grid"></div>

<script>
const LABELS  = {labels_json};
const CHARTS  = {charts_json};
const MODELS  = {models_json};
const COLORS  = {colors_json};

function buildLegend() {{
  const el = document.getElementById('legend');
  MODELS.forEach(m => {{
    const s = document.createElement('span');
    s.innerHTML = `<i style="background:${{COLORS[m]}}"></i>${{m}}`;
    el.appendChild(s);
  }});
}}

function makeErrorBarPlugin() {{
  return {{
    id: 'errorBars',
    afterDatasetsDraw(chart) {{
      const ctx = chart.ctx;
      chart.data.datasets.forEach((ds, di) => {{
        if (!ds.std) return;
        const meta = chart.getDatasetMeta(di);
        meta.data.forEach((pt, i) => {{
          const std = ds.std[i];
          if (std == null || std === 0) return;
          const x  = pt.x;
          const yScale = chart.scales.y;
          const mean = ds.data[i];
          if (mean == null) return;

          let yTop, yBot;
          if (yScale.type === 'logarithmic') {{
            const lo = Math.max(mean - std, mean * 0.01);
            yTop = yScale.getPixelForValue(mean + std);
            yBot = yScale.getPixelForValue(lo);
          }} else {{
            yTop = yScale.getPixelForValue(mean + std);
            yBot = yScale.getPixelForValue(mean - std);
          }}

          ctx.save();
          ctx.strokeStyle = ds.borderColor;
          ctx.lineWidth   = 1;
          ctx.globalAlpha = 0.55;
          ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBot); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(x - 4, yTop); ctx.lineTo(x + 4, yTop); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(x - 4, yBot); ctx.lineTo(x + 4, yBot); ctx.stroke();
          ctx.restore();
        }});
      }});
    }}
  }};
}}

function renderCharts() {{
  const grid   = document.getElementById('grid');
  const plugin = makeErrorBarPlugin();
  Chart.register(plugin);

  CHARTS.forEach(c => {{
    const card = document.createElement('div');
    card.className = 'card';
    const hint = c.direction === 'lower' ? 'nižšie = lepšie' : 'vyššie = lepšie';
    card.innerHTML = `
      <div class="card-title">${{c.label}}</div>
      <div class="card-hint">${{hint}}${{c.logScale ? ' · log škála' : ''}}</div>
      <div class="chart-wrap"><canvas id="${{c.id}}" role="img"
        aria-label="Line chart of ${{c.label}} per experiment group"></canvas></div>`;
    grid.appendChild(card);

    new Chart(document.getElementById(c.id), {{
      type: 'line',
      data: {{ labels: LABELS, datasets: c.datasets }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              label: ctx => {{
                const std = ctx.dataset.std?.[ctx.dataIndex];
                const v   = ctx.parsed.y;
                const fmt = n => n >= 1000
                  ? n.toLocaleString('sk', {{maximumFractionDigits: 0}})
                  : n.toLocaleString('sk', {{maximumFractionDigits: 2}});
                return `${{ctx.dataset.label}}: ${{fmt(v)}}${{std != null ? ' ± ' + fmt(std) : ''}}`;
              }}
            }}
          }}
        }},
        scales: {{
          y: {{
            type: c.logScale ? 'logarithmic' : 'linear',
            ticks: {{
              font: {{ size: 11 }},
              callback: v => v >= 10000
                ? (v/1000).toFixed(0)+'k'
                : v >= 1000 ? v.toLocaleString('sk') : +v.toPrecision(4)
            }},
            grid: {{ color: 'rgba(0,0,0,0.05)' }}
          }},
          x: {{
            ticks: {{ font: {{ size: 11 }}, maxRotation: 30 }},
            grid: {{ display: false }}
          }}
        }}
      }}
    }});
  }});
}}

buildLegend();
renderCharts();
</script>
</body>
</html>
"""

OUT_PATH.write_text(HTML, encoding="utf-8")
print(f"Saved: {OUT_PATH}")