import pandas as pd
from pathlib import Path

csv_path = Path("results/metrics/test_predictions.csv")
df = pd.read_csv(csv_path)

rows_html = []
for _, r in df.iterrows():
    p = str(r["Priority Status"]).lower()
    if "healthy" in p:
        b_cls = "badge-healthy"
    elif "critical" in p:
        b_cls = "badge-critical"
    elif "high" in p:
        b_cls = "badge-high"
    else:
        b_cls = "badge-moderate"

    delta = float(r["Delta Conf (INT8-FP32) (%)"])
    delta_col = "#34d399" if delta >= 0 else "#fbbf24"

    row_str = f"""          <tr>
            <td>{int(r['ID']):02d}</td>
            <td><a class="img-link" href="../test_predictions/{r['Raw Image (Without Predictions)']}" target="_blank"><img class="img-thumb" src="../test_predictions/{r['Raw Image (Without Predictions)']}" alt="raw"> {r['Image Filename']}</a></td>
            <td><a class="img-link" href="../test_predictions/{r['Annotated Image (With Predictions)']}" target="_blank"><img class="img-thumb" src="../test_predictions/{r['Annotated Image (With Predictions)']}" alt="pred"> {r['Annotated Image (With Predictions)']}</a></td>
            <td><strong>{r['PyTorch Baseline Class']}</strong></td>
            <td><div class="conf-bar"><div class="conf-fill" style="width: {r['PyTorch Conf (%)']}%;"></div></div>{r['PyTorch Conf (%)']:.2f}%</td>
            <td><strong>{r['ONNX FP32 Conf (%)']:.2f}%</strong></td>
            <td><strong style="color:#38bdf8;">{r['ONNX INT8 Conf (%)']:.2f}%</strong></td>
            <td style="color:{delta_col}; font-weight:600;">{delta:+.2f}%</td>
            <td><span class="badge {b_cls}">{r['Priority Status']}</span></td>
          </tr>"""
    rows_html.append(row_str)

body_rows = "\n".join(rows_html)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PaddySnap AI — Test Evaluation Results (Baseline vs ONNX FP32 vs INT8)</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #10b981;
      --danger: #ef4444;
      --warning: #f59e0b;
      --info: #38bdf8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 30px 20px; }}
    .container {{ max-width: 1350px; margin: 0 auto; }}
    header {{ text-align: center; margin-bottom: 30px; }}
    header h1 {{ font-size: 2.2rem; color: #ffffff; display: flex; align-items: center; justify-content: center; gap: 10px; }}
    header p {{ color: var(--text-muted); margin-top: 8px; font-size: 1rem; }}
    
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 30px; }}
    .stat-card {{ background: var(--card-bg); border: 1px solid var(--border); padding: 20px; border-radius: 12px; text-align: center; }}
    .stat-card h3 {{ font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; letter-spacing: 0.5px; }}
    .stat-card .val {{ font-size: 1.8rem; font-weight: bold; color: #fff; }}
    .stat-card .val.green {{ color: var(--accent); }}
    .stat-card .val.blue {{ color: var(--info); }}
    
    .table-container {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 40px; }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.90rem; }}
    th {{ background: #182234; padding: 14px 14px; color: var(--text-muted); font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.5px; }}
    td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover {{ background: rgba(255, 255, 255, 0.03); }}
    
    .img-thumb {{ width: 52px; height: 70px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); transition: transform 0.2s ease, box-shadow 0.2s ease; cursor: pointer; display: inline-block; vertical-align: middle; }}
    .img-thumb:hover {{ transform: scale(2.3); z-index: 99; box-shadow: 0 10px 25px rgba(0,0,0,0.7); position: relative; border-color: var(--accent); }}
    .img-link {{ text-decoration: none; display: flex; align-items: center; gap: 8px; color: #cbd5e1; font-weight: 500; font-size: 0.88rem; }}
    .img-link:hover {{ color: var(--accent); }}

    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; }}
    .badge-healthy {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .badge-critical {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
    .badge-high {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .badge-moderate {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
    
    .conf-bar {{ background: #334155; height: 7px; border-radius: 4px; width: 60px; overflow: hidden; display: inline-block; vertical-align: middle; margin-right: 6px; }}
    .conf-fill {{ height: 100%; background: var(--accent); border-radius: 4px; }}
    
    .gallery-preview {{ margin-top: 20px; }}
    .gallery-preview h2 {{ font-size: 1.4rem; margin-bottom: 16px; color: #fff; }}
    .tab-bar {{ display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }}
    .tab-btn {{ background: var(--card-bg); border: 1px solid var(--border); color: var(--text-muted); padding: 10px 18px; border-radius: 8px; cursor: pointer; font-size: 0.90rem; font-weight: 600; transition: all 0.2s; }}
    .tab-btn.active {{ background: #2563eb; color: #fff; border-color: #3b82f6; }}
    .collage-box {{ background: var(--card-bg); border: 1px solid var(--border); padding: 16px; border-radius: 12px; text-align: center; }}
    .collage-box img {{ max-width: 100%; border-radius: 8px; border: 1px solid var(--border); }}
    .grid-view {{ display: none; }}
    .grid-view.active {{ display: block; }}
    .compare-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .compare-col h3 {{ margin-bottom: 10px; font-size: 1.1rem; color: #cbd5e1; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🌾 PaddySnap AI</h1>
      <p>Test Set Multi-Model Clinical Evaluation (PyTorch Baseline vs ONNX FP32 vs ONNX INT8)</p>
    </header>

    <div class="stats-grid">
      <div class="stat-card">
        <h3>Champion Architecture</h3>
        <div class="val blue">EfficientNetV2-S</div>
      </div>
      <div class="stat-card">
        <h3>Model Parity Rate</h3>
        <div class="val green">100.0% (20/20)</div>
      </div>
      <div class="stat-card">
        <h3>INT8 Size Reduction</h3>
        <div class="val green">73.9% (20.07 MB)</div>
      </div>
      <div class="stat-card">
        <h3>Mean Conf. Drift</h3>
        <div class="val">1.44%</div>
      </div>
    </div>

    <!-- Collages & Infographic Section -->
    <div class="gallery-preview">
      <h2>📸 Visual Artifacts & Benchmarks</h2>
      <div class="tab-bar">
        <button class="tab-btn active" onclick="showTab('tab-infographic')">📊 FP32 vs INT8 Benchmark Infographic</button>
        <button class="tab-btn" onclick="showTab('tab-master')">🖼️ All 20 in One PNG (Master Comparison)</button>
        <button class="tab-btn" onclick="showTab('tab-compare')">⚖️ Side-by-Side Dual Grids</button>
        <button class="tab-btn" onclick="showTab('tab-raw')">🌿 Raw Images (Without Predictions)</button>
        <button class="tab-btn" onclick="showTab('tab-pred')">🏷️ Diagnosed Images (With Predictions)</button>
      </div>

      <div id="tab-infographic" class="grid-view active collage-box">
        <h3 style="margin-bottom:12px; color:#cbd5e1;">📊 ONNX FP32 vs INT8 Parity & Quantization Benchmark Infographic</h3>
        <p style="color:#94a3b8; margin-bottom:14px; font-size:0.9rem;">Click image to open full high-resolution visual comparison</p>
        <a href="../graphs/onnx_fp32_vs_int8_visual_comparison.png" target="_blank">
          <img src="../graphs/onnx_fp32_vs_int8_visual_comparison.png" alt="ONNX FP32 vs INT8 Visual Benchmark">
        </a>
      </div>

      <div id="tab-master" class="grid-view collage-box">
        <h3 style="margin-bottom:12px; color:#cbd5e1;">🖼️ Master 20-Sample Diagnostic Grid (All in One PNG)</h3>
        <p style="color:#94a3b8; margin-bottom:14px; font-size:0.9rem;">Click image to open full 3690x2130 high-resolution view in a new tab</p>
        <a href="../test_predictions/all_20_raw_vs_predicted_grid.png" target="_blank">
          <img src="../test_predictions/all_20_raw_vs_predicted_grid.png" alt="All 20 Raw vs Predicted Grid">
        </a>
      </div>

      <div id="tab-compare" class="grid-view collage-box">
        <div class="compare-row">
          <div class="compare-col">
            <h3>🌿 Clean Field Images (Without Predictions)</h3>
            <a href="../test_predictions/raw_20_images_grid.png" target="_blank">
              <img src="../test_predictions/raw_20_images_grid.png" alt="Raw 20 Images Grid">
            </a>
          </div>
          <div class="compare-col">
            <h3>🏷️ AI Diagnosed Predictions (With Overlay & Confidence)</h3>
            <a href="../test_predictions/all_20_predictions_grid.png" target="_blank">
              <img src="../test_predictions/all_20_predictions_grid.png" alt="All 20 Predictions Grid">
            </a>
          </div>
        </div>
      </div>

      <div id="tab-raw" class="grid-view collage-box">
        <h3 style="margin-bottom:12px; color:#cbd5e1;">🌿 20 Clean Test Images (Without Predictions)</h3>
        <a href="../test_predictions/raw_20_images_grid.png" target="_blank">
          <img src="../test_predictions/raw_20_images_grid.png" alt="Raw 20 Images Grid">
        </a>
      </div>

      <div id="tab-pred" class="grid-view collage-box">
        <h3 style="margin-bottom:12px; color:#cbd5e1;">🏷️ 20 AI Diagnosed Test Images (With Overlays)</h3>
        <a href="../test_predictions/all_20_predictions_grid.png" target="_blank">
          <img src="../test_predictions/all_20_predictions_grid.png" alt="All 20 Predictions Grid">
        </a>
      </div>
    </div>

    <br><br>

    <!-- Table Section -->
    <h2 style="font-size: 1.4rem; margin-bottom: 16px; color: #fff;">📊 Multi-Model Diagnostic Log & Confidence Comparison (20 Samples)</h2>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Raw Field Leaf</th>
            <th>Annotated Overlay</th>
            <th>Diagnosis</th>
            <th>PyTorch (Baseline)</th>
            <th>ONNX FP32</th>
            <th>ONNX INT8</th>
            <th>Δ Conf (INT8-FP32)</th>
            <th>Action Status</th>
          </tr>
        </thead>
        <tbody>
{body_rows}
        </tbody>
      </table>
    </div>
  </div>

  <script>
    function showTab(tabId) {{
      document.querySelectorAll('.grid-view').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      event.currentTarget.classList.add('active');
    }}
  </script>
</body>
</html>
"""

with open("results/metrics/test_results.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("results/metrics/test_results.html successfully written!")
