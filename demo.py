"""Generate a self-contained interactive HTML explorer for the retained capability-accounting tables.

Requires Python 3.10+ standard library only. No external packages, GPU, or API keys needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import webbrowser
from pathlib import Path

ROOT = Path(__file__).absolute().parent
DATA_DIR = ROOT / "data" / "retained_tables"
OUT_DIR = ROOT / "demo_output"


def load_csv(rel_path: str) -> tuple[list[dict[str, str]], str]:
    p = DATA_DIR / rel_path
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, sha


def build_explorer_html() -> Path:
    bench_rows, bench_sha = load_csv("tables/table_profile_distance_benchmark.csv")
    cal_rows, cal_sha = load_csv("tables/table_profile_caliper_sensitivity_display.csv")
    title_rows, title_sha = load_csv("tables/table_profile_title_stability_sensitivity.csv")
    pair_rows, pair_sha = load_csv("illustrative_pair_selection_manifest.csv")
    ext_rows, ext_sha = load_csv("tables/table_profile_external_association.csv")

    payload = {
        "benchmark": {"rows": bench_rows, "sha256": bench_sha},
        "caliper": {"rows": cal_rows, "sha256": cal_sha},
        "title_stability": {"rows": title_rows, "sha256": title_sha},
        "quartiles": {"rows": pair_rows, "sha256": pair_sha},
        "external": {"rows": ext_rows, "sha256": ext_sha},
        "case_study": {
            "pair": "Rehabilitation Counselors (SOC 21-1015) vs. Optometrists (SOC 29-1041)",
            "published_scores": [0.503, 0.504],
            "full_precision_gap": 0.0015,
            "jensen_shannon_distance": 0.4414,
            "abilities": [
                {"name": "Information Ordering", "counselors": 0.191, "optometrists": 0.215, "channel": "Exposure-Amplifying"},
                {"name": "Deductive Reasoning", "counselors": 0.123, "optometrists": 0.122, "channel": "Exposure-Amplifying"},
                {"name": "Inductive Reasoning", "counselors": 0.117, "optometrists": 0.120, "channel": "Exposure-Amplifying"},
                {"name": "Speech Recognition", "counselors": 0.119, "optometrists": 0.078, "channel": "Exposure-Amplifying"},
                {"name": "Arm-Hand Steadiness", "counselors": -0.013, "optometrists": -0.181, "channel": "Exposure-Buffering"},
                {"name": "Finger Dexterity", "counselors": -0.098, "optometrists": -0.170, "channel": "Exposure-Buffering"},
            ],
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Occupational AI Exposure & Signed Capability Balance Sheet Explorer</title>
<style>
  :root {{
    --bg: #f8fafc; --card: #ffffff; --ink: #0f172a; --muted: #475569;
    --accent: #1d4ed8; --pos: #0284c7; --neg: #dc2626; --border: #e2e8f0;
  }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--ink); margin: 0; padding: 24px; line-height: 1.5; }}
  .container {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ font-size: 1.65rem; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 20px; font-size: 0.95rem; }}
  .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
  .tab-btn {{ padding: 8px 14px; border: 1px solid var(--border); background: var(--card); border-radius: 6px; cursor: pointer; font-weight: 600; color: var(--muted); }}
  .tab-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 10px; }}
  th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border); }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  .sha {{ font-family: monospace; font-size: 0.78rem; color: var(--muted); margin-top: 10px; }}
  .bar-row {{ display: flex; align-items: center; gap: 12px; margin: 8px 0; font-size: 0.88rem; }}
  .bar-label {{ width: 190px; font-weight: 500; }}
  .bar-val {{ font-family: monospace; width: 140px; }}
  .badge-pos {{ color: var(--pos); font-weight: 600; }}
  .badge-neg {{ color: var(--neg); font-weight: 600; }}
</style>
</head>
<body>
<div class="container">
  <h1>Occupational AI Exposure: Signed Capability Balance Sheet Explorer</h1>
  <div class="subtitle">Interactive local viewer for retained capability-accounting tables, chronological O*NET revision benchmarks, and case decompositions.</div>
  <div class="tabs">
    <button class="tab-btn active" onclick="showTab('benchmark', this)">1. Iso-Exposure vs. O*NET Revisions</button>
    <button class="tab-btn" onclick="showTab('case', this)">2. Case Study & Quartile Pairs</button>
    <button class="tab-btn" onclick="showTab('caliper', this)">3. Caliper & Title-Stability Sensitivity</button>
    <button class="tab-btn" onclick="showTab('external', this)">4. ACS & OEWS Wage Regressions</button>
  </div>
  <div id="content"></div>
</div>
<script>
const DATA = {json.dumps(payload)};
function renderTable(rows) {{
  if (!rows || !rows.length) return '';
  const cols = Object.keys(rows[0]);
  let h = '<table><thead><tr>' + cols.map(c => `<th>${{c}}</th>`).join('') + '</tr></thead><tbody>';
  for (const r of rows) {{
    h += '<tr>' + cols.map(c => `<td>${{r[c] ?? ''}}</td>`).join('') + '</tr>';
  }}
  return h + '</tbody></table>';
}}
function showTab(tab, btn) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  const el = document.getElementById('content');
  if (tab === 'benchmark') {{
    el.innerHTML = `<div class="card"><h3>Compositional Distance Among Near-Score Pairs vs. Longitudinal O*NET Revisions</h3>
      <p>Across 983 cross-major occupation pairs within 0.025 SD on both published and reconstructed AIOE, median base-2 Jensen–Shannon distance is <b>0.1731</b> (exceeding the 90th percentile of O*NET 24.3–27.1 and 24.3–30.3 skill updates for <b>67.7%</b> and <b>61.3%</b> of pairs).</p>
      ${{renderTable(DATA.benchmark.rows)}}
      <div class="sha">Source: tables/table_profile_distance_benchmark.csv | SHA-256: ${{DATA.benchmark.sha256}}</div></div>`;
  }} else if (tab === 'case') {{
    const cs = DATA.case_study;
    let bars = cs.abilities.map(a => `<div class="bar-row"><div class="bar-label">${{a.name}}</div>
      <div class="bar-val">Counselors: <span class="${{a.counselors>=0?'badge-pos':'badge-neg'}}">${{a.counselors>=0?'+':''}}${{a.counselors.toFixed(3)}}</span></div>
      <div class="bar-val">Optometrists: <span class="${{a.optometrists>=0?'badge-pos':'badge-neg'}}">${{a.optometrists>=0?'+':''}}${{a.optometrists.toFixed(3)}}</span></div>
      <div>(${{a.channel}})</div></div>`).join('');
    el.innerHTML = `<div class="card"><h3>Signed Capability Balance Sheet: ${{cs.pair}}</h3>
      <p>Published AIOE: <b>0.503 vs. 0.504</b> (gap = 0.0015 SD) | 52-Ability Jensen–Shannon Distance: <b>0.4414</b></p>${{bars}}</div>
      <div class="card"><h3>Illustrative Near-Score Pairs Across All Four AIOE Quartiles</h3>
      ${{renderTable(DATA.quartiles.rows)}}
      <div class="sha">Source: illustrative_pair_selection_manifest.csv | SHA-256: ${{DATA.quartiles.sha256}}</div></div>`;
  }} else if (tab === 'caliper') {{
    el.innerHTML = `<div class="card"><h3>Panel A: Score-Proximity Caliper & Major-Group Sensitivity</h3>
      ${{renderTable(DATA.caliper.rows)}}
      <div class="sha">Source: tables/table_profile_caliper_sensitivity_display.csv | SHA-256: ${{DATA.caliper.sha256}}</div></div>
      <div class="card"><h3>Panel B: Title-Stable Longitudinal O*NET Revision Benchmarks</h3>
      ${{renderTable(DATA.title_stability.rows)}}
      <div class="sha">Source: tables/table_profile_title_stability_sensitivity.csv | SHA-256: ${{DATA.title_stability.sha256}}</div></div>`;
  }} else if (tab === 'external') {{
    el.innerHTML = `<div class="card"><h3>Bifurcated Adjustment Regime: Short-Run Occupational Wage Regressions (ACS & OEWS)</h3>
      ${{renderTable(DATA.external.rows)}}
      <div class="sha">Source: tables/table_profile_external_association.csv | SHA-256: ${{DATA.external.sha256}}</div></div>`;
  }}
}}
showTab('benchmark');
</script>
</body>
</html>"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and open the interactive capability-accounting results viewer.")
    parser.add_argument("--no-open", action="store_true", help="Generate HTML without opening a browser.")
    args = parser.parse_args()
    out_path = build_explorer_html()
    print(f"Generated interactive explorer at: {out_path}")
    if not args.no_open:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
