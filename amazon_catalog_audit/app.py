"""
Catalog Health Checker — standalone single-page tool.

Upload a Category Listing Report (CLR) and get a plain-English health report
back. Optionally also upload a Business Report to flag zero-sales SKUs for
deletion. No Helium10 keyword file needed for this simplified version.

Run with:  python3 -m amazon_catalog_audit.app
Then open: http://127.0.0.1:8800
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from amazon_catalog_audit.audit_engine import run_audit

app = FastAPI(title="Catalog Health Checker")

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Catalog Health Checker</title>
<style>
  :root {
    --bg-a: #f3effb;
    --bg-b: #eef6f6;
    --card: #ffffff;
    --border: #e7e3f3;
    --text: #1f1b2e;
    --muted: #6e6a85;
    --brand: #7c3aed;
    --brand-2: #c026d3;
    --brand-dark: #6d28d9;
    --teal: #0d9488;
    --good: #0f9d6e;
    --good-bg: #e7faf1;
    --warn: #d97706;
    --warn-bg: #fff7e8;
    --bad: #e11d48;
    --bad-bg: #fdeef1;
    --radius: 16px;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: linear-gradient(160deg, var(--bg-a) 0%, var(--bg-b) 55%, #fdf2f8 100%);
    color: var(--text);
    margin: 0;
    padding: 0 20px 90px;
    min-height: 100vh;
  }
  .wrap { max-width: 960px; margin: 0 auto; position: relative; }

  .hero {
    position: relative;
    padding: 56px 10px 40px;
    text-align: center;
    overflow: hidden;
  }
  .hero .blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(50px);
    opacity: 0.35;
    z-index: 0;
  }
  .hero .blob.b1 { width: 280px; height: 280px; background: var(--brand); top: -90px; left: -60px; }
  .hero .blob.b2 { width: 260px; height: 260px; background: var(--teal); top: -60px; right: -80px; }
  .hero .blob.b3 { width: 220px; height: 220px; background: var(--brand-2); bottom: -120px; left: 40%; }
  .hero-inner { position: relative; z-index: 1; }
  .hero h1 {
    font-size: 2.2em;
    margin: 0 0 10px;
    font-weight: 800;
    background: linear-gradient(100deg, var(--brand) 0%, var(--brand-2) 60%, var(--teal) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .hero p { color: var(--muted); margin: 0; font-size: 1.08em; max-width: 560px; margin: 0 auto; }

  .upload-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 30px;
    box-shadow: 0 10px 30px -12px rgba(124, 58, 237, 0.18);
    position: relative;
    z-index: 1;
  }
  .upload-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 18px;
  }
  .upload-slot { display: flex; flex-direction: column; }
  .slot-title { font-weight: 700; font-size: 0.92em; margin-bottom: 8px; }
  .slot-title .required { color: var(--bad); font-weight: 600; font-size: 0.8em; }
  .slot-title .optional { color: var(--muted); font-weight: 500; font-size: 0.8em; }
  .drop {
    border: 2px dashed #d8cdf2;
    border-radius: var(--radius);
    padding: 36px 20px;
    text-align: center;
    color: var(--muted);
    cursor: pointer;
    transition: border-color .15s, background .15s, transform .1s;
    flex: 1;
    background: linear-gradient(160deg, #fbfaff 0%, #f7fbfb 100%);
  }
  .drop:hover { transform: translateY(-1px); }
  .drop.over { border-color: var(--brand); background: #f3ebff; }
  .drop .icon { font-size: 2.2em; margin-bottom: 8px; }
  .drop p { margin: 4px 0; font-size: 0.92em; }
  .drop .hint { font-size: 0.8em; color: #a39fc1; }
  input[type=file] { display: none; }

  .actions { display: flex; justify-content: center; margin-top: 22px; }
  button {
    padding: 13px 30px;
    font-size: 1em;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    background: linear-gradient(100deg, var(--brand) 0%, var(--brand-2) 100%);
    color: white;
    cursor: pointer;
    transition: filter .15s, transform .1s, box-shadow .15s;
    box-shadow: 0 8px 20px -8px rgba(124, 58, 237, 0.55);
  }
  button:hover:not(:disabled) { filter: brightness(1.06); transform: translateY(-1px); }
  button:disabled { background: #d4d1e3; box-shadow: none; cursor: default; }

  #status { text-align: center; margin-top: 16px; color: var(--muted); font-size: 0.95em; }
  #status.error { color: var(--bad); font-weight: 600; }

  #report { margin-top: 36px; }

  .score-banner {
    display: flex;
    align-items: center;
    gap: 26px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 30px;
    margin-bottom: 26px;
    box-shadow: 0 10px 30px -14px rgba(31, 27, 46, 0.12);
  }
  .score-circle {
    flex-shrink: 0;
    width: 116px; height: 116px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.1em; font-weight: 800;
    border: 7px solid;
    background: white;
  }
  .score-circle.good { color: var(--good); border-color: var(--good); background: var(--good-bg); }
  .score-circle.warn { color: var(--warn); border-color: var(--warn); background: var(--warn-bg); }
  .score-circle.bad { color: var(--bad); border-color: var(--bad); background: var(--bad-bg); }
  .score-banner .label { font-size: 1.25em; font-weight: 800; margin-bottom: 4px; }
  .score-banner .sub { color: var(--muted); font-size: 0.95em; }

  h2.section-title {
    font-size: 1.3em;
    margin: 34px 0 14px;
    display: flex; align-items: center; gap: 8px;
    font-weight: 800;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 16px;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    cursor: pointer;
    transition: transform .12s, box-shadow .12s, border-color .12s;
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: var(--accent);
  }
  .card:hover { transform: translateY(-3px); box-shadow: 0 14px 28px -16px rgba(31,27,46,0.22); border-color: #d8cdf2; }
  .card.active { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent); }
  .card .top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  .card .name { font-weight: 700; font-size: 0.95em; display: flex; align-items: center; gap: 7px; }
  .card .icon-chip {
    width: 28px; height: 28px;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95em;
    background: color-mix(in srgb, var(--accent) 16%, white);
    flex-shrink: 0;
  }
  .card .pill {
    font-size: 0.78em;
    font-weight: 800;
    padding: 3px 10px;
    border-radius: 999px;
    flex-shrink: 0;
  }
  .pill.good { color: var(--good); background: var(--good-bg); }
  .pill.warn { color: var(--warn); background: var(--warn-bg); }
  .pill.bad { color: var(--bad); background: var(--bad-bg); }
  .card .bar-track { height: 6px; background: #f0edf9; border-radius: 6px; margin: 13px 0 9px; overflow: hidden; }
  .card .bar-fill { height: 100%; border-radius: 6px; }
  .card .bar-fill.good { background: var(--good); }
  .card .bar-fill.warn { background: var(--warn); }
  .card .bar-fill.bad { background: var(--bad); }
  .card .count { color: var(--muted); font-size: 0.85em; }
  .card .note {
    margin-top: 8px;
    font-size: 0.78em;
    color: #8b5cf6;
    background: #f5f0ff;
    border-radius: 8px;
    padding: 6px 9px;
    line-height: 1.4;
  }
  .card .click-hint { font-size: 0.72em; color: #b3aed1; margin-top: 8px; }

  .filter-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(100deg, #f3ebff, #eafaf7);
    border: 1px solid #e3d8fb;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 14px;
    font-size: 0.92em;
  }
  .filter-banner button.clear {
    background: white;
    color: var(--brand-dark);
    border: 1px solid #d8cdf2;
    box-shadow: none;
    font-size: 0.85em;
    padding: 6px 14px;
    font-weight: 700;
  }
  .filter-banner button.clear:hover { background: #f7f2ff; filter: none; transform: none; }

  .table-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: 0 10px 30px -16px rgba(31,27,46,0.12);
  }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 13px 16px; font-size: 0.92em; vertical-align: top; }
  thead th {
    background: #faf8ff;
    color: var(--muted);
    font-weight: 700;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    border-bottom: 1px solid var(--border);
  }
  tbody tr { border-bottom: 1px solid var(--border); }
  tbody tr:last-child { border-bottom: none; }
  tbody tr:hover { background: #fbf9ff; }
  .title-cell { max-width: 280px; }
  .sev-badge {
    display: inline-block;
    font-size: 0.78em;
    font-weight: 800;
    padding: 3px 11px;
    border-radius: 999px;
    white-space: nowrap;
  }
  .sev-Critical { color: var(--bad); background: var(--bad-bg); }
  .sev-High { color: var(--warn); background: var(--warn-bg); }
  .sev-Medium { color: #5b21b6; background: #f1eaff; }
  .sev-Low { color: var(--muted); background: #f1f0f5; }
  ul.issues { margin: 0; padding-left: 16px; color: var(--text); }
  ul.issues li { margin-bottom: 3px; }
  .advice { color: var(--muted); font-size: 0.88em; margin-top: 6px; font-style: italic; }

  .empty-state {
    text-align: center;
    padding: 50px 20px;
    color: var(--muted);
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }
  .empty-state .icon { font-size: 2.4em; margin-bottom: 8px; }

  .spinner {
    width: 18px; height: 18px;
    border: 3px solid #ece6fa;
    border-top-color: var(--brand);
    border-radius: 50%;
    display: inline-block;
    vertical-align: middle;
    margin-right: 8px;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="blob b1"></div>
    <div class="blob b2"></div>
    <div class="blob b3"></div>
    <div class="hero-inner">
      <h1>Catalog Health Checker</h1>
      <p>Upload your Category Listing Report and get an instant catalog health breakdown — titles, bullet points, descriptions, variations, and attributes.</p>
    </div>
  </div>

  <div class="upload-card">
    <div class="upload-grid">
      <div class="upload-slot">
        <div class="slot-title">Category Listing Report <span class="required">(required)</span></div>
        <div class="drop" id="drop-clr" data-target="clr">
          <div class="icon">📦</div>
          <p><strong>Click to browse</strong> or drag your file here</p>
          <p class="hint">Accepted formats: .xlsx, .csv, .txt</p>
        </div>
        <input type="file" id="file-clr" accept=".xlsx,.csv,.txt">
      </div>
      <div class="upload-slot">
        <div class="slot-title">Business Report (Child SKU, YTD) <span class="optional">(optional)</span></div>
        <div class="drop" id="drop-br" data-target="br">
          <div class="icon">💰</div>
          <p><strong>Click to browse</strong> or drag your file here</p>
          <p class="hint">Used to flag zero-sales SKUs for deletion</p>
        </div>
        <input type="file" id="file-br" accept=".xlsx,.csv,.txt">
      </div>
    </div>
    <div class="actions">
      <button id="go" disabled>Analyze Catalog</button>
    </div>
    <div id="status"></div>
  </div>

  <div id="report"></div>
</div>

<script>
const goBtn = document.getElementById('go');
const statusEl = document.getElementById('status');
const reportEl = document.getElementById('report');
let chosenFiles = { clr: null, br: null };
let lastData = null;
let activeFilter = null;

function wireDropzone(dropId, inputId, key) {
  const drop = document.getElementById(dropId);
  const fileInput = document.getElementById(inputId);
  drop.addEventListener('click', () => fileInput.click());
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('over');
    if (e.dataTransfer.files.length) setFile(key, drop, e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) setFile(key, drop, fileInput.files[0]);
  });
}

function setFile(key, dropEl, f) {
  chosenFiles[key] = f;
  dropEl.querySelector('p').innerHTML = '<strong>Selected:</strong> ' + f.name;
  goBtn.disabled = !chosenFiles.clr;
  statusEl.textContent = '';
  statusEl.className = '';
}

wireDropzone('drop-clr', 'file-clr', 'clr');
wireDropzone('drop-br', 'file-br', 'br');

function tier(score) {
  if (score >= 80) return 'good';
  if (score >= 50) return 'warn';
  return 'bad';
}

goBtn.addEventListener('click', async () => {
  if (!chosenFiles.clr) return;
  goBtn.disabled = true;
  statusEl.className = '';
  statusEl.innerHTML = '<span class="spinner"></span>Analyzing your catalog... this can take a moment for large files.';
  reportEl.innerHTML = '';

  const fd = new FormData();
  fd.append('file', chosenFiles.clr);
  if (chosenFiles.br) fd.append('business_report', chosenFiles.br);

  try {
    const res = await fetch('/analyze', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({detail: 'Unknown error'}));
      throw new Error(err.detail || 'Analysis failed');
    }
    const data = await res.json();
    lastData = data;
    activeFilter = null;
    render();
    statusEl.textContent = '';
  } catch (e) {
    statusEl.className = 'error';
    statusEl.textContent = 'Error: ' + e.message;
  } finally {
    goBtn.disabled = false;
  }
});

const MODULE_META = {
  variation_audit: {
    label: 'Variations', icon: '🧬', accent: '#7c3aed',
    desc: 'Parent/child structure & variation themes',
    categories: ['Orphan Child', 'Single Child Parent', 'Invalid Variation Theme Structure'],
  },
  duplicate_detection: {
    label: 'Duplicate Listings', icon: '🪞', accent: '#db2777',
    desc: 'Near-identical titles & virtual bundle overlap',
    categories: ['Possible Duplicate Listing', 'Virtual Bundle Duplicate'],
  },
  browse_node_audit: {
    label: 'Category Placement', icon: '🗂️', accent: '#2563eb',
    desc: "Products outside the store's main category",
    categories: ['Incorrect Browse Node'],
  },
  index_coverage: {
    label: 'Keyword Indexing', icon: '🔍', accent: '#0891b2',
    desc: 'Keyword presence in titles, bullets & backend terms',
    categories: ['Unindexed Keyword'],
    note: 'ℹ️ This checks whether a keyword\\'s text appears in your listing (indexed or not) — it does not check search rank position. Position requires live search queries per keyword, which is a separate rank-tracking feature.',
  },
  content_completeness: {
    label: 'Content & Attributes', icon: '📝', accent: '#d97706',
    desc: 'Bullet points, descriptions, key attributes',
    categories: ['Missing Attribute', 'Missing Bullet Point'],
  },
  aplus_brand_story: {
    label: 'A+ Content & Brand Story', icon: '🎨', accent: '#0d9488',
    desc: 'Enhanced content presence',
    categories: ['A+ Content Missing', 'Brand Story Missing'],
    note: 'ℹ️ Standard Category Listing Report exports don\\'t contain A+ Content or Brand Story status — that data lives in Seller Central\\'s A+ Content Manager, reachable only via Amazon\\'s separate SP-API A+ Content endpoint (requires Brand Registry + API approval).',
  },
  suppression_risk: {
    label: 'Suppression Risk', icon: '⚠️', accent: '#e11d48',
    desc: 'Title length, banned words, missing images',
    categories: ['Suppression Risk', 'Banned Keyword Usage'],
  },
};

function setFilter(key) {
  activeFilter = (activeFilter === key) ? null : key;
  render();
}

function render() {
  const d = lastData;
  if (!d) return;
  const t = tier(d.overall_health_score);
  let html = `
    <div class="score-banner">
      <div class="score-circle ${t}">${d.overall_health_score}</div>
      <div>
        <div class="label">Overall Catalog Health Score</div>
        <div class="sub">${d.asin_matrix_table.length} of your products have at least one issue flagged below. Click any area below to drill in.</div>
      </div>
    </div>

    <h2 class="section-title">📊 Breakdown by Area</h2>
    <div class="grid">
      ${Object.entries(d.modules).map(([key, v]) => {
        const meta = MODULE_META[key] || { label: key, icon: '•', accent: '#7c3aed', desc: '', categories: [] };
        const ct = tier(v.score);
        const isActive = activeFilter === key;
        return `
          <div class="card ${isActive ? 'active' : ''}" style="--accent:${meta.accent}" onclick="setFilter('${key}')">
            <div class="top">
              <span class="name"><span class="icon-chip">${meta.icon}</span>${meta.label}</span>
              <span class="pill ${ct}">${v.score}</span>
            </div>
            <div class="bar-track"><div class="bar-fill ${ct}" style="width:${v.score}%"></div></div>
            <div class="count">${v.errors_count} issue(s) &middot; ${meta.desc}</div>
            ${meta.note ? `<div class="note">${meta.note}</div>` : ''}
            <div class="click-hint">${isActive ? 'Click again to clear filter' : 'Click to filter the table below'}</div>
          </div>
        `;
      }).join('')}
    </div>
  `;

  if (d.zero_sales_candidates && d.zero_sales_candidates.length > 0 && !activeFilter) {
    html += `
      <h2 class="section-title">🗑️ Recommended for Deletion — Zero Sales (${d.zero_sales_candidates.length})</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Product</th><th>Sessions</th><th>Units Ordered</th><th>Revenue</th><th>Recommendation</th></tr>
          </thead>
          <tbody>
            ${d.zero_sales_candidates.map(row => `
              <tr>
                <td class="title-cell">
                  <div style="font-weight:700;">${escapeHtml((row.title || '').slice(0, 70))}${row.title && row.title.length > 70 ? '…' : ''}</div>
                  <div style="color:var(--muted); font-size:0.85em; margin-top:2px;">ASIN: ${row.asin || '-'} &middot; SKU: ${row.sku || '-'}</div>
                </td>
                <td>${row.sessions}</td>
                <td>${row.units_ordered}</td>
                <td>$${row.revenue.toFixed(2)}</td>
                <td style="color:var(--muted); font-size:0.88em;">${escapeHtml(row.recommendation)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  let rows = d.asin_matrix_table;
  let filterLabel = null;
  if (activeFilter) {
    const meta = MODULE_META[activeFilter];
    filterLabel = meta ? meta.label : activeFilter;
    const catSet = new Set(meta ? meta.categories : []);
    rows = rows.filter(r => (r.categories || []).some(c => catSet.has(c)));
  }

  if (filterLabel) {
    html += `
      <div class="filter-banner">
        <span>Showing <strong>${rows.length}</strong> product(s) with issues in <strong>${filterLabel}</strong></span>
        <button class="clear" onclick="setFilter('${activeFilter}')">Clear filter</button>
      </div>
    `;
  } else {
    html += `<h2 class="section-title">🛠️ Products Needing Attention (${rows.length})</h2>`;
  }

  if (rows.length === 0) {
    html += `
      <div class="empty-state">
        <div class="icon">✅</div>
        <div>No issues found${filterLabel ? ' in ' + filterLabel : ''} — this catalog looks healthy here.</div>
      </div>
    `;
  } else {
    html += `
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Product</th><th>Severity</th><th>Issues Found</th></tr>
          </thead>
          <tbody>
            ${rows.map(row => `
              <tr>
                <td class="title-cell">
                  <div style="font-weight:700;">${escapeHtml((row.title || '').slice(0, 70))}${row.title && row.title.length > 70 ? '…' : ''}</div>
                  <div style="color:var(--muted); font-size:0.85em; margin-top:2px;">ASIN: ${row.asin || '-'} &middot; SKU: ${row.sku || '-'}</div>
                </td>
                <td><span class="sev-badge sev-${row.severity}">${row.severity}</span></td>
                <td>
                  <ul class="issues">${row.issues.map(i => `<li>${escapeHtml(i)}</li>`).join('')}</ul>
                  ${row.ai_consultant_advice ? `<div class="advice">${escapeHtml(row.ai_consultant_advice)}</div>` : ''}
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }
  reportEl.innerHTML = html;
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    business_report: Optional[UploadFile] = File(None),
) -> JSONResponse:
    suffix = Path(file.filename or "upload").suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    br_path: str | None = None
    if business_report is not None and business_report.filename:
        br_suffix = Path(business_report.filename).suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=br_suffix) as br_tmp:
            br_tmp.write(await business_report.read())
            br_path = br_tmp.name

    try:
        result = run_audit(tmp_path, business_report_path=br_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        if br_path:
            Path(br_path).unlink(missing_ok=True)

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8800)
