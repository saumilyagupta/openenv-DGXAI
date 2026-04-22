from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
LOG = RESULTS / "logs" / "run.log"
RL_LOG = RESULTS / "logs" / "rl_run.log"
TOTAL_TASKS = 974
RL_TOTAL = 200

app = FastAPI(title="benckmark-codeforge monitor")


def _scan_mode(mode: str) -> dict[str, Any]:
    d = RAW / mode
    if not d.exists():
        return {"n": 0, "passed": 0, "reasons": {}, "latest": []}
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime)
    n = 0
    passed = 0
    reasons: dict[str, int] = {}
    latest_list: list[dict[str, Any]] = []
    for p in files:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n += 1
        if r.get("passed"):
            passed += 1
        reason = str(r.get("reason", "?"))
        reasons[reason] = reasons.get(reason, 0) + 1
    # last 8 processed
    for p in files[-8:][::-1]:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        latest_list.append(
            {
                "task_id": r.get("task_id"),
                "passed": bool(r.get("passed")),
                "reason": r.get("reason"),
                "latency": r.get("latency_seconds"),
                "eval_count": r.get("eval_count"),
                "code": (r.get("extracted_code") or "")[:800],
            }
        )
    return {"n": n, "passed": passed, "reasons": reasons, "latest": latest_list}


def _scan_rl() -> dict[str, Any]:
    d = RAW / "rl"
    if not d.exists():
        return {"n": 0, "first_try_pass": 0, "final_pass": 0, "iter_dist": {},
                "mean_iters_pass": 0.0, "mean_reward": 0.0, "latest": []}
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime)
    n = 0
    first_pass = 0
    final_pass = 0
    iter_dist: dict[str, int] = {}
    iters_to_pass: list[int] = []
    rewards: list[float] = []
    latest: list[dict[str, Any]] = []
    for p in files:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n += 1
        if r.get("final_passed"):
            final_pass += 1
        if r.get("first_pass_iter") == 0:
            first_pass += 1
        k = str(r.get("iters_used", 0))
        iter_dist[k] = iter_dist.get(k, 0) + 1
        if r.get("first_pass_iter") is not None:
            iters_to_pass.append(r["first_pass_iter"] + 1)
        rewards.append(float(r.get("best_reward", 0.0)))
    for p in files[-6:][::-1]:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        its = r.get("iterations", []) or []
        last_it = its[-1] if its else {}
        latest.append({
            "task_id": r.get("task_id"),
            "final_passed": bool(r.get("final_passed")),
            "iters_used": r.get("iters_used"),
            "first_pass_iter": r.get("first_pass_iter"),
            "best_quality": r.get("best_quality"),
            "best_reward": r.get("best_reward"),
            "last_reason": last_it.get("sandbox_reason"),
            "last_code": (last_it.get("extracted_code") or "")[:600],
        })
    return {
        "n": n,
        "first_try_pass": first_pass,
        "final_pass": final_pass,
        "iter_dist": iter_dist,
        "mean_iters_pass": round(sum(iters_to_pass) / len(iters_to_pass), 2) if iters_to_pass else 0.0,
        "mean_reward": round(sum(rewards) / max(len(rewards), 1), 4),
        "latest": latest,
    }


def _tail_log(n: int = 30) -> list[str]:
    if not LOG.exists():
        return []
    try:
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return lines[-n:]


def _tail_file(path: Path, n: int = 25) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except Exception:
        return []


@app.get("/api/status")
def status() -> JSONResponse:
    wo = _scan_mode("without_mcp")
    wi = _scan_mode("with_mcp")
    rl = _scan_rl()
    return JSONResponse(
        {
            "total": TOTAL_TASKS,
            "rl_total": RL_TOTAL,
            "without_mcp": wo,
            "with_mcp": wi,
            "rl": rl,
            "log_tail": _tail_file(LOG, 20),
            "rl_log_tail": _tail_file(RL_LOG, 20),
        }
    )


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>benckmark-codeforge monitor</title>
<style>
body{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#0b0f14;color:#e6edf3;margin:0;padding:16px}
h1{margin:0 0 12px 0;font-size:18px;color:#7ee787}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#11161d;border:1px solid #222b36;border-radius:8px;padding:14px}
.card h2{margin:0 0 8px 0;font-size:15px;color:#79c0ff}
.bar{height:14px;background:#1a2230;border-radius:4px;overflow:hidden;margin:6px 0}
.fill{height:100%;background:linear-gradient(90deg,#2ea043,#7ee787);transition:width .3s}
.stat{display:flex;justify-content:space-between;padding:3px 0;font-size:13px}
.stat span:last-child{color:#d2a8ff}
.reason{display:inline-block;background:#1a2230;border-radius:4px;padding:2px 8px;margin:2px;font-size:12px}
.reason.pass{color:#7ee787}
.reason.fail{color:#ff7b72}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid #222b36}
th{color:#8b949e;font-weight:normal}
.ok{color:#7ee787}
.no{color:#ff7b72}
pre{background:#0d1117;padding:8px;margin:4px 0;border-radius:4px;font-size:11px;max-height:180px;overflow:auto;white-space:pre-wrap;word-break:break-all}
#log{background:#0d1117;padding:10px;border-radius:6px;font-size:11px;max-height:260px;overflow:auto;white-space:pre-wrap}
.foot{color:#6e7681;font-size:11px;margin-top:10px}
details>summary{cursor:pointer;color:#79c0ff;font-size:12px}
</style></head><body>
<h1>benckmark-codeforge — live monitor</h1>
<div class="grid">
  <div class="card" id="wo"></div>
  <div class="card" id="wi"></div>
</div>
<div class="card" style="margin-top:16px" id="rl"></div>
<div class="grid" style="margin-top:16px">
  <div class="card">
    <h2>one-shot log (run.log tail)</h2>
    <div id="log"></div>
  </div>
  <div class="card">
    <h2>RL log (rl_run.log tail)</h2>
    <div id="rllog"></div>
  </div>
</div>
<div class="foot">auto-refresh every 2s · read-only · port 7870</div>
<script>
function renderMode(id,name,data,total){
  const pct = total? (data.n/total*100).toFixed(1):0;
  const pa  = data.n? (data.passed/data.n*100).toFixed(1):'0.0';
  let reasons = '';
  for(const k in data.reasons){
    const cls = k==='pass'?'pass':'fail';
    reasons += `<span class="reason ${cls}">${k}: ${data.reasons[k]}</span>`;
  }
  let rows = '';
  for(const r of data.latest){
    const cls = r.passed?'ok':'no';
    const mark = r.passed?'PASS':'FAIL';
    rows += `<tr><td>${r.task_id}</td><td class="${cls}">${mark}</td><td>${r.reason}</td><td>${r.latency}s</td><td>${r.eval_count}</td></tr>`;
  }
  let code = '';
  if(data.latest.length){
    const last = data.latest[0];
    code = `<details><summary>last generation (task ${last.task_id})</summary><pre>${(last.code||'').replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]))}</pre></details>`;
  }
  document.getElementById(id).innerHTML = `
    <h2>${name}</h2>
    <div class="stat"><span>progress</span><span>${data.n} / ${total} (${pct}%)</span></div>
    <div class="bar"><div class="fill" style="width:${pct}%"></div></div>
    <div class="stat"><span>pass@1</span><span>${pa}% (${data.passed}/${data.n})</span></div>
    <div style="margin-top:8px">${reasons}</div>
    <table><thead><tr><th>id</th><th>result</th><th>reason</th><th>lat</th><th>tok</th></tr></thead><tbody>${rows}</tbody></table>
    ${code}`;
}
function renderRL(data,total){
  const pct = total? (data.n/total*100).toFixed(1):0;
  const firstPct = data.n? (data.first_try_pass/data.n*100).toFixed(1):'0.0';
  const finalPct = data.n? (data.final_pass/data.n*100).toFixed(1):'0.0';
  let dist = '';
  for(const k in data.iter_dist){dist += `<span class="reason">iters=${k}: ${data.iter_dist[k]}</span>`;}
  let rows = '';
  for(const r of data.latest){
    const cls = r.final_passed?'ok':'no';
    const mark = r.final_passed?'PASS':'FAIL';
    const fpi = r.first_pass_iter===null?'-':r.first_pass_iter;
    rows += `<tr><td>${r.task_id}</td><td class="${cls}">${mark}</td><td>${fpi}</td><td>${r.iters_used}</td><td>${(r.best_quality??0).toFixed(2)}</td><td>${(r.best_reward??0).toFixed(2)}</td><td>${r.last_reason||'-'}</td></tr>`;
  }
  let code = '';
  if(data.latest.length){
    const last = data.latest[0];
    code = `<details><summary>last generation (task ${last.task_id})</summary><pre>${(last.last_code||'').replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]))}</pre></details>`;
  }
  document.getElementById('rl').innerHTML = `
    <h2>RL agent (multi-iter MCP + refine)</h2>
    <div class="stat"><span>progress</span><span>${data.n} / ${total} (${pct}%)</span></div>
    <div class="bar"><div class="fill" style="width:${pct}%"></div></div>
    <div class="stat"><span>first-try pass@1</span><span>${firstPct}% (${data.first_try_pass}/${data.n})</span></div>
    <div class="stat"><span>final pass@k</span><span>${finalPct}% (${data.final_pass}/${data.n})</span></div>
    <div class="stat"><span>mean iters to pass</span><span>${data.mean_iters_pass}</span></div>
    <div class="stat"><span>mean best reward</span><span>${data.mean_reward}</span></div>
    <div style="margin-top:8px">${dist}</div>
    <table><thead><tr><th>id</th><th>final</th><th>first_pass_iter</th><th>iters</th><th>q</th><th>reward</th><th>last_reason</th></tr></thead><tbody>${rows}</tbody></table>
    ${code}`;
}
async function tick(){
  try{
    const r = await fetch('/api/status',{cache:'no-store'});
    const d = await r.json();
    renderMode('wo','without_mcp',d.without_mcp,d.total);
    renderMode('wi','with_mcp',d.with_mcp,d.total);
    renderRL(d.rl,d.rl_total);
    document.getElementById('log').textContent = d.log_tail.join('\\n');
    document.getElementById('rllog').textContent = d.rl_log_tail.join('\\n');
  }catch(e){document.getElementById('log').textContent='monitor error: '+e}
}
tick();setInterval(tick,2000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7870, log_level="warning")
