#!/usr/bin/env python3
"""
REST + WebSocket control plane. Everything the dashboard (2.7) and any external client consumes.

    build/venv-services/bin/python3 services/api.py [--port 8080]
    curl localhost:8080/health | jq
    open http://<orin>:8080/docs          # FastAPI's generated API browser

VSS counterpart: `vss-video-analytics-api`. Paths mirror the Phase 2 plan; the agent endpoints
sit on top of the VSS-named tools in `services/agent.py`, so a client written against this
survives a move to a real VSS backend.

## Read-mostly by design

The store is opened **read-only** for queries (`mode=ro`) and the API never writes incident rows
except through the explicit verify endpoint. The event service owns that table. SQLite is in WAL
mode, so these reads never block the writer — which is why the dashboard can poll while 20 camera
streams are producing events.

## What is deliberately NOT here

`nvmultiurisrcbin` (DeepStream's REST-native stream add/remove) is **absent on this install**, so
there is no way to add a camera to a running pipeline. `POST /streams` therefore changes the
configured stream count and reports that a restart is required, rather than pretending to hot-add
a source. Saying so is more useful than a 501.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import LLM, TOOLS, ask, vocabulary  # noqa: E402
from metrics import Sampler  # noqa: E402
from media import browser_playable  # noqa: E402
from search_service import search, summarise, sync  # noqa: E402
from store import connect  # noqa: E402

app = FastAPI(title="Industrial Safety Monitoring API",
              description="Incidents, clips, analytics and the Q&A agent for a 20-camera "
                          "DeepStream pipeline on Jetson AGX Orin.",
              version="2.6")

CFG = yaml.safe_load((ROOT / "configs/services.yml").read_text())
DEMO = yaml.safe_load((ROOT / "configs/demo.yml").read_text())
DB_PATH = ROOT / (CFG.get("store", {}).get("path") or "data/events.db")
CLIPS_DIR = ROOT / (CFG.get("clips", {}).get("dir") or "data/clips")
AGENT_LLM = LLM(os.environ.get("AGENT_ENDPOINT", "http://127.0.0.1:8001/v1"),
                os.environ.get("AGENT_MODEL", "Nemotron-Nano-9B-v2"))

# One tegrastats process for the whole API, started on boot. Sampling every 2s keeps an hour of
# history in the ring buffer; the dashboard polls it rather than each client spawning its own.
SAMPLER = Sampler(interval_ms=int(os.environ.get("METRICS_INTERVAL_MS", "2000")))


@app.on_event("startup")
def _start_sampler():
    SAMPLER.start()


@app.on_event("shutdown")
def _stop_sampler():
    SAMPLER.stop()


def db_ro() -> sqlite3.Connection:
    """A fresh read-only connection per request.

    Per-request rather than shared: sqlite3 connections are not safe to use across threads, and
    FastAPI runs sync endpoints in a threadpool. Opening is microseconds against a local file.
    """
    if not DB_PATH.exists():
        raise HTTPException(503, "no incident database yet — start the event service")
    c = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5.0)
    c.row_factory = sqlite3.Row
    return c


def _proc_running(pattern: str) -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True)
    pids = []
    for line in out.stdout.splitlines():
        if pattern in line and "grep" not in line and "bash -c" not in line:
            try:
                pids.append(int(line.split()[0]))
            except (ValueError, IndexError):
                pass
    return pids


def _proc_args(pattern: str) -> list[str]:
    """The command line of a running process, so status can report what is ACTUALLY running."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if pattern in line and "grep" not in line and "bash -c" not in line:
            return line.split()
    return []


def _arg_value(args: list[str], flag: str) -> str | None:
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    return None


# ---------------------------------------------------------------------------------------------
# health / status
# ---------------------------------------------------------------------------------------------
@app.get("/health", tags=["status"])
def health():
    """Liveness of every moving part, so one call tells an operator what is down."""
    import urllib.error
    import urllib.request

    def probe(url: str) -> bool:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            return False

    n_incidents = None
    if DB_PATH.exists():
        try:
            with db_ro() as c:
                n_incidents = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        except sqlite3.Error:
            pass
    return {
        "status": "ok",
        "pipeline": bool(_proc_running("safety_pipeline.py")),
        "event_service": bool(_proc_running("event_service.py")),
        "clip_service": bool(_proc_running("clip_service.py")),
        "reasoning_service": bool(_proc_running("reasoning_service.py")),
        "vlm": probe("http://127.0.0.1:8000/v1/models"),
        "agent_llm": probe(AGENT_LLM.url.replace("/chat/completions", "/models")),
        "database": DB_PATH.exists(),
        "incidents": n_incidents,
        "clips": len(list(CLIPS_DIR.glob("*.mp4"))) if CLIPS_DIR.exists() else 0,
    }


@app.get("/pipeline/status", tags=["pipeline"])
def pipeline_status():
    """What is ACTUALLY running, not what the config file says.

    These diverge constantly — the pipeline is usually started with explicit `--streams`, while
    `demo.yml` keeps whatever was last persisted. Reporting the config value made the dashboard
    header read "1 streams" while 12 were running, which is exactly the kind of quiet
    misstatement an operator would act on. Live values win; the configured ones are reported
    alongside so a mismatch is visible rather than hidden.
    """
    pids = _proc_running("safety_pipeline.py")
    cfg = DEMO["pipeline"]
    args = _proc_args("safety_pipeline.py") if pids else []

    cfg_streams = int(cfg["streams"])
    cfg_dfi = int(cfg.get("drop_frame_interval", 0) or 0)
    streams = int(_arg_value(args, "--streams") or cfg_streams)
    dfi_arg = _arg_value(args, "--drop-frame-interval")
    dfi = int(dfi_arg) if dfi_arg is not None else cfg_dfi
    fps = (30 // dfi) if dfi > 1 else 30

    return {
        "running": bool(pids), "pids": pids,
        "streams": streams,
        "source_mode": _arg_value(args, "--source") or cfg["source_mode"],
        "drop_frame_interval": dfi,
        "analytics_fps_per_stream": fps,
        "realtime_target_fps": streams * fps,
        "zones": ("--zones" in args) if pids else (
            ROOT / "configs/analytics/analytics.txt").exists(),
        "events": ("--events" in args) if pids else None,
        "rtsp_out": ("--rtsp-out" in args) if pids else False,
        "configured": {"streams": cfg_streams, "drop_frame_interval": cfg_dfi},
        "config_differs": bool(pids) and (streams != cfg_streams or dfi != cfg_dfi),
    }


@app.post("/pipeline/start", tags=["pipeline"])
def pipeline_start(streams: int = Query(None, ge=1, le=20), source: str = Query(None),
                   rtsp_out: bool = Query(False), rgb_capture: bool = Query(None)):
    """Start the pipeline detached. Refuses if one is already running.

    `rtsp_out` adds the encode branch that publishes the tiled wall to mediamtx. It is a start-time
    flag rather than a toggle because the branch is part of the pipeline graph — there is no way to
    attach an encoder to a running DeepStream pipeline, so turning live view on genuinely means
    restarting. Saying so is better than a toggle that silently does nothing.
    """
    if _proc_running("safety_pipeline.py"):
        raise HTTPException(409, "pipeline already running — stop it first")
    args = ["python3", "app/safety_pipeline.py",
            "--source", source or DEMO["pipeline"]["source_mode"],
            "--streams", str(streams or DEMO["pipeline"]["streams"]),
            "--no-display", "--zones", "--events", "--fps", "--stats"]
    if rtsp_out:
        args.append("--rtsp-out")
    # Subject crops. Defaults to on for RTSP and off for file: over RTSP there is no source file
    # to cut a crop from, so this branch is the only way to get one, while in file mode the crop
    # comes out of the .mp4 at the incident PTS and the branch would be pure cost.
    want_crop = rgb_capture
    if want_crop is None:
        want_crop = (source or DEMO["pipeline"]["source_mode"]) == "rtsp"
    if want_crop:
        args.append("--rgb-capture")
    log = ROOT / "logs/pipeline_api.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as fh:
        subprocess.Popen(args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(2)
    return {"started": True, "pids": _proc_running("safety_pipeline.py"), "log": str(log)}


@app.post("/pipeline/stop", tags=["pipeline"])
def pipeline_stop():
    """Stop with SIGKILL, not SIGTERM.

    `pipeline.wait()` blocks inside C++ so SIGINT/SIGTERM are ignored — see project_skill.md.
    The pipeline has no state to flush, so killing it is safe.
    """
    pids = _proc_running("safety_pipeline.py")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"stopped": pids}


@app.patch("/pipeline/config", tags=["pipeline"])
def pipeline_config(streams: int = Query(None, ge=1, le=20),
                    drop_frame_interval: int = Query(None, ge=0, le=30)):
    """Change the persisted pipeline config. Takes effect on next start."""
    global DEMO
    path = ROOT / "configs/demo.yml"
    cfg = yaml.safe_load(path.read_text())
    if streams is not None:
        cfg["pipeline"]["streams"] = streams
    if drop_frame_interval is not None:
        cfg["pipeline"]["drop_frame_interval"] = drop_frame_interval
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    DEMO = cfg
    return {"updated": cfg["pipeline"],
            "restart_required": bool(_proc_running("safety_pipeline.py"))}


# ---------------------------------------------------------------------------------------------
# streams
# ---------------------------------------------------------------------------------------------
@app.get("/streams", tags=["streams"])
def streams_list():
    with db_ro() as c:
        seen = {r[0]: r[1] for r in c.execute(
            "SELECT camera_id, COUNT(*) FROM events GROUP BY camera_id")}
    n = DEMO["pipeline"]["streams"]
    return {"streams": [{"id": f"cam{i:02d}", "index": i, "incidents": seen.get(i, 0)}
                        for i in range(1, n + 1)], "count": n}


@app.post("/streams", tags=["streams"])
def streams_set(count: int = Query(..., ge=1, le=20)):
    """Set the stream count.

    NOT a hot-add. `nvmultiurisrcbin` — DeepStream's REST-native dynamic source element — is
    absent on this install, so a running pipeline cannot gain a camera. This persists the new
    count and tells the caller a restart is needed, which is honest; pretending to hot-add and
    silently doing nothing would be worse.
    """
    r = pipeline_config(streams=count)
    return {**r, "note": "stream count persisted; restart the pipeline to apply "
                         "(nvmultiurisrcbin absent — no hot-add on this install)"}


# ---------------------------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------------------------
@app.get("/events", tags=["events"])
def events_list(camera: int = None, type: str = None, severity: str = None, zone: str = None,
                vlm_verdict: str = None, text: str = None, hours: float = None,
                open_only: bool = False, limit: int = Query(50, ge=1, le=500),
                offset: int = Query(0, ge=0)):
    with db_ro() as c:
        if text:
            # FTS needs a writable index; queries stay read-only, so text search opens its own
            # connection to refresh it rather than making the whole API a writer.
            w = connect(DB_PATH)
            w.row_factory = sqlite3.Row
            sync(w)
            rows = search(w, text=text, camera_id=camera, event_type=type, severity=severity,
                          zone=zone, vlm_verdict=vlm_verdict,
                          since_ts=(time.time() - hours * 3600) if hours else None,
                          open_only=open_only, limit=limit + offset)
            w.close()
        else:
            rows = search(c, camera_id=camera, event_type=type, severity=severity, zone=zone,
                          vlm_verdict=vlm_verdict,
                          since_ts=(time.time() - hours * 3600) if hours else None,
                          open_only=open_only, limit=limit + offset)
    page = rows[offset:offset + limit]
    return {"events": [_public(r) for r in page], "count": len(page), "offset": offset}


def _public(r: dict) -> dict:
    return {
        "id": r["event_id"], "ts": r["ts"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["ts"])),
        "camera_id": r["camera_id"], "sensor": f"cam{r['camera_id']:02d}",
        "type": r["type"], "severity": r["severity"], "zone": r.get("zone"),
        "label": r.get("label"), "state": r.get("state"),
        "vlm_verdict": r.get("vlm_verdict") or "unverified",
        "vlm_reason": r.get("vlm_reason"),
        "duration_s": round(r["duration_s"], 1) if r.get("duration_s") is not None else None,
        "ended": r.get("ended_ts") is not None,
        "hits": r.get("hits"),
        "clip_url": f"/clips/{r['event_id']}" if r.get("clip_uri") else None,
        # Exposed so the UI can say WHY there is no clip. "expired" (aged out under retention,
        # incident kept) and "skipped" (no source timestamp, so it could never be located) are
        # different facts, and an empty video box states neither.
        "clip_state": r.get("clip_state"),
        # How many times a still-open incident has been re-raised, and how long it has been
        # running. `open_minutes` is computed here rather than in the browser so every client
        # agrees on it — a client clock that is minutes out would otherwise report a different
        # duration for the same incident.
        "reminder_count": r.get("reminder_count") or 0,
        "open_minutes": (round((time.time() - r["ts"]) / 60.0, 1)
                         if r.get("ended_ts") is None else None),
    }


@app.get("/events/{event_id}", tags=["events"])
def event_detail(event_id: str):
    with db_ro() as c:
        r = c.execute("SELECT * FROM events WHERE event_id = ? OR event_id LIKE ?",
                      (event_id, f"{event_id}%")).fetchone()
    if not r:
        raise HTTPException(404, f"no incident {event_id}")
    return _public(dict(r))


@app.post("/events/{event_id}/verify", tags=["events"])
def event_verify(event_id: str, verdict: str = Query(..., pattern="^(confirmed|rejected)$"),
                 note: str = None):
    """Operator override of a VLM verdict.

    The only write this API makes to incident rows. Marked `state='verified'` and the note
    prefixed so a human decision is never mistaken for the model's — an operator correcting the
    VLM is exactly the signal worth keeping distinguishable.
    """
    w = connect(DB_PATH)
    cur = w.execute(
        "UPDATE events SET vlm_verdict = ?, state = 'verified', "
        "  vlm_reason = COALESCE(?, vlm_reason) WHERE event_id = ? OR event_id LIKE ?",
        (verdict, f"[operator] {note}" if note else None, event_id, f"{event_id}%"))
    w.commit()
    n = cur.rowcount
    w.close()
    if not n:
        raise HTTPException(404, f"no incident {event_id}")
    return {"id": event_id, "vlm_verdict": verdict, "by": "operator"}


# ---------------------------------------------------------------------------------------------
# analytics
# ---------------------------------------------------------------------------------------------
@app.get("/analytics/summary", tags=["analytics"])
def analytics_summary(hours: float = None):
    with db_ro() as c:
        return summarise(c, hours)


@app.get("/analytics/timeseries", tags=["analytics"])
def analytics_timeseries(hours: float = 24, buckets: int = Query(24, ge=1, le=200)):
    """Incident counts over time, bucketed — what the dashboard plots."""
    now = time.time()
    start = now - hours * 3600
    width = (now - start) / buckets
    with db_ro() as c:
        rows = c.execute(
            "SELECT ts, type, severity FROM events WHERE ts >= ? ORDER BY ts", (start,)
        ).fetchall()
    series = [{"t": start + i * width, "total": 0, "by_type": {}, "by_severity": {}}
              for i in range(buckets)]
    for r in rows:
        i = min(buckets - 1, int((r["ts"] - start) / width)) if width > 0 else 0
        b = series[i]
        b["total"] += 1
        b["by_type"][r["type"]] = b["by_type"].get(r["type"], 0) + 1
        b["by_severity"][r["severity"]] = b["by_severity"].get(r["severity"], 0) + 1
    return {"buckets": series, "bucket_seconds": round(width, 1), "hours": hours}


@app.get("/analytics/calendar", tags=["analytics"])
def analytics_calendar(days: int = Query(35, ge=1, le=366)):
    """Per-DAY incident counts for the calendar, in the server's local timezone.

    Local, not UTC: an operator reading a calendar means their own working day, and a shift that
    starts at 08:00 local should not appear split across two cells. `ts` is wall-clock epoch, so
    the day boundary is computed with localtime rather than by slicing an ISO string.

    Counts are of INCIDENTS, never of people. A "how many people had PPE violations" figure was
    tried here as SUM(hits) — the tracked sightings folded into each incident — and came out at
    **23,122 for 20 incidents**, because `hits` counts every re-observation of the same situation.
    Displaying that next to the word "people" would have been a fabrication with a plausible
    number attached, which is worse than not answering. The store has no headcount and this
    endpoint does not invent one; the labels in the UI say "incidents" for the same reason.
    """
    since = time.time() - days * 86400
    with db_ro() as c:
        rows = c.execute(
            "SELECT date(ts, 'unixepoch', 'localtime') AS d, "
            "       COUNT(*) AS n, "
            "       SUM(CASE WHEN type='ppe_violation' THEN 1 ELSE 0 END) AS ppe, "
            "       SUM(CASE WHEN type='fire_alert' THEN 1 ELSE 0 END) AS fire, "
            "       SUM(CASE WHEN type='overcrowding' THEN 1 ELSE 0 END) AS crowd, "
            "       SUM(CASE WHEN type='hazard_alert' THEN 1 ELSE 0 END) AS hazard, "
            "       SUM(CASE WHEN severity='critical' THEN 1 ELSE 0 END) AS critical, "
            "       SUM(CASE WHEN vlm_verdict='confirmed' THEN 1 ELSE 0 END) AS confirmed "
            "  FROM events WHERE ts >= ? GROUP BY d ORDER BY d", (since,)).fetchall()
    return {"days": [dict(r) for r in rows],
            "range_days": days,
            "timezone": time.strftime("%Z")}


@app.get("/analytics/day/{day}", tags=["analytics"])
def analytics_day(day: str, limit: int = Query(100, ge=1, le=500)):
    """Everything that happened on one calendar day — what a calendar cell opens into."""
    with db_ro() as c:
        rows = c.execute(
            "SELECT * FROM events WHERE date(ts, 'unixepoch', 'localtime') = ? "
            " ORDER BY ts DESC LIMIT ?", (day, limit)).fetchall()
        agg = c.execute(
            "SELECT COUNT(*) n, "
            "  SUM(CASE WHEN type='ppe_violation' THEN 1 ELSE 0 END) ppe, "
            "  SUM(CASE WHEN vlm_verdict='confirmed' THEN 1 ELSE 0 END) confirmed, "
            "  COUNT(DISTINCT camera_id) cameras "
            " FROM events WHERE date(ts, 'unixepoch', 'localtime') = ?", (day,)).fetchone()
        by_type = {r[0]: r[1] for r in c.execute(
            "SELECT type, COUNT(*) FROM events "
            " WHERE date(ts, 'unixepoch', 'localtime') = ? GROUP BY type", (day,))}
        by_zone = {(r[0] or "no zone"): r[1] for r in c.execute(
            "SELECT zone, COUNT(*) FROM events "
            " WHERE date(ts, 'unixepoch', 'localtime') = ? GROUP BY zone ORDER BY 2 DESC", (day,))}
    return {"day": day, "summary": dict(agg), "by_type": by_type, "by_zone": by_zone,
            "events": [_public(dict(r)) for r in rows]}


@app.get("/analytics/zones", tags=["analytics"])
def analytics_zones():
    """Which zones actually generate incidents — the pie chart's data.

    Split by type as well as total, because "the spill zone has the most incidents" and "the spill
    zone has the most FIRE incidents" are different operational facts and a single total hides the
    second. `no zone` is kept rather than dropped: an incident outside every configured polygon is
    a real incident, and silently omitting it would make the shares add up to a lie.
    """
    with db_ro() as c:
        rows = c.execute(
            "SELECT COALESCE(zone, 'no zone') z, COUNT(*) n, "
            "  SUM(CASE WHEN type='ppe_violation' THEN 1 ELSE 0 END) ppe, "
            "  SUM(CASE WHEN type='overcrowding' THEN 1 ELSE 0 END) crowd, "
            "  SUM(CASE WHEN type IN ('fire_alert','hazard_alert') THEN 1 ELSE 0 END) danger, "
            "  SUM(CASE WHEN vlm_verdict='confirmed' THEN 1 ELSE 0 END) confirmed, "
            "  SUM(CASE WHEN severity='critical' THEN 1 ELSE 0 END) critical "
            " FROM events GROUP BY z ORDER BY n DESC").fetchall()
    total = sum(r["n"] for r in rows) or 1
    return {"zones": [{**dict(r), "share": round(100 * r["n"] / total, 1)} for r in rows],
            "total": total}


@app.get("/analytics/cameras", tags=["analytics"])
def analytics_cameras():
    with db_ro() as c:
        rows = c.execute(
            "SELECT camera_id, COUNT(*) n, "
            "  SUM(CASE WHEN severity='high' THEN 1 ELSE 0 END) high, "
            "  SUM(CASE WHEN vlm_verdict='confirmed' THEN 1 ELSE 0 END) confirmed, "
            "  SUM(CASE WHEN vlm_verdict='rejected' THEN 1 ELSE 0 END) rejected, "
            "  SUM(CASE WHEN ended_ts IS NULL THEN 1 ELSE 0 END) open "
            "FROM events GROUP BY camera_id ORDER BY camera_id").fetchall()
    return {"cameras": [{"sensor": f"cam{r['camera_id']:02d}", **{k: r[k] for k in
                        ("n", "high", "confirmed", "rejected", "open")}} for r in rows]}


# ---------------------------------------------------------------------------------------------
# clips — with HTTP range support
# ---------------------------------------------------------------------------------------------
@app.get("/clips/{event_id}", tags=["clips"])
def clip(event_id: str, request: Request):
    """Serve an evidence clip, honouring HTTP Range.

    Range support is not optional: a browser `<video>` element issues a range request to seek,
    and a server that answers 200-with-everything makes the scrub bar dead and forces a full
    re-download on every seek. Returning 206 with `Accept-Ranges` is what makes the dashboard's
    player usable.
    """
    with db_ro() as c:
        r = c.execute("SELECT event_id, clip_uri, clip_state FROM events "
                      " WHERE event_id = ? OR event_id LIKE ?",
                      (event_id, f"{event_id}%")).fetchone()
    if not r:
        raise HTTPException(404, f"no incident {event_id}")
    if not r["clip_uri"]:
        raise HTTPException(404, f"incident has no clip (clip_state={r['clip_state']})")
    path = ROOT / r["clip_uri"]
    if not path.exists():
        raise HTTPException(410, "clip file has been deleted by retention")
    path = browser_playable(path)

    size = path.stat().st_size
    rng = request.headers.get("range")
    if not rng:
        return FileResponse(path, media_type="video/mp4",
                            headers={"Accept-Ranges": "bytes"})
    try:
        units, _, rest = rng.partition("=")
        s, _, e = rest.partition("-")
        start = int(s) if s else 0
        end = int(e) if e else size - 1
    except ValueError:
        raise HTTPException(416, "malformed Range header")
    if units.strip() != "bytes" or start >= size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    end = min(end, size - 1)

    with open(path, "rb") as fh:
        fh.seek(start)
        data = fh.read(end - start + 1)
    return Response(data, status_code=206, media_type="video/mp4", headers={
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(len(data)),
    })


@app.get("/clips", tags=["clips"])
def clips_list(limit: int = Query(20, ge=1, le=200)):
    db = connect(DB_PATH)
    try:
        out = TOOLS["get_clips"](db, max_count=limit)
    finally:
        db.close()
    for c in out["clips"]:
        c["url"] = f"/clips/{c['id']}"
    return out


# ---------------------------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------------------------
@app.post("/agent/chat", tags=["agent"])
async def agent_chat(body: dict):
    """Ask a question. `history` carries prior turns so follow-ups work."""
    q = (body or {}).get("question", "").strip()
    if not q:
        raise HTTPException(400, "question is required")
    history = (body or {}).get("history") or []

    def _run():
        # The connection is opened INSIDE the worker thread. sqlite3 connections are bound to the
        # thread that created them, so opening on the event loop and passing it here raises
        # "SQLite objects created in a thread can only be used in that same thread" — which is
        # exactly what happened.
        db = connect(DB_PATH)
        try:
            return ask(db, q, AGENT_LLM, history)
        finally:
            db.close()

    # ask() does blocking HTTP to the LLM; keep the event loop free so /events and the WebSocket
    # stay responsive while a question is being answered.
    res = await asyncio.to_thread(_run)
    rows = res["result"].get("incidents") or res["result"].get("clips") or []
    if not isinstance(rows, list):
        rows = []
    # The agent works in filesystem paths (`data/clips/<id>.mp4`) because its tools are also a CLI
    # and a future MCP surface. A browser cannot open those. Attaching the HTTP URL here — at the
    # boundary that actually speaks HTTP — is what lets the dashboard play a clip the agent
    # returned, rather than describing one the operator cannot see.
    for r in rows:
        if isinstance(r, dict) and r.get("id") and (r.get("clip") or r.get("has_clip")):
            r["clip_url"] = f"/clips/{r['id']}"
    # `counts` are SQL aggregates over the whole matching set, not something the model produced.
    # They are returned alongside the prose so the dashboard can show the exact per-camera numbers
    # as a table: the model narrates, the database counts. Same split as Phase 2.4, where the VLM
    # answers perception questions and the verdict is computed in code — a model asked to
    # enumerate nine cameras from a payload reliably narrates three of them.
    counts = res["result"].get("counts") if isinstance(res["result"], dict) else None
    if not counts and res["tool"] == "get_summary" and isinstance(res["result"], dict):
        counts = res["result"]
    return {"question": q, "answer": res["answer"], "cited_ids": res["cited_ids"],
            "tool": res["tool"], "plan": res["plan"],
            "incidents": rows[:20],
            "counts": counts,
            "dropped_citations": res.get("dropped_citations", []),
            "relaxed_filters": res.get("relaxed_filters", [])}


@app.get("/agent/tools", tags=["agent"])
def agent_tools():
    db = connect(DB_PATH)
    try:
        v = vocabulary(db)
    finally:
        db.close()
    return {"tools": [{"name": n, "description": (f.__doc__ or "").strip().splitlines()[0]}
                      for n, f in TOOLS.items()],
            "vocabulary": v}


# ---------------------------------------------------------------------------------------------
# live event stream
# ---------------------------------------------------------------------------------------------
@app.websocket("/events/stream")
async def events_stream(ws: WebSocket):
    """Push new incidents as they land.

    Polls the store rather than subscribing to Redis: the store is where transitions have already
    been folded into incidents, so a client sees the same objects the REST API returns instead of
    raw per-track events it would have to reassemble.
    """
    await ws.accept()
    last = time.time()
    try:
        while True:
            await asyncio.sleep(2)
            try:
                with db_ro() as c:
                    rows = c.execute(
                        "SELECT * FROM events WHERE ts > ? ORDER BY ts LIMIT 50", (last,)
                    ).fetchall()
            except (sqlite3.Error, HTTPException):
                continue
            if rows:
                last = max(r["ts"] for r in rows)
                await ws.send_text(json.dumps({"events": [_public(dict(r)) for r in rows]}))
    except (WebSocketDisconnect, RuntimeError):
        return


# ---------------------------------------------------------------------------------------------
# rtsp output toggle
# ---------------------------------------------------------------------------------------------
@app.post("/rtsp/{state}", tags=["live"])
def rtsp(state: str, request: Request):
    """Turn the tiled RTSP output on or off.

    Off by default because the encode branch is a real cost. It runs on dedicated NVENC silicon
    rather than the GPU, so it does not compete with inference — but it is still a choice an
    operator should make knowingly, which is why it is an endpoint rather than always-on.
    """
    if state not in ("on", "off"):
        raise HTTPException(400, "state must be 'on' or 'off'")
    r = subprocess.run(["bash", "scripts/serve_rtsp.sh", "start" if state == "on" else "stop"],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    return {"rtsp": state, "ok": r.returncode == 0,
            "output": (r.stdout + r.stderr).strip()[-400:],
            "url": f"rtsp://{request.url.hostname or os.uname().nodename}:8554"
                   f"{DEMO['sinks']['rtsp_out']['mount']}" if state == "on" else None,
            "note": "pipeline must be started with --rtsp-out to publish"}


@app.get("/live/status", tags=["live"])
def live_status(request: Request):
    """Why the live view is or is not showing a picture.

    Three different failures all look identical in the browser — a black box:

      1. mediamtx is not running        -> WHEP connection refused
      2. mediamtx is running, nobody publishes -> WHEP 404, path not ready
      3. both fine, the viewer's WebRTC failed -> everything green here

    They have completely different fixes, and (2) is the easy one to get wrong because
    `POST /rtsp/on` starts the *server* while the *publisher* is a pipeline start-time flag.
    Reporting a single "unavailable" would leave the operator guessing between them, so this
    reports each layer separately and names the next action.
    """
    server, paths, api_err = False, [], None
    try:
        with urllib.request.urlopen("http://127.0.0.1:9997/v3/paths/list", timeout=3) as fh:
            paths = json.load(fh).get("items", [])
            server = True
    except Exception as e:  # noqa: BLE001 — any failure here means "cannot see the server"
        api_err = str(e)[:120]
        # The control API is only enabled by the current serve_rtsp.sh; an older mediamtx started
        # before that change is still running and still serving video. Fall back to the process
        # check so this does not report "down" for a server that is plainly up.
        server = bool(_proc_running("mediamtx"))

    mount = str(DEMO["sinks"]["rtsp_out"]["mount"]).lstrip("/") or "safety"
    path = next((p for p in paths if p["name"] == mount), None)
    publishing = bool(path and path.get("ready"))
    pipe = _proc_running("safety_pipeline.py")
    pipe_rtsp = "--rtsp-out" in (_proc_args("safety_pipeline.py") if pipe else [])

    if not server:
        reason, action = "mediamtx is not running", "POST /rtsp/on"
    elif not pipe:
        reason, action = "no pipeline is running", "POST /pipeline/start?rtsp_out=true"
    elif not pipe_rtsp:
        reason = "the running pipeline was started without --rtsp-out, so nothing is published"
        action = "restart it: POST /pipeline/stop then POST /pipeline/start?rtsp_out=true"
    elif not publishing:
        reason, action = "pipeline has --rtsp-out but the path is not ready yet", "wait a few seconds"
    else:
        reason, action = None, None

    # The host the CALLER used, not this machine's own name. `os.uname().nodename` is whatever the
    # platform decided: on AWS it is the internal `ip-172-31-x-y`, which resolves nowhere outside
    # the VPC, and on any cloud instance it is a name the viewer's DNS has never heard of. The
    # request already carries the address that demonstrably reaches this API, so every URL below
    # is built from that and follows the viewer whether they came by public IP, private IP,
    # tunnel or proxy. scripts/demo_up.sh prints addresses instead of `hostname` for the same
    # reason — a hostname only works if the viewer's DNS agrees.
    host = request.url.hostname or os.uname().nodename
    return {
        "ready": publishing, "server": server, "publishing": publishing,
        "pipeline_running": bool(pipe), "pipeline_rtsp_out": pipe_rtsp,
        "path": mount, "readers": len(path.get("readers", [])) if path else 0,
        "reason": reason, "action": action,
        "api_error": api_err,
        "urls": {"rtsp": f"rtsp://{host}:8554/{mount}",
                 "whep": f"http://{host}:8889/{mount}/whep",
                 "hls": f"http://{host}:8888/{mount}/index.m3u8"},
    }


@app.post("/alerts/test", tags=["alerts"])
def alerts_test(type: str = Query("fire_alert"), camera: int = Query(1, ge=1, le=20),
                zone: str = Query(None)):
    """Inject a TEST alert through the real event path, for exercising the alarm UI.

    This publishes to Redis exactly as the pipeline does, so it travels the whole chain — event
    service, incident store, WebSocket, dashboard — rather than being poked straight into the
    database. Testing the alarm by faking the last step would prove nothing about the first ones.

    The label is prefixed "TEST" and the reason says so, because an operator (or a later reader of
    the database) must never mistake an injected alert for a detection. It exists because this
    synthetic footage contains no fire, so the fire path cannot otherwise be demonstrated.

    ## Two bugs this endpoint had, both of which made it lie

    It used to emit a single opening event with the default `track_id = -1`. Every injection on a
    camera therefore shared the track key `(camera, type, -1)`, and the store's first branch —
    "this track is already counted against an incident" — absorbed it as a severity escalation.
    The result: the FIRST test alert on a camera appeared, and every one after it vanished, while
    the endpoint cheerfully returned `injected: true`. Worse, the injected incident never closed
    (no track ever cleared), so the camera stayed poisoned indefinitely.

    Fixed by giving each injection a unique `track_id` and emitting the matching CLOSE, so the
    test incident is a complete open/close pair like a real one and leaves nothing open behind it.

    The second bug is the one that let the first hide: `injected: true` described a successful
    *publish*, not a visible alert. It now waits for the row and reports `incident_created`, so a
    swallowed alert is reported as swallowed. A test tool that cannot fail is not a test tool.
    """
    sys.path.insert(0, str(ROOT / "app"))
    from events import Event, EventEmitter

    em = EventEmitter(host=(CFG["events"]["redis"]["host"]),
                      port=int(CFG["events"]["redis"]["port"]),
                      stream=CFG["events"]["redis"]["stream"])
    # Unique per injection, and negative so it can never collide with a real tracker id.
    track = -abs(int(time.time() * 1000) % 2_000_000_000) - 2
    common = dict(camera_id=camera, type=type, track_id=track,
                  severity="critical" if type == "fire_alert" else "high",
                  label=f"TEST {'FIRE' if type == 'fire_alert' else type.upper()}",
                  zone=zone,
                  vlm_reason="Injected via POST /alerts/test — not a real detection.")
    ev = Event(**common)
    em.emit(ev)
    # The close carries the same track so the incident's reference count returns to zero. Without
    # it the incident stays open forever and swallows every later injection on this camera.
    em.emit(Event(**common, ended=True, duration_s=1.0))
    em.close()

    # Wait for the incident to actually land. The whole point of routing through Redis is that the
    # chain is real, which means it can fail — so this reports what the database ended up with
    # rather than what was sent.
    created, waited = False, 0.0
    while waited < 5.0:
        time.sleep(0.25)
        waited += 0.25
        db = connect(DB_PATH)
        try:
            got = db.execute("SELECT 1 FROM events WHERE event_id = ?", (ev.event_id,)).fetchone()
        finally:
            db.close()
        if got:
            created = True
            break

    return {"injected": True, "incident_created": created,
            "event_id": ev.event_id, "type": type, "camera": f"cam{camera:02d}",
            "latency_s": round(waited, 2) if created else None,
            "note": ("published to Redis on the normal path; labelled TEST" if created else
                     "published, but no NEW incident appeared — it merged into an incident "
                     f"already open on cam{camera:02d}. The dashboard alarms on new incidents, "
                     "so it will not fire. Close or clear that incident first.")}


@app.get("/notify/status", tags=["alerts"])
def notify_status():
    """Is the Telegram channel configured, running, and delivering?

    Reports the three things that are separately capable of being wrong — enabled in config,
    credentials present, service running — plus what has actually been sent. "No messages
    arriving" has at least four causes and they need different fixes.
    """
    cfg = (CFG.get("notify") or {})
    tg = cfg.get("telegram") or {}
    token_env = tg.get("token_env", "TELEGRAM_BOT_TOKEN")
    chat_env = tg.get("chat_env", "TELEGRAM_CHAT_ID")
    running = bool(_proc_running("notify_service.py"))

    with db_ro() as c:
        try:
            rows = {r[0]: r[1] for r in c.execute(
                "SELECT notify_state, COUNT(*) FROM events GROUP BY notify_state")}
            last = c.execute(
                "SELECT event_id, camera_id, type, notify_ts, notify_error FROM events "
                " WHERE notify_ts IS NOT NULL ORDER BY notify_ts DESC LIMIT 1").fetchone()
        except sqlite3.OperationalError:
            rows, last = {}, None      # database predates the notify columns

    enabled = bool(tg.get("enabled"))
    has_creds = bool(os.environ.get(token_env) and os.environ.get(chat_env))
    if not enabled:
        reason = "disabled in configs/services.yml (notify.telegram.enabled)"
    elif not running:
        reason = ("enabled but the notify service is not running — it does not start without "
                  f"{token_env} and {chat_env} in .env at the repo root")
    else:
        reason = None

    return {
        "enabled": enabled, "running": running,
        # The API process may not have the credentials in its own environment even when the
        # notifier does, so this is reported as a hint rather than as the verdict.
        "credentials_visible_to_api": has_creds,
        "by_state": rows, "reason": reason,
        "last": ({"id": last[0][:8], "sensor": f"cam{last[1]:02d}", "type": last[2],
                  "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last[3])),
                  "error": last[4]} if last else None),
        "policy": {k: cfg.get(k) for k in
                   ("always_types", "confirmed_types", "min_severity",
                    "cooldown_per_camera_s", "max_per_minute")},
    }


@app.get("/metrics/system", tags=["metrics"])
def metrics_system(minutes: float = Query(5.0, ge=0.1, le=60),
                   max_points: int = Query(240, ge=10, le=1000)):
    """Current system utilisation plus recent history, for the dashboard's graphs.

    Everything is a percentage so one y-axis serves the whole chart — mixing a percentage with a
    megabyte count would need a second scale, and a dual-axis chart is the mistake to avoid. RAM
    is therefore returned as a percentage AND as raw MB for the tooltip.
    """
    cur = SAMPLER.current()
    return {
        "current": cur,
        "history": SAMPLER.history(minutes, max_points),
        "interval_ms": SAMPLER.interval_ms,
        "samples_buffered": len(SAMPLER.buf),
        # Which telemetry produced these: "tegrastats" on Jetson, "nvidia-smi" on a discrete GPU.
        # The dashboard names it when reporting an error, so a failure reads as the tool that
        # actually failed rather than as whichever one this file was written against.
        "backend": SAMPLER.backend,
        # What this is running on. The dashboard shows it because the same image and the same UI
        # now serve a Jetson and an x86 box, and which one you are looking at is not otherwise
        # visible from the numbers.
        "platform": SAMPLER.platform,
        "device": SAMPLER.device,
        "arch": SAMPLER.arch,
        "error": SAMPLER.error,
    }


@app.get("/system", response_class=HTMLResponse, include_in_schema=False)
def system_page():
    """Developer reference: models, engines, measured throughput and latency, wiring.

    Static by design. It is a document to read next to the running dashboard, not a second
    monitor — mixing live values into it would leave a reader unsure which numbers are history
    and which are now, and every figure on it is a measurement with a script behind it.
    """
    page = ROOT / "dashboard/system.html"
    if not page.exists():
        raise HTTPException(404, "system page not found")
    return HTMLResponse(page.read_text())


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    dash = ROOT / "dashboard/index.html"
    if dash.exists():
        return HTMLResponse(dash.read_text())
    return HTMLResponse(
        "<h2>Industrial Safety Monitoring API</h2>"
        "<p>Dashboard lands in Phase 2.7. Meanwhile: "
        "<a href='/docs'>/docs</a> for the API browser, "
        "<a href='/health'>/health</a> for status.</p>")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    if (ROOT / "dashboard").exists():
        app.mount("/static", StaticFiles(directory=ROOT / "dashboard"), name="static")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
