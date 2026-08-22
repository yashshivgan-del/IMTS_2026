"""HTTP/WebSocket bridge for the block sorting demo.

Serves the Three.js scene, streams state at 20 Hz, and exposes the same
operations as the MCP tools over HTTP so the browser buttons work without
the cloud agent.

    python -m sim.bridge
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from aiohttp import WSMsgType, web

from .config import load_config
from .manager import CellManager

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent.parent / "static"
STREAM_HZ = 20


def build_app(mgr: CellManager | None = None) -> web.Application:
    cfg = load_config()
    if mgr is None:
        mgr = CellManager(cfg)
        mgr.backend.home(cfg)

    app = web.Application()
    app["cfg"] = cfg
    app["mgr"] = mgr
    app["clients"] = set()

    # API routes
    app.router.add_get("/api/scene", _scene)
    app.router.add_get("/api/status", _status)
    app.router.add_get("/api/detect", _detect)
    app.router.add_post("/api/plan", _plan)
    app.router.add_post("/api/execute", _execute)
    app.router.add_get("/api/job/{job_id}", _job)
    app.router.add_post("/api/reset", _reset)
    app.router.add_post("/api/recover", _recover)
    app.router.add_post("/api/estop", _estop)
    app.router.add_get("/ws", _ws)

    # Static files - serve from root for relative paths in HTML
    app.router.add_get("/", _index)
    app.router.add_static("/", STATIC)

    app.on_startup.append(_start_stream)
    app.on_cleanup.append(_stop_stream)
    return app


# -- HTTP ------------------------------------------------------------------- #

async def _index(request: web.Request) -> web.Response:
    return web.FileResponse(STATIC / "index.html")


async def _scene(request: web.Request) -> web.Response:
    return web.json_response(request.app["cfg"].to_scene())


async def _status(request: web.Request) -> web.Response:
    return web.json_response(request.app["mgr"].status())


async def _detect(request: web.Request) -> web.Response:
    return web.json_response(request.app["mgr"].detect_blocks())


async def _plan(request: web.Request) -> web.Response:
    body = await request.json()
    res = request.app["mgr"].plan_sort(sequence=body.get("sequence", []))
    return web.json_response(res)


async def _execute(request: web.Request) -> web.Response:
    body = await request.json()
    mgr: CellManager = request.app["mgr"]
    job_id, violations = mgr.execute_sort(body.get("plan_id", ""))
    if job_id is None:
        return web.json_response(
            {"ok": False, "violations": [v.__dict__ for v in violations]}, status=409
        )
    return web.json_response({"ok": True, "job_id": job_id})


async def _job(request: web.Request) -> web.Response:
    return web.json_response(
        request.app["mgr"].job_status(request.match_info["job_id"])
    )


async def _reset(request: web.Request) -> web.Response:
    return web.json_response(request.app["mgr"].reset_blocks())


async def _recover(request: web.Request) -> web.Response:
    return web.json_response(request.app["mgr"].recover())


async def _estop(request: web.Request) -> web.Response:
    return web.json_response(request.app["mgr"].estop())


# -- WebSocket -------------------------------------------------------------- #

async def _ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    request.app["clients"].add(ws)
    log.info("client connected (%d total)", len(request.app["clients"]))
    try:
        await ws.send_json({"type": "scene", "data": request.app["cfg"].to_scene()})
        async for msg in ws:
            if msg.type is WSMsgType.ERROR:
                break
    finally:
        request.app["clients"].discard(ws)
    return ws


async def _stream(app: web.Application) -> None:
    """Push status at 20 Hz - includes joints, gripper, block positions."""
    mgr: CellManager = app["mgr"]
    period = 1.0 / STREAM_HZ
    while True:
        try:
            payload = json.dumps({"type": "status", "data": mgr.status()})
            for ws in list(app["clients"]):
                if not ws.closed:
                    await ws.send_str(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("stream tick failed")
        await asyncio.sleep(period)


async def _start_stream(app: web.Application) -> None:
    app["stream"] = asyncio.create_task(_stream(app))


async def _stop_stream(app: web.Application) -> None:
    task = app.get("stream")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    web.run_app(build_app(), host="127.0.0.1", port=8801)


if __name__ == "__main__":
    main()
