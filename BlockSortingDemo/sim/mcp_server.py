"""MCP server for block sorting: what the Karini cloud agent sees.

The AI copilot decides WHAT order blocks should be in.
This process decides WHAT IS SAFE to execute.

Tools:
  detect_blocks    - "What blocks are on the table and where?"
  plan_sort        - "I want this sequence. Is it feasible?"
  execute_sort     - "Do it."
  get_sort_status  - "How is it going?"
  reset_blocks     - "Put them back for another demo."

    python -m sim.mcp_server
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP

from .config import load_config
from .manager import CellManager

log = logging.getLogger(__name__)

mcp = FastMCP("karini-block-sorting")

_cfg = load_config()
_mgr = CellManager(_cfg)
_mgr.backend.home(_cfg)
log.info("block sorting cell manager up (backend=%s)", _mgr.backend.name)


@mcp.tool()
def detect_blocks() -> dict:
    """Detect all colored blocks on the table using the overhead camera.

    Returns the position of each block (color, x, y coordinates) and
    the available target slots. Call this to see the current state of
    the workspace before planning a sort.
    """
    return _mgr.detect_blocks()


@mcp.tool()
def plan_sort(sequence: list[str]) -> dict:
    """Plan a block sorting operation.

    Takes a list of block color IDs in the desired order, e.g.:
    ["green", "red", "yellow"]

    This will assign:
      - sequence[0] -> Slot 1
      - sequence[1] -> Slot 2
      - sequence[2] -> Slot 3

    Returns a plan_id if the plan is valid, or a list of violations
    explaining what's wrong. Checks reachability, envelope limits,
    and time budget.

    Available block IDs: green, red, yellow
    """
    return _mgr.plan_sort(sequence)


@mcp.tool()
def execute_sort(plan_id: str) -> dict:
    """Execute a sort plan that plan_sort already approved.

    Accepts only a plan_id from plan_sort: single-use, expiring.
    Returns a job_id to poll with get_sort_status.
    """
    job_id, violations = _mgr.execute_sort(plan_id)
    if job_id is None:
        return {"ok": False, "violations": [v.__dict__ for v in violations]}
    return {"ok": True, "job_id": job_id}


@mcp.tool()
def get_sort_status(job_id: str) -> dict:
    """Poll a running sort operation.

    Returns current progress, which blocks have been placed, and
    what the robot is currently doing.
    """
    return _mgr.job_status(job_id)


@mcp.tool()
def reset_blocks() -> dict:
    """Reset all blocks to their starting positions (removes them from slots).

    The robot will physically pick each block from its current position
    and move it back to its original scattered position on the table.

    Use this to:
    - Clear all slots and move blocks back to starting positions
    - Remove blocks from slots physically
    - Prepare for a fresh sort
    - Start over between demo runs

    Returns a job_id to poll with get_sort_status (same as execute_sort).
    If blocks are already at starting positions, returns immediately.
    """
    return _mgr.reset_blocks()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Block sorting MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport mode: stdio (local/IDE) or sse (remote/Karini)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind SSE server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8803, help="Port for SSE server (default: 8803)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    # Start the bridge server in a background thread so the 3D scene
    # reflects the same state the MCP tools are operating on.
    import asyncio
    import threading
    from aiohttp import web
    from .bridge import build_app

    def run_bridge():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = build_app(mgr=_mgr)
        web.run_app(app, host="127.0.0.1", port=8802, print=None, loop=loop)

    bridge_thread = threading.Thread(target=run_bridge, daemon=True)
    bridge_thread.start()
    log.info("bridge server starting on http://127.0.0.1:8802")

    if args.transport == "sse":
        log.info("Starting MCP server in SSE mode on %s:%d", args.host, args.port)
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        log.info("Starting MCP server in stdio mode (local/IDE)")
        mcp.run()


if __name__ == "__main__":
    main()
