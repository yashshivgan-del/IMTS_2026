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

# Load .env file if present (for OPENAI_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastmcp import FastMCP

from .config import load_config
from .manager import CellManager

log = logging.getLogger(__name__)

mcp = FastMCP("karini-block-sorting")

_cfg = load_config()
_mgr = CellManager(_cfg)
try:
    _mgr.backend.home(_cfg)
except Exception:
    # For mqtt_proxy specifically: home() blocks waiting for a laptop-side
    # agent to reply on robo/result. If no agent is subscribed yet (e.g.
    # testing the MQTT publish path standalone, per
    # docs/AGENT_MQTT_FLOW.md), this times out -- don't let that crash the
    # whole server on startup; log it and continue. Tool calls that need a
    # live agent will still fail/timeout individually until one connects,
    # but the server itself stays up so the agent can connect and other
    # tools (or a later home retry) can be exercised.
    log.exception(
        "startup home() failed (backend=%s) -- continuing without a "
        "confirmed home position. If using mqtt_proxy, this is expected "
        "until a laptop agent is subscribed and replying.",
        _mgr.backend.name,
    )
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
def plan_arrangement(target_state: dict[str, str]) -> dict:
    """Plan moves to reach a desired board state with minimal changes.

    This is the PRIMARY planning tool. It accepts a target state describing
    which block should be in which slot. Only specify slots that need to change —
    unmentioned slots remain as-is.

    Parameters:
        target_state: dict mapping slot_id to block_id
            Keys: "slot_1", "slot_2", "slot_3"
            Values: "green", "red", "yellow"

    Examples:
        {"slot_2": "green"}
            → Put green in Slot 2, leave everything else unchanged.

        {"slot_1": "red", "slot_3": "yellow"}
            → Put red in Slot 1 and yellow in Slot 3, leave Slot 2 unchanged.

        {"slot_1": "red", "slot_2": "green", "slot_3": "yellow"}
            → Full arrangement: Red, Green, Yellow left to right.

    The planner automatically:
        - Detects conflicts (slot already occupied by wrong block)
        - Relocates conflicting blocks to safe positions first
        - Skips blocks already in their correct slot
        - Computes the minimal set of moves

    Returns a plan_id to pass to execute_sort, or violations if invalid.
    After execute_sort, poll get_sort_status until done, then call
    detect_blocks to verify the final state.
    """
    return _mgr.plan_arrangement(target_state)


@mcp.tool()
def execute_sort(plan_id: str) -> dict:
    """Execute a plan that was approved by plan_sort or plan_arrangement.

    Accepts a plan_id (single-use, expiring after 120 seconds).
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


@mcp.tool()
def reset_arm() -> dict:
    """Send the arm to its initialization position.

    Call this if the arm is in an unexpected position, stuck, or after an error.
    The arm will run its firmware initialization routine and return to the safe home pose.
    Returns ok when complete.
    """
    try:
        _mgr.backend.home(_cfg, do_init=True)
        return {"ok": True, "message": "Arm returned to init position"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@mcp.tool()
def validate_placement() -> dict:
    """Capture the mat with the camera and use a vision model to describe
    the current block arrangement on the grid.

    Call this after all placements are complete to verify the final state.
    Returns a text description of where each block is on the grid.
    """
    import os, base64
    try:
        import openai
    except ImportError:
        return {"error": "openai package not installed on EC2"}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set on EC2"}

    if not hasattr(_mgr.backend, "capture_image"):
        return {"error": "capture_image not supported by current backend"}

    image_b64 = _mgr.backend.capture_image()
    if not image_b64:
        return {"error": "No image returned from Pi"}

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "This is a top-down view of a robotic arm sorting mat. "
                        "There are TWO grids visible:\n"
                        "1. A LARGER PINK grid (6 cells: 3 rows x 2 columns) — this is the PLACEMENT grid. "
                        "The grid is oriented so that the arm base is at the BOTTOM of the image. "
                        "Row A is the INNERMOST row (farthest from the arm, toward the top of the image). "
                        "Row C is the OUTERMOST row (closest to the arm, toward the bottom). "
                        "Column 1 is on the RIGHT side, column 2 is on the LEFT side. "
                        "So A1=top-right, A2=top-left, B1=middle-right, B2=middle-left, C1=bottom-right, C2=bottom-left.\n"
                        "2. A SMALLER grid (4 cells, no pink border) visible elsewhere — this is just the pick-up zone, IGNORE IT.\n\n"
                        "Please analyze the image and tell me:\n"
                        "1. For each cell in the LARGE PINK grid (A1, A2, B1, B2, C1, C2): is there a block? If yes, what color (green, yellow, or orange)?\n"
                        "2. Is the robotic arm inside the large pink grid? It should NOT be inside the grid after placement.\n"
                        "3. Are all blocks fully inside their grid cells or are any hanging over the edge?\n"
                        "4. Overall: did the placements look successful?\n\n"
                        "Be specific and concise. Format your answer as:\n"
                        "- A1: [color or empty]\n"
                        "- A2: [color or empty]\n"
                        "- B1: [color or empty]\n"
                        "- B2: [color or empty]\n"
                        "- C1: [color or empty]\n"
                        "- C2: [color or empty]\n"
                        "- Arm in grid: [yes/no]\n"
                        "- Assessment: [success/issues found]"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
        }],
        max_tokens=300,
    )
    description = response.choices[0].message.content
    log.info("validate_placement VLM response: %s", description)
    return {"description": description}


@mcp.tool()
def detect_kit_plan(placements: list[dict]) -> dict:
    """Detect all blocks and return their arm coordinates for a kit plan.

    This is phase 1 of the two-phase kit plan flow. Send the desired
    placements (color, cell, seq) and get back the detected pick/place
    coordinates for each. Review them, then call execute_plan to run.

    Parameters:
        placements: list of dicts, each with:
            - color: block color id (e.g. "green", "yellow", "orange")
            - cell:  target grid cell (e.g. "A1", "B2", "C1")
            - seq:   execution order (1 = first, 2 = second, etc.)

    Example:
        [
          {"color": "green",  "cell": "A2", "seq": 1},
          {"color": "yellow", "cell": "B2", "seq": 2},
          {"color": "orange", "cell": "C2", "seq": 3},
          {"color": "orange", "cell": "B1", "seq": 4}
        ]

    Returns the same list with pick_x, pick_y, place_x, place_y added.
    Pass the result directly to execute_plan.
    """
    if not hasattr(_mgr.backend, "detect_kit_plan"):
        return {"error": "detect_kit_plan not supported by current backend"}
    resolved = _mgr.backend.detect_kit_plan(placements)
    return {"placements": resolved}


@mcp.tool()
def execute_single_placement(placement: dict) -> dict:
    """Execute a single block placement with explicit coordinates.

    Call this once per block after detect_kit_plan. Completes in ~25-30 seconds.
    Call in seq order (seq=1 first, then seq=2, etc.).

    Parameters:
        placement: dict with:
            - color:   block color id (e.g. "green")
            - cell:    target grid cell (e.g. "A2")
            - seq:     sequence number (for ordering)
            - pick_x:  arm x coordinate to pick from (mm)
            - pick_y:  arm y coordinate to pick from (mm)
            - place_x: arm x coordinate to place at (mm)
            - place_y: arm y coordinate to place at (mm)

    Returns placement result with color, cell, seq, and pick/place status.
    """
    if not hasattr(_mgr.backend, "execute_single_placement"):
        return {"error": "execute_single_placement not supported by current backend"}
    completed = _mgr.backend.execute_single_placement(placement)
    return completed


@mcp.tool()
def execute_plan(placements: list[dict]) -> dict:
    """Execute a pre-resolved kit plan with explicit coordinates.

    This is phase 2 of the two-phase kit plan flow. Takes the output of
    detect_kit_plan (with pick/place coordinates added) and executes the
    placements in seq order.

    Parameters:
        placements: list of dicts, each with:
            - color:   block color id
            - cell:    target grid cell
            - seq:     execution order (sorted ascending)
            - pick_x:  arm x coordinate to pick from (mm)
            - pick_y:  arm y coordinate to pick from (mm)
            - place_x: arm x coordinate to place at (mm)
            - place_y: arm y coordinate to place at (mm)

    Returns list of completed placements with color, cell, seq.
    """
    if not hasattr(_mgr.backend, "execute_plan"):
        return {"error": "execute_plan not supported by current backend"}
    completed = _mgr.backend.execute_plan(placements)
    return {"completed": completed}


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
        # handle_signals=False: aiohttp's default tries to register a
        # SIGINT handler for graceful shutdown, which only works in the
        # process's MAIN thread -- this runs in a background thread, so
        # that registration fails and crashes the whole process
        # ("set_wakeup_fd only works in main thread of the main
        # interpreter"). The main MCP server (mcp.run()) already owns
        # signal handling for the process; this bridge server doesn't
        # need its own.
        web.run_app(app, host="127.0.0.1", port=8802, print=None, loop=loop,
                    handle_signals=False)

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
