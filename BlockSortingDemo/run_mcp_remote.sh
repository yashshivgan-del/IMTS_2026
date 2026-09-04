#!/usr/bin/env bash
set -e
# Launch block-sorting MCP server in SSE mode for remote clients (Karini AI platform)
cd "$(dirname "$0")"
if [ -f ".venv/bin/python" ]; then
    exec .venv/bin/python -m sim.mcp_server --transport sse --host 0.0.0.0 --port 8803 "$@"
else
    exec python3 -m sim.mcp_server --transport sse --host 0.0.0.0 --port 8803 "$@"
fi
