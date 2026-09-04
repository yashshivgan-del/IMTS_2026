#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ -f ".venv/bin/python" ]; then
    exec .venv/bin/python -m sim.mcp_server "$@"
else
    exec python3 -m sim.mcp_server "$@"
fi
