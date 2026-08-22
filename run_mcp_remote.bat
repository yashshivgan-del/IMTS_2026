@echo off
REM Launch block-sorting MCP server in SSE mode for remote clients (Karini AI platform)
cd /d "C:\Users\karin\Desktop\IMTS 2026\BlockSortingDemo"
"C:\Users\karin\Desktop\IMTS 2026\BlockSortingDemo\.venv\Scripts\python.exe" -m sim.mcp_server --transport sse --host 0.0.0.0 --port 8803
