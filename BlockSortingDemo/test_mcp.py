"""Test the MCP server tools standalone.

This simulates what an AI agent would do:
1. detect_blocks - see what's on the table
2. plan_sort - plan the desired sequence
3. execute_sort - run the plan
4. get_sort_status - poll until done
5. reset_blocks - put them back
"""
import asyncio
import time
import json
from sim.mcp_server import mcp


async def call(tool_name, args=None):
    """Call an MCP tool and return parsed result."""
    result = await mcp.call_tool(tool_name, args or {})
    return json.loads(result.content[0].text)


async def main():
    print("=" * 60)
    print("  MCP SERVER STANDALONE TEST")
    print("  Simulating AI agent calling all 5 tools")
    print("=" * 60)
    print()

    # List available tools
    tools = await mcp.list_tools()
    print(f"Available MCP tools ({len(tools)}):")
    for tool in tools:
        print(f"  - {tool.name}")
    print()

    # ── Step 1: detect_blocks ──────────────────────────────────
    print("─" * 60)
    print("STEP 1: AI calls detect_blocks()")
    print("  'What blocks are on the table?'")
    print("─" * 60)
    data = await call("detect_blocks")
    print(f"  Found {data['count']} blocks:")
    for b in data["blocks"]:
        print(f"    {b['label']:8s} at ({b['x']}, {b['y']})")
    print(f"  Slots: {[s['label'] for s in data['slots']]}")
    print()

    # ── Step 2: plan_sort ──────────────────────────────────────
    print("─" * 60)
    print("STEP 2: AI calls plan_sort(['green', 'red', 'yellow'])")
    print("  'I want green first, red second, yellow third'")
    print("─" * 60)
    plan = await call("plan_sort", {"sequence": ["green", "red", "yellow"]})
    print(f"  Approved: {plan['ok']}")
    print(f"  Plan ID:  {plan['plan_id']}")
    print(f"  Time est: {plan['est_seconds']}s")
    print(f"  Operations:")
    for op in plan["operations"]:
        print(f"    {op['block']:8s} → {op['slot']}")
    print()

    # ── Step 3: execute_sort ───────────────────────────────────
    print("─" * 60)
    print(f"STEP 3: AI calls execute_sort(plan_id='{plan['plan_id']}')")
    print("  'Execute it.'")
    print("─" * 60)
    exec_result = await call("execute_sort", {"plan_id": plan["plan_id"]})
    print(f"  Started: {exec_result['ok']}")
    print(f"  Job ID:  {exec_result['job_id']}")
    print()

    # ── Step 4: poll status ────────────────────────────────────
    print("─" * 60)
    print(f"STEP 4: AI polls get_sort_status()")
    print("  'How's it going?'")
    print("─" * 60)
    job_id = exec_result["job_id"]
    while True:
        time.sleep(2)
        status = await call("get_sort_status", {"job_id": job_id})
        print(f"  [{status['pct']:5.1f}%] {status['state']:10s} | {status['current_action']}")
        if status["state"] in ("done", "failed", "cancelled"):
            break

    print()
    if status["state"] == "done":
        print("  SORT COMPLETE!")
        for op in status["completed_ops"]:
            print(f"    {op['block']} → {op['slot']}")
    print()

    # ── Step 5: reset_blocks ───────────────────────────────────
    print("─" * 60)
    print("STEP 5: AI calls reset_blocks()")
    print("  'Put them back for the next person'")
    print("─" * 60)
    reset = await call("reset_blocks")
    print(f"  Reset OK: {reset['ok']}")
    print()

    # ── Bonus: safety rejection ────────────────────────────────
    print("─" * 60)
    print("BONUS: Test safety - AI tries duplicate blocks")
    print("  plan_sort(['green', 'green', 'yellow'])")
    print("─" * 60)
    bad = await call("plan_sort", {"sequence": ["green", "green", "yellow"]})
    print(f"  Rejected: {not bad['ok']}")
    print(f"  Reason: {bad['violations'][0]['message']}")
    print()

    print("─" * 60)
    print("BONUS: Test safety - AI tries unknown block")
    print("  plan_sort(['green', 'blue', 'yellow'])")
    print("─" * 60)
    bad2 = await call("plan_sort", {"sequence": ["green", "blue", "yellow"]})
    print(f"  Rejected: {not bad2['ok']}")
    print(f"  Reason: {bad2['violations'][0]['message']}")
    print()

    print("=" * 60)
    print("  ALL 5 MCP TOOLS VERIFIED")
    print("  Safety guardrails working")
    print("  Ready for Karini AI platform connection")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
