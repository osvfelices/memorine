"""
Memorine CLI — dispatches to MCP server or dashboard.

    memorine          → starts the MCP server (default)
    memorine ui       → launches the terminal dashboard
    memorine stats    → prints database statistics
    memorine reindex  → rebuilds embeddings for all facts
"""

import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"

    if cmd in ("ui", "dashboard"):
        try:
            from .dashboard import run_dashboard
        except ImportError:
            print("Dashboard requires textual: pip install memorine[ui]")
            sys.exit(1)
        agent_id = _get_flag("--agent") or "default"
        db_path = _get_flag("--db")
        run_dashboard(agent_id, db_path)

    elif cmd == "stats":
        from . import Mind
        agent_id = _get_flag("--agent") or "default"
        db_path = _get_flag("--db")
        brain = Mind(agent_id, db_path=db_path)
        s = brain.stats()
        print(f"Agent: {agent_id}")
        print(f"Facts (active):   {s['facts_active']}")
        print(f"Facts (inactive): {s['facts_inactive']}")
        print(f"Events:           {s['events_total']}")
        print(f"Procedures:       {s['procedures_total']}")
        print(f"Shared:           {s['shared_total']}")
        print(f"Embeddings:       {s['embeddings']}")
        size = s['db_size_bytes']
        if size > 1_000_000:
            print(f"DB size:          {size / 1_000_000:.1f} MB")
        elif size > 1_000:
            print(f"DB size:          {size / 1_000:.1f} KB")
        else:
            print(f"DB size:          {size} bytes")

    elif cmd == "reindex":
        from . import Mind
        agent_id = _get_flag("--agent") or "default"
        db_path = _get_flag("--db")
        brain = Mind(agent_id, db_path=db_path)
        count = brain.reindex_embeddings()
        if count:
            print(f"Reindexed {count} facts for agent '{agent_id}'.")
        else:
            print("No embeddings reindexed. Install memorine[embeddings] first.")

    elif cmd in ("serve", "--help", "-h"):
        if cmd in ("--help", "-h"):
            print(__doc__.strip())
            sys.exit(0)
        from .mcp_server import main as mcp_main
        mcp_main()

    else:
        # Default: treat unknown args as MCP server (backwards compatible)
        from .mcp_server import main as mcp_main
        mcp_main()


def _get_flag(name):
    """Extract a --flag value from sys.argv."""
    for i, arg in enumerate(sys.argv):
        if arg == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None
