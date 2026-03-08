"""
Memorine — Human-like memory for AI agents.
Zero dependencies. Zero tokens. Zero cost. Just memory that works.

    from memorine import Mind

    brain = Mind("marc")
    brain.learn("abuse@hostkey.nl is the contact for HOSTKEY")
    brain.recall("hostkey")  # finds it + reinforces the memory
"""

__version__ = "0.2.0"

import time

from .db import get_connection, init_schema, get_stats
from . import cortex, hippocampus, cerebellum, amygdala, synapses


class Mind:
    """A single agent's mind. Each agent gets their own Mind instance.

    Usage:
        brain = Mind("marc")
        brain.learn("some fact")
        brain.recall("search query")
    """

    def __init__(self, agent_id, db_path=None):
        self.agent_id = agent_id
        self.conn = get_connection(db_path)
        init_schema(self.conn)

    # ── CORTEX: Facts ──────────────────────────────────────────

    def learn(self, fact, category="general", confidence=1.0,
              source=None, weight=None, relates_to=None):
        """Store a fact. Returns (fact_id, contradictions).

        Automatically detects contradictions with existing knowledge.
        Near-duplicates are reinforced instead of duplicated.
        """
        return cortex.learn(
            self.conn, self.agent_id, fact,
            category=category, confidence=confidence,
            source=source, weight=weight, relates_to=relates_to
        )

    def learn_batch(self, facts):
        """Batch-learn multiple facts at once. Much faster for bulk imports.

        facts: list of dicts with keys: fact, category, confidence, source, weight
        Returns: list of (fact_id, contradictions) tuples.
        """
        return cortex.learn_batch(self.conn, self.agent_id, facts)

    def recall(self, query, limit=5, offset=0, include_shared=True):
        """Search memory for facts matching a query.

        Uses semantic search when embeddings are available, with FTS5
        as a complement and fallback. Results ranked by a blend of
        semantic similarity and effective weight. Accessed memories get
        reinforced, just like human recall strengthens memory.
        """
        return cortex.recall(
            self.conn, self.agent_id, query,
            limit=limit, offset=offset, include_shared=include_shared
        )

    def forget(self, fact_id):
        """Soft-delete a fact. Only works on facts owned by this agent."""
        cortex.forget(self.conn, fact_id, self.agent_id)

    def correct(self, fact_id, new_value, confidence=None):
        """Update a fact that turned out to be wrong. Only works on own facts."""
        cortex.update_fact(self.conn, fact_id, new_value, self.agent_id, confidence)

    def connect(self, fact_a, fact_b, relation="related"):
        """Create an association between two facts. At least one must be yours."""
        cortex.link(self.conn, fact_a, fact_b, relation, agent_id=self.agent_id)

    def associations(self, fact_id, depth=1):
        """Get facts associated with a given fact."""
        return cortex.associations(self.conn, fact_id, depth)

    def facts(self, limit=None, offset=0):
        """List all active facts."""
        return cortex.all_facts(self.conn, self.agent_id, limit=limit, offset=offset)

    # ── HIPPOCAMPUS: Events ────────────────────────────────────

    def log(self, event, context=None, tags=None, caused_by=None):
        """Record something that happened.

        Use caused_by to build causal chains:
            e1 = brain.log("DNS timeout on domain X")
            e2 = brain.log("Scan failed", caused_by=e1)
        """
        return hippocampus.log_event(
            self.conn, self.agent_id, event,
            context=context, tags=tags, caused_by=caused_by
        )

    def events(self, query=None, since=None, until=None, tags=None,
               limit=20, offset=0):
        """Search past events by text, time range, or tags."""
        return hippocampus.recall_events(
            self.conn, self.agent_id, query=query,
            since=since, until=until, tags=tags, limit=limit, offset=offset
        )

    def why(self, event_id):
        """Trace the causal chain: what caused this event?"""
        return hippocampus.causal_chain(self.conn, event_id, "up")

    def consequences(self, event_id):
        """What did this event cause?"""
        return hippocampus.causal_chain(self.conn, event_id, "down")

    def timeline(self, since=None, until=None, limit=50):
        """Get chronological event timeline."""
        return hippocampus.timeline(
            self.conn, self.agent_id, since=since, until=until, limit=limit
        )

    # ── CEREBELLUM: Procedures ─────────────────────────────────

    def procedure(self, name, description=None, steps=None):
        """Get or create a procedure. Returns a ProcedureRun context manager.

        Usage:
            with brain.procedure("scan_site") as run:
                run.step("detect_cdn", success=True)
                run.step("probe_subdomains", success=False, error="timeout")
        """
        proc = cerebellum.get_procedure(self.conn, self.agent_id, name)
        if not proc:
            proc_id = cerebellum.create_procedure(
                self.conn, self.agent_id, name, description, steps
            )
        else:
            proc_id = proc["id"]
        return ProcedureRun(self.conn, proc_id)

    def anticipate(self, task_description):
        """Predict what you'll need for a task.

        Returns best procedure, recommended steps, warnings about
        steps that often fail, and past errors to avoid.
        """
        return cerebellum.anticipate(
            self.conn, self.agent_id, task_description
        )

    def procedures(self):
        """List all active procedures."""
        return cerebellum.list_procedures(self.conn, self.agent_id)

    # ── AMYGDALA: Maintenance ──────────────────────────────────

    def cleanup(self, threshold=0.05):
        """Deactivate faded memories below threshold."""
        return amygdala.cleanup_faded(self.conn, self.agent_id, threshold)

    def stats(self):
        """Database statistics: fact counts, events, procedures, db size."""
        return get_stats(self.conn, self.agent_id)

    def reindex_embeddings(self):
        """Rebuild all embeddings. Run this after installing memorine[embeddings]."""
        try:
            from . import embeddings
            if embeddings.is_available():
                return embeddings.reindex_all(self.conn, self.agent_id)
        except Exception:
            pass
        return 0

    # ── SYNAPSES: Sharing ──────────────────────────────────────

    def share(self, fact_text, to_agent=None, category="shared"):
        """Learn a fact and share it with another agent (or everyone)."""
        return synapses.share_fact(
            self.conn, self.agent_id, fact_text,
            to_agent=to_agent, category=category
        )

    def shared_with_me(self, limit=20):
        """Get facts other agents have shared with me."""
        return synapses.shared_with_me(self.conn, self.agent_id, limit)

    def team_knowledge(self, category=None, limit=50):
        """Get collective team knowledge."""
        return synapses.team_knowledge(self.conn, category, limit)

    # ── PROFILE ────────────────────────────────────────────────

    def profile(self, max_facts=20, max_events=10):
        """Generate a cognitive profile — a summary of what this agent knows.

        Returns a plain text block ready to inject into a system prompt.
        No LLM needed — built from structured data.
        """
        now = time.time()
        lines = [f"# Memory Profile: {self.agent_id}"]

        # Top facts by effective weight
        all_f = cortex.all_facts(self.conn, self.agent_id)
        weighted = []
        for f in all_f:
            ew = amygdala.effective_weight(f, now)
            if ew > 0.1:
                weighted.append((ew, f))
        weighted.sort(key=lambda x: x[0], reverse=True)

        if weighted:
            lines.append("\n## Key Knowledge")
            for ew, f in weighted[:max_facts]:
                lines.append(f"- {f['fact']} [{f['category']}]")

        # Shared knowledge
        shared = synapses.shared_with_me(self.conn, self.agent_id, limit=10)
        if shared:
            lines.append("\n## Team Knowledge")
            for s in shared:
                lines.append(f"- {s['fact']} (from {s['from_agent']})")

        # Recent events
        recent = hippocampus.timeline(self.conn, self.agent_id, limit=max_events)
        if recent:
            lines.append("\n## Recent Activity")
            for e in recent:
                lines.append(f"- {e['event']}")

        # Active procedures + warnings
        procs = cerebellum.list_procedures(self.conn, self.agent_id)
        if procs:
            lines.append("\n## Known Procedures")
            for p in procs:
                rate = ""
                if p["total_runs"] > 0:
                    sr = round(p["successes"] / p["total_runs"] * 100)
                    rate = f" ({sr}% success, {p['total_runs']} runs)"
                lines.append(f"- {p['name']}{rate}")

        return "\n".join(lines)


class ProcedureRun:
    """Context manager for tracking a procedure execution."""

    def __init__(self, conn, procedure_id):
        self.conn = conn
        self.procedure_id = procedure_id
        self.run_id = None
        self._step_count = 0
        self._success = True
        self._error = None

    def __enter__(self):
        self.run_id = cerebellum.start_run(self.conn, self.procedure_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._success = False
            self._error = str(exc_val)
        cerebellum.complete_run(
            self.conn, self.run_id, self._success, self._error
        )
        return False

    def step(self, description, success=True, error=None, duration_ms=None):
        """Record a step result."""
        self._step_count += 1
        if not success:
            self._success = False
            self._error = error
        cerebellum.log_step(
            self.conn, self.run_id, self._step_count, description,
            success=success, error=error, duration_ms=duration_ms
        )
