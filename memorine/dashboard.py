"""
Memorine dashboard — terminal UI for browsing agent memory.
Requires: pip install memorine[ui]
"""

from datetime import datetime

from textual.app import App, ComposeWidget
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    DataTable, Footer, Header, Input, Label, Static, TabbedContent, TabPane,
)

from . import Mind, amygdala


class FactsPane(Static):
    """Shows all facts with decay weights."""

    def compose(self):
        yield DataTable(id="facts-table")

    def on_mount(self):
        table = self.query_one("#facts-table", DataTable)
        table.add_columns("ID", "Fact", "Category", "Confidence", "Weight", "Accesses")
        self.refresh_data()

    def refresh_data(self):
        import time
        table = self.query_one("#facts-table", DataTable)
        table.clear()
        brain = self.app.brain
        now = time.time()
        facts = brain.facts(limit=200)
        for f in facts:
            ew = amygdala.effective_weight(f, now)
            fact_text = f["fact"][:80] + "..." if len(f["fact"]) > 80 else f["fact"]
            table.add_row(
                str(f["id"]),
                fact_text,
                f["category"],
                f"{f['confidence']:.2f}",
                f"{ew:.3f}",
                str(f["access_count"]),
            )


class EventsPane(Static):
    """Shows event timeline."""

    def compose(self):
        yield DataTable(id="events-table")

    def on_mount(self):
        table = self.query_one("#events-table", DataTable)
        table.add_columns("ID", "Time", "Event", "Tags")
        self.refresh_data()

    def refresh_data(self):
        table = self.query_one("#events-table", DataTable)
        table.clear()
        brain = self.app.brain
        events = brain.timeline(limit=200)
        for e in events:
            ts = datetime.fromtimestamp(e["timestamp"]).strftime("%Y-%m-%d %H:%M")
            event_text = e["event"][:80] + "..." if len(e["event"]) > 80 else e["event"]
            tags = e.get("tags", "") or ""
            if isinstance(tags, list):
                tags = ", ".join(tags)
            table.add_row(str(e["id"]), ts, event_text, tags)


class ProceduresPane(Static):
    """Shows procedure success rates."""

    def compose(self):
        yield DataTable(id="procs-table")

    def on_mount(self):
        table = self.query_one("#procs-table", DataTable)
        table.add_columns("Name", "Runs", "Success", "Failures", "Rate", "Avg Duration")
        self.refresh_data()

    def refresh_data(self):
        table = self.query_one("#procs-table", DataTable)
        table.clear()
        brain = self.app.brain
        procs = brain.procedures()
        for p in procs:
            total = p["total_runs"]
            rate = f"{p['successes'] / total * 100:.0f}%" if total > 0 else "-"
            dur = f"{p['avg_duration_ms']:.0f}ms" if p["avg_duration_ms"] else "-"
            table.add_row(
                p["name"],
                str(total),
                str(p["successes"]),
                str(p["failures"]),
                rate,
                dur,
            )


class SearchResults(Static):
    """Shows search results."""

    def compose(self):
        yield DataTable(id="search-table")

    def on_mount(self):
        table = self.query_one("#search-table", DataTable)
        table.add_columns("ID", "Fact", "Weight", "Category")

    def show_results(self, results):
        table = self.query_one("#search-table", DataTable)
        table.clear()
        for r in results:
            fact_text = r["fact"][:80] + "..." if len(r["fact"]) > 80 else r["fact"]
            table.add_row(
                str(r["id"]),
                fact_text,
                f"{r['effective_weight']:.3f}",
                r["category"],
            )


class MemorineDashboard(App):
    """Terminal dashboard for browsing Memorine memory."""

    CSS = """
    #search-input {
        dock: top;
        margin: 0 1;
    }
    #stats-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    TabbedContent {
        height: 1fr;
    }
    DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("/", "focus_search", "Search"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, agent_id="default", db_path=None):
        super().__init__()
        self.brain = Mind(agent_id, db_path=db_path)
        self._agent_id = agent_id

    def compose(self):
        yield Header(show_clock=True)
        yield Input(placeholder="Search memory... (press / to focus)", id="search-input")
        with TabbedContent("Facts", "Events", "Procedures", "Search"):
            with TabPane("Facts"):
                yield FactsPane()
            with TabPane("Events"):
                yield EventsPane()
            with TabPane("Procedures"):
                yield ProceduresPane()
            with TabPane("Search"):
                yield SearchResults()
        yield Label(self._build_stats(), id="stats-bar")
        yield Footer()

    def _build_stats(self):
        s = self.brain.stats()
        emb = " | embeddings: on" if s["embeddings"] > 0 else ""
        return (
            f" agent: {self._agent_id} | "
            f"facts: {s['facts_active']} | "
            f"events: {s['events_total']} | "
            f"procedures: {s['procedures_total']}{emb}"
        )

    def on_input_submitted(self, event):
        query = event.value.strip()
        if not query:
            return
        results = self.brain.recall(query, limit=20)
        search_pane = self.query_one(SearchResults)
        search_pane.show_results(results)
        # Switch to search tab
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-4"

    def action_focus_search(self):
        self.query_one("#search-input", Input).focus()

    def action_refresh(self):
        for pane in self.query(FactsPane):
            pane.refresh_data()
        for pane in self.query(EventsPane):
            pane.refresh_data()
        for pane in self.query(ProceduresPane):
            pane.refresh_data()
        stats_bar = self.query_one("#stats-bar", Label)
        stats_bar.update(self._build_stats())

    @property
    def title(self):
        return f"Memorine - {self._agent_id}"


def run_dashboard(agent_id="default", db_path=None):
    app = MemorineDashboard(agent_id=agent_id, db_path=db_path)
    app.run()
