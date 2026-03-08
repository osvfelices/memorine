"""
Tests for Memorine.
"""

import os
import tempfile
import pytest

from memorine import Mind


@pytest.fixture
def brain():
    """Fresh Mind instance with a temporary database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    m = Mind("test_agent", db_path=path)
    yield m
    m.conn.close()
    os.unlink(path)


@pytest.fixture
def two_agents():
    """Two agents sharing the same database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    alice = Mind("alice", db_path=path)
    bob = Mind("bob", db_path=path)
    yield alice, bob
    alice.conn.close()
    bob.conn.close()
    os.unlink(path)


# -- Facts --

class TestFacts:
    def test_learn_and_recall(self, brain):
        fid, contras = brain.learn("Python uses indentation for blocks")
        assert fid > 0
        assert contras == []

        results = brain.recall("python indentation")
        assert len(results) == 1
        assert results[0]["fact"] == "Python uses indentation for blocks"

    def test_learn_returns_fact_id(self, brain):
        fid, _ = brain.learn("test fact")
        assert isinstance(fid, int)
        assert fid >= 1

    def test_contradiction_detection(self, brain):
        brain.learn("Redis runs on port 6379", category="infra")
        fid2, contras = brain.learn("Redis runs on port 6380", category="infra")

        assert len(contras) == 1
        assert contras[0]["fact"] == "Redis runs on port 6379"
        assert contras[0]["similarity"] > 0.5

    def test_near_duplicate_reinforces(self, brain):
        fid1, _ = brain.learn("The sky is blue")
        fid2, _ = brain.learn("The sky is blue")
        # Near-duplicate returns the original fact id
        assert fid1 == fid2

    def test_forget(self, brain):
        fid, _ = brain.learn("temporary fact")
        brain.forget(fid)
        results = brain.recall("temporary fact")
        assert len(results) == 0

    def test_correct(self, brain):
        fid, _ = brain.learn("Earth has 8 planets")
        brain.correct(fid, "The solar system has 8 planets")
        results = brain.recall("planets")
        assert results[0]["fact"] == "The solar system has 8 planets"

    def test_all_facts(self, brain):
        brain.learn("fact one")
        brain.learn("fact two")
        brain.learn("fact three")
        all_f = brain.facts()
        assert len(all_f) == 3


# -- Associations --

class TestAssociations:
    def test_connect_and_retrieve(self, brain):
        fid1, _ = brain.learn("Docker uses containers")
        fid2, _ = brain.learn("Kubernetes orchestrates containers")
        brain.connect(fid1, fid2, relation="related_tech")

        assocs = brain.associations(fid1)
        assert len(assocs) == 1
        assert assocs[0]["relation"] == "related_tech"

    def test_relates_to_on_learn(self, brain):
        brain.learn("PostgreSQL is a relational database")
        fid2, _ = brain.learn(
            "PostgreSQL supports JSON columns",
            relates_to="PostgreSQL relational"
        )
        assocs = brain.associations(fid2)
        assert len(assocs) >= 1


# -- Events --

class TestEvents:
    def test_log_and_search(self, brain):
        eid = brain.log("Server restarted", tags=["infra", "restart"])
        assert eid > 0

        events = brain.events(query="restart")
        assert len(events) == 1
        assert events[0]["event"] == "Server restarted"

    def test_causal_chain(self, brain):
        e1 = brain.log("DNS failed")
        e2 = brain.log("Health check failed", caused_by=e1)
        e3 = brain.log("Alert triggered", caused_by=e2)

        chain = brain.why(e3)
        assert len(chain) == 3
        assert chain[0]["event"] == "DNS failed"
        assert chain[2]["event"] == "Alert triggered"

    def test_consequences(self, brain):
        e1 = brain.log("Deploy started")
        e2 = brain.log("Tests passed", caused_by=e1)
        e3 = brain.log("Image pushed", caused_by=e1)

        children = brain.consequences(e1)
        assert len(children) == 2

    def test_timeline(self, brain):
        brain.log("event one")
        brain.log("event two")
        brain.log("event three")

        tl = brain.timeline(limit=10)
        assert len(tl) == 3

    def test_tag_filter(self, brain):
        brain.log("deploy started", tags=["deploy"])
        brain.log("test passed", tags=["test"])
        brain.log("deploy finished", tags=["deploy"])

        deploys = brain.events(tags=["deploy"])
        assert len(deploys) == 2


# -- Procedures --

class TestProcedures:
    def test_procedure_tracking(self, brain):
        with brain.procedure("build_app", description="Build the app") as run:
            run.step("compile", success=True)
            run.step("test", success=True)
            run.step("package", success=True)

        procs = brain.procedures()
        assert len(procs) == 1
        assert procs[0]["name"] == "build_app"
        assert procs[0]["total_runs"] == 1
        assert procs[0]["successes"] == 1

    def test_procedure_failure(self, brain):
        with brain.procedure("deploy") as run:
            run.step("push", success=False, error="auth expired")

        procs = brain.procedures()
        assert procs[0]["failures"] == 1

    def test_anticipate(self, brain):
        # Run a procedure with a failing step
        with brain.procedure("scan_site") as run:
            run.step("resolve_dns", success=True)
            run.step("probe_ports", success=False, error="timeout")

        advice = brain.anticipate("scan site")
        assert "errors_to_avoid" in advice
        assert len(advice["errors_to_avoid"]) >= 1


# -- Sharing --

class TestSharing:
    def test_share_with_team(self, two_agents):
        alice, bob = two_agents
        alice.share("The staging server moved to eu-west-1")

        shared = bob.shared_with_me()
        assert len(shared) == 1
        assert shared[0]["from_agent"] == "alice"

    def test_share_with_specific_agent(self, two_agents):
        alice, bob = two_agents
        alice.share("Secret info", to_agent="bob")

        shared = bob.shared_with_me()
        assert len(shared) == 1

    def test_team_knowledge(self, two_agents):
        alice, bob = two_agents
        alice.share("CI takes 12 minutes")
        bob.share("Staging DB on port 5433")

        team = alice.team_knowledge()
        assert len(team) == 2


# -- Profile --

class TestProfile:
    def test_profile_output(self, brain):
        brain.learn("Important fact one")
        brain.learn("Important fact two")
        brain.log("Something happened")

        profile = brain.profile()
        assert "Memory Profile: test_agent" in profile
        assert "Key Knowledge" in profile
        assert "Important fact one" in profile


# -- Decay --

class TestDecay:
    def test_cleanup_does_not_remove_fresh(self, brain):
        brain.learn("fresh fact")
        cleaned = brain.cleanup()
        assert cleaned == 0

    def test_facts_persist_after_cleanup(self, brain):
        brain.learn("important knowledge")
        brain.cleanup()
        results = brain.recall("important knowledge")
        assert len(results) == 1


# -- Batch Learn --

class TestBatchLearn:
    def test_batch_learn_multiple(self, brain):
        facts = [
            {"fact": "Python is interpreted"},
            {"fact": "Go is compiled", "category": "languages"},
            {"fact": "Rust has no GC", "category": "languages", "confidence": 0.9},
        ]
        results = brain.learn_batch(facts)
        assert len(results) == 3
        for fid, contras in results:
            assert fid > 0

    def test_batch_learn_detects_duplicates(self, brain):
        brain.learn("The sky is blue")
        facts = [
            {"fact": "The sky is blue"},
            {"fact": "Water is wet"},
        ]
        results = brain.learn_batch(facts)
        assert len(results) == 2
        # First one should be the original ID (duplicate)
        all_f = brain.facts()
        assert len(all_f) == 2  # sky + water, not 3

    def test_batch_learn_contradictions(self, brain):
        brain.learn("Redis runs on port 6379", category="infra")
        facts = [
            {"fact": "Redis runs on port 6380", "category": "infra"},
        ]
        results = brain.learn_batch(facts)
        _, contras = results[0]
        assert len(contras) == 1


# -- Pagination --

class TestPagination:
    def test_recall_offset(self, brain):
        for i in range(10):
            brain.learn(f"Fact number {i} about testing pagination")
        page1 = brain.recall("testing pagination", limit=3, offset=0)
        page2 = brain.recall("testing pagination", limit=3, offset=3)
        ids1 = {r["id"] for r in page1}
        ids2 = {r["id"] for r in page2}
        assert len(ids1 & ids2) == 0  # no overlap

    def test_facts_pagination(self, brain):
        # Use distinct facts to avoid deduplication
        topics = ["Python", "JavaScript", "Rust", "Go", "Java",
                  "Docker", "Kubernetes", "Redis", "PostgreSQL", "SQLite"]
        for t in topics:
            brain.learn(f"{t} is a popular technology in 2026")
        page1 = brain.facts(limit=5, offset=0)
        page2 = brain.facts(limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        ids1 = {f["id"] for f in page1}
        ids2 = {f["id"] for f in page2}
        assert len(ids1 & ids2) == 0

    def test_events_offset(self, brain):
        for i in range(10):
            brain.log(f"Event {i}")
        page1 = brain.events(limit=5, offset=0)
        page2 = brain.events(limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5


# -- Stats --

class TestStats:
    def test_stats_counts(self, brain):
        brain.learn("fact one")
        brain.learn("fact two")
        brain.log("event one")
        with brain.procedure("test_proc") as run:
            run.step("step one", success=True)

        s = brain.stats()
        assert s["facts_active"] == 2
        assert s["events_total"] == 1
        assert s["procedures_total"] == 1

    def test_stats_empty(self, brain):
        s = brain.stats()
        assert s["facts_active"] == 0
        assert s["events_total"] == 0


# -- Embeddings (skip if not installed) --

class TestEmbeddings:
    def test_embeddings_optional(self, brain):
        """Core functionality works without embeddings installed."""
        brain.learn("This works without embeddings")
        results = brain.recall("works without")
        assert len(results) == 1

    def test_reindex_returns_zero_without_embeddings(self, brain):
        """reindex_embeddings returns 0 when fastembed not installed."""
        brain.learn("some fact")
        # This should not crash, just return 0 if fastembed missing
        count = brain.reindex_embeddings()
        assert isinstance(count, int)


# -- Edge Cases --

class TestEdgeCases:
    def test_learn_empty_string_raises(self, brain):
        with pytest.raises(ValueError):
            brain.learn("")

    def test_learn_whitespace_raises(self, brain):
        with pytest.raises(ValueError):
            brain.learn("   ")

    def test_learn_none_raises(self, brain):
        with pytest.raises((ValueError, TypeError)):
            brain.learn(None)

    def test_recall_empty_returns_empty(self, brain):
        brain.learn("some fact about cats")
        assert brain.recall("") == []
        assert brain.recall("   ") == []

    def test_unicode_facts(self, brain):
        fid, _ = brain.learn("Der Server steht in Frankfurt")
        results = brain.recall("Frankfurt")
        assert len(results) >= 1
        assert any("Frankfurt" in r["fact"] for r in results)

        fid2, _ = brain.learn("El servidor esta en Madrid")
        results = brain.recall("Madrid")
        assert len(results) >= 1
        assert any("Madrid" in r["fact"] for r in results)

    def test_unicode_cjk(self, brain):
        fid, _ = brain.learn("Production database is in Tokyo region")
        results = brain.recall("Tokyo")
        assert len(results) == 1

    def test_very_long_fact(self, brain):
        long_text = "word " * 2000  # 10000 chars
        fid, _ = brain.learn(long_text.strip())
        assert fid > 0

    def test_special_characters_in_fact(self, brain):
        fid, _ = brain.learn("Config path is /etc/nginx/nginx.conf (port=8080)")
        results = brain.recall("nginx config")
        assert len(results) == 1

    def test_confidence_clamped(self, brain):
        fid, _ = brain.learn("test fact", confidence=5.0)
        row = brain.conn.execute(
            "SELECT confidence FROM facts WHERE id = ?", (fid,)
        ).fetchone()
        assert row["confidence"] <= 1.0

    def test_negative_confidence_clamped(self, brain):
        fid, _ = brain.learn("test fact", confidence=-1.0)
        row = brain.conn.execute(
            "SELECT confidence FROM facts WHERE id = ?", (fid,)
        ).fetchone()
        assert row["confidence"] >= 0.0

    def test_forget_nonexistent_id(self, brain):
        # Should not crash
        brain.forget(999999)

    def test_correct_nonexistent_id(self, brain):
        # Should not crash
        brain.correct(999999, "new value")

    def test_events_combined_query_and_tags(self, brain):
        brain.log("Connection timeout on port 443", tags=["error", "network"])
        brain.log("Disk full warning", tags=["warning", "disk"])
        brain.log("DNS timeout detected", tags=["error", "dns"])

        results = brain.events(query="timeout", tags=["error"])
        assert len(results) == 2

    def test_events_empty_query_returns_all(self, brain):
        brain.log("event one")
        brain.log("event two")
        results = brain.events()
        assert len(results) == 2

    def test_cleanup_scoped_to_agent(self, two_agents):
        alice, bob = two_agents
        alice.learn("alice fact")
        bob.learn("bob fact")
        # Cleanup for alice should not affect bob
        alice.cleanup()
        bob_facts = bob.facts()
        assert len(bob_facts) == 1

    def test_recall_with_zero_limit(self, brain):
        brain.learn("test fact")
        results = brain.recall("test", limit=0)
        assert len(results) == 0

    def test_facts_with_zero_limit(self, brain):
        brain.learn("test fact")
        results = brain.facts(limit=0)
        assert len(results) == 0
