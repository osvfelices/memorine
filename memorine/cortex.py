"""
Memorine cortex — facts, associations, and contradiction detection.
No LLM needed. Pure algorithmic memory.
"""

import re
import time

from . import amygdala


def _tokenize(text):
    """Simple word tokenizer for similarity checks."""
    return set(re.findall(r"\w{3,}", text.lower()))


def _jaccard(a, b):
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _get_embeddings():
    """Try to load the embeddings module. Returns None if unavailable."""
    try:
        from . import embeddings
        if embeddings.is_available():
            return embeddings
    except Exception:
        pass
    return None


def learn(conn, agent_id, fact, category="general", confidence=1.0,
          source=None, weight=None, relates_to=None):
    """Store a new fact. Detects contradictions automatically.

    Returns (fact_id, contradictions) where contradictions is a list
    of older facts that may conflict with this one.
    """
    if not fact or not isinstance(fact, str) or not fact.strip():
        raise ValueError("fact must be a non-empty string")
    if not agent_id or not isinstance(agent_id, str):
        raise ValueError("agent_id must be a non-empty string")
    confidence = max(0.0, min(float(confidence), 1.0))

    now = time.time()
    w = weight if weight is not None else 1.0
    w = max(0.1, min(float(w), 10.0))
    tokens = _tokenize(fact)

    # Check for contradictions: same agent, same category, similar text
    contradictions = []
    existing = conn.execute(
        "SELECT id, fact, confidence, weight FROM facts "
        "WHERE agent_id = ? AND category = ? AND active = 1",
        (agent_id, category)
    ).fetchall()

    for row in existing:
        sim = _jaccard(tokens, _tokenize(row["fact"]))
        if sim >= 0.5 and sim < 0.95:
            contradictions.append({
                "id": row["id"],
                "fact": row["fact"],
                "similarity": round(sim, 3),
            })
            if confidence >= row["confidence"]:
                amygdala.weaken(conn, row["id"])
        elif sim >= 0.95:
            amygdala.reinforce(conn, row["id"])
            return row["id"], []

    cur = conn.execute(
        "INSERT INTO facts (agent_id, fact, category, confidence, weight, "
        "source, created_at, updated_at, last_accessed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (agent_id, fact, category, confidence, w, source, now, now, now)
    )
    fact_id = cur.lastrowid
    conn.commit()

    # Generate embedding if available
    emb = _get_embeddings()
    if emb:
        try:
            emb.embed_fact(conn, fact_id, fact)
        except Exception:
            pass

    # Create association if relates_to was given
    if relates_to:
        link_by_text(conn, agent_id, fact_id, relates_to)

    # Mark contradicted facts as superseded
    for c in contradictions:
        if confidence >= 0.8:
            conn.execute(
                "UPDATE facts SET superseded_by = ? WHERE id = ?",
                (fact_id, c["id"])
            )
    if contradictions:
        conn.commit()

    return fact_id, contradictions


def learn_batch(conn, agent_id, facts_list):
    """Batch-learn multiple facts. Much faster for bulk imports.

    facts_list: list of dicts with keys: fact, category, confidence, source, weight
    Returns: list of (fact_id, contradictions) tuples.
    """
    # Pre-load existing facts for contradiction checking
    all_existing = conn.execute(
        "SELECT id, fact, category, confidence, weight FROM facts "
        "WHERE agent_id = ? AND active = 1",
        (agent_id,)
    ).fetchall()

    # Index by category for fast lookup
    by_category = {}
    for row in all_existing:
        cat = row["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(row)

    now = time.time()
    results = []
    new_facts_for_embedding = []

    for item in facts_list:
        fact_text = item["fact"]
        category = item.get("category", "general")
        confidence = item.get("confidence", 1.0)
        source = item.get("source")
        w = item.get("weight", 1.0)
        tokens = _tokenize(fact_text)

        contradictions = []
        is_duplicate = False

        for row in by_category.get(category, []):
            sim = _jaccard(tokens, _tokenize(row["fact"]))
            if sim >= 0.5 and sim < 0.95:
                contradictions.append({
                    "id": row["id"],
                    "fact": row["fact"],
                    "similarity": round(sim, 3),
                })
                if confidence >= row["confidence"]:
                    amygdala.weaken(conn, row["id"])
            elif sim >= 0.95:
                amygdala.reinforce(conn, row["id"])
                results.append((row["id"], []))
                is_duplicate = True
                break

        if is_duplicate:
            continue

        cur = conn.execute(
            "INSERT INTO facts (agent_id, fact, category, confidence, weight, "
            "source, created_at, updated_at, last_accessed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (agent_id, fact_text, category, confidence, w, source, now, now, now)
        )
        fact_id = cur.lastrowid
        new_facts_for_embedding.append((fact_id, fact_text))

        for c in contradictions:
            if confidence >= 0.8:
                conn.execute(
                    "UPDATE facts SET superseded_by = ? WHERE id = ?",
                    (fact_id, c["id"])
                )

        results.append((fact_id, contradictions))

    conn.commit()

    # Batch-embed all new facts at once
    emb = _get_embeddings()
    if emb and new_facts_for_embedding:
        try:
            emb.embed_facts_batch(conn, new_facts_for_embedding)
        except Exception:
            pass

    return results


def recall(conn, agent_id, query, limit=5, offset=0,
           include_shared=True, min_weight=0.0):
    """Search facts using semantic search (if available) + FTS5.

    Tries semantic search first for meaning-based matches, then merges
    with FTS5 keyword results. Falls back to pure FTS5 if embeddings
    are not installed.
    """
    # Short-circuit on empty or whitespace-only queries
    if not query or not query.strip():
        return []

    now = time.time()
    seen_ids = set()
    candidates = []

    # Try semantic search first
    emb = _get_embeddings()
    if emb:
        try:
            semantic_results = emb.semantic_search(
                conn, query, agent_id, limit=limit * 2,
                include_shared=include_shared
            )
            for row in semantic_results:
                sem_score = row.get("semantic_score", 0.0)
                # Filter out low-relevance noise
                if sem_score < 0.55:
                    continue
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    ew = amygdala.effective_weight(row, now)
                    # Blend semantic score with effective weight
                    score = 0.6 * sem_score + 0.4 * ew
                    candidates.append({
                        "id": row["id"],
                        "fact": row["fact"],
                        "category": row["category"],
                        "confidence": row["confidence"],
                        "effective_weight": ew,
                        "source": row["source"],
                        "agent_id": row["agent_id"],
                        "own": row["agent_id"] == agent_id,
                        "_score": score,
                        "_method": "semantic",
                    })
        except Exception:
            pass

    # FTS5 search (always runs as complement or fallback)
    fts_query = " OR ".join(re.findall(r"\w{3,}", query.lower()))
    if fts_query:
        sql = """
            SELECT f.id, f.fact, f.category, f.confidence, f.weight,
                   f.last_accessed, f.access_count, f.source, f.agent_id,
                   facts_fts.rank
            FROM facts_fts
            JOIN facts f ON f.id = facts_fts.rowid
            WHERE facts_fts MATCH ? AND f.active = 1
        """
        params = [fts_query]

        if include_shared:
            sql += " AND (f.agent_id = ? OR f.id IN (SELECT fact_id FROM shared WHERE to_agent = ? OR to_agent IS NULL))"
            params.extend([agent_id, agent_id])
        else:
            sql += " AND f.agent_id = ?"
            params.append(agent_id)

        sql += " ORDER BY facts_fts.rank LIMIT ?"
        params.append(limit * 3)

        rows = conn.execute(sql, params).fetchall()

        for row in rows:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                ew = amygdala.effective_weight(row, now)
                if ew >= min_weight:
                    candidates.append({
                        "id": row["id"],
                        "fact": row["fact"],
                        "category": row["category"],
                        "confidence": row["confidence"],
                        "effective_weight": ew,
                        "source": row["source"],
                        "agent_id": row["agent_id"],
                        "own": row["agent_id"] == agent_id,
                        "_score": ew,
                        "_method": "fts5",
                    })

    # Sort by blended score
    candidates.sort(key=lambda x: x["_score"], reverse=True)

    # Apply offset and limit
    candidates = candidates[offset:offset + limit]

    # Clean internal fields
    results = []
    for c in candidates:
        c.pop("_score", None)
        c.pop("_method", None)
        results.append(c)

    # Reinforce accessed memories
    for r in results:
        if r["own"]:
            amygdala.reinforce(conn, r["id"], boost=0.05)

    return results


def forget(conn, fact_id, agent_id=None):
    """Deactivate a fact (soft delete). Only the owning agent can forget."""
    sql = "UPDATE facts SET active = 0 WHERE id = ?"
    params = [fact_id]
    if agent_id:
        sql += " AND agent_id = ?"
        params.append(agent_id)
    conn.execute(sql, params)
    # Remove embedding so semantic search never finds forgotten facts
    try:
        conn.execute("DELETE FROM fact_embeddings WHERE fact_id = ?", (fact_id,))
    except Exception:
        pass  # Table may not exist if embeddings not installed
    conn.commit()


def update_fact(conn, fact_id, new_value, agent_id=None, confidence=None):
    """Update a fact's content. Only the owning agent can correct."""
    now = time.time()
    updates = ["fact = ?", "updated_at = ?"]
    params = [new_value, now]
    if confidence is not None:
        updates.append("confidence = ?")
        params.append(confidence)
    params.append(fact_id)
    sql = f"UPDATE facts SET {', '.join(updates)} WHERE id = ?"
    if agent_id:
        sql += " AND agent_id = ?"
        params.append(agent_id)
    conn.execute(sql, params)
    conn.commit()

    # Re-embed if available
    emb = _get_embeddings()
    if emb:
        try:
            emb.embed_fact(conn, fact_id, new_value)
        except Exception:
            pass


def link(conn, fact_a, fact_b, relation="related", strength=1.0, agent_id=None):
    """Create an association between two facts. Validates ownership if agent_id given."""
    if agent_id:
        # Verify at least one fact belongs to the agent
        check = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE id IN (?, ?) AND agent_id = ? AND active = 1",
            (fact_a, fact_b, agent_id)
        ).fetchone()[0]
        if check == 0:
            return  # Neither fact belongs to this agent
    now = time.time()
    conn.execute(
        "INSERT INTO fact_links (fact_a, fact_b, relation, strength, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (fact_a, fact_b, relation, strength, now)
    )
    conn.commit()


def link_by_text(conn, agent_id, fact_id, search_text):
    """Link a fact to existing facts matching search_text."""
    matches = recall(conn, agent_id, search_text, limit=3, include_shared=False)
    for m in matches:
        if m["id"] != fact_id:
            link(conn, fact_id, m["id"])


def associations(conn, fact_id, depth=1):
    """Get all facts associated with a given fact."""
    seen = {fact_id}
    current = {fact_id}
    results = []

    for _ in range(depth):
        if not current:
            break
        placeholders = ",".join("?" * len(current))
        rows = conn.execute(f"""
            SELECT fl.*, f.fact, f.category
            FROM fact_links fl
            JOIN facts f ON (f.id = CASE WHEN fl.fact_a IN ({placeholders})
                            THEN fl.fact_b ELSE fl.fact_a END)
            WHERE (fl.fact_a IN ({placeholders}) OR fl.fact_b IN ({placeholders}))
              AND f.active = 1
        """, list(current) * 3).fetchall()

        next_level = set()
        for row in rows:
            other = row["fact_b"] if row["fact_a"] in seen else row["fact_a"]
            if other not in seen:
                seen.add(other)
                next_level.add(other)
                results.append({
                    "id": other,
                    "fact": row["fact"],
                    "category": row["category"],
                    "relation": row["relation"],
                    "strength": row["strength"],
                })
        current = next_level

    return results


def all_facts(conn, agent_id, include_inactive=False, limit=None, offset=0):
    """List all facts for an agent."""
    sql = "SELECT * FROM facts WHERE agent_id = ?"
    params = [agent_id]
    if not include_inactive:
        sql += " AND active = 1"
    sql += " ORDER BY updated_at DESC"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    return [dict(r) for r in conn.execute(sql, params).fetchall()]
