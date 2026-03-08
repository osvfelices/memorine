"""
Memorine embeddings — optional semantic search.
Only loaded when fastembed and sqlite-vec are installed.
pip install memorine[embeddings]
"""

_AVAILABLE = False
_EMBEDDER = None

try:
    from fastembed import TextEmbedding
    import sqlite_vec
    _AVAILABLE = True
except ImportError:
    pass


def is_available():
    return _AVAILABLE


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None and _AVAILABLE:
        _EMBEDDER = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _EMBEDDER


def init_vec_schema(conn):
    """Create the vector storage table. Safe to call multiple times."""
    if not _AVAILABLE:
        return
    sqlite_vec.load(conn)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fact_embeddings
        USING vec0(fact_id INTEGER PRIMARY KEY, embedding FLOAT[384])
    """)
    conn.commit()


def embed_fact(conn, fact_id, text):
    """Generate and store embedding for a single fact."""
    embedder = _get_embedder()
    if not embedder:
        return
    vectors = list(embedder.embed([text]))
    if not vectors:
        return
    vec = vectors[0].tolist()
    # Upsert: delete existing then insert
    conn.execute("DELETE FROM fact_embeddings WHERE fact_id = ?", (fact_id,))
    conn.execute(
        "INSERT INTO fact_embeddings (fact_id, embedding) VALUES (?, ?)",
        (fact_id, _serialize_vec(vec))
    )
    conn.commit()


def embed_facts_batch(conn, facts_list):
    """Batch-embed multiple facts at once. Much faster than one-by-one.

    facts_list: list of (fact_id, text) tuples.
    """
    embedder = _get_embedder()
    if not embedder or not facts_list:
        return
    ids, texts = zip(*facts_list)
    vectors = list(embedder.embed(list(texts)))

    for fact_id, vec in zip(ids, vectors):
        vec_list = vec.tolist()
        conn.execute("DELETE FROM fact_embeddings WHERE fact_id = ?", (fact_id,))
        conn.execute(
            "INSERT INTO fact_embeddings (fact_id, embedding) VALUES (?, ?)",
            (fact_id, _serialize_vec(vec_list))
        )
    conn.commit()


def semantic_search(conn, query_text, agent_id, limit=10, include_shared=True):
    """Find facts by meaning, not just keywords.

    Returns list of fact dicts with a semantic_score field (0-1, higher = better).
    """
    embedder = _get_embedder()
    if not embedder:
        return []

    query_vec = list(embedder.embed([query_text]))
    if not query_vec:
        return []

    vec = query_vec[0].tolist()

    # KNN search against all embeddings (sqlite-vec requires k = ?)
    rows = conn.execute("""
        SELECT fact_id, distance
        FROM fact_embeddings
        WHERE embedding MATCH ?
        AND k = ?
    """, (_serialize_vec(vec), limit * 3)).fetchall()

    if not rows:
        return []

    # Fetch full fact rows and filter by agent
    fact_ids = [r["fact_id"] for r in rows]
    distances = {r["fact_id"]: r["distance"] for r in rows}

    placeholders = ",".join("?" * len(fact_ids))
    if include_shared:
        sql = f"""
            SELECT * FROM facts
            WHERE id IN ({placeholders}) AND active = 1
            AND (agent_id = ? OR id IN (
                SELECT fact_id FROM shared WHERE to_agent = ? OR to_agent IS NULL
            ))
        """
        params = fact_ids + [agent_id, agent_id]
    else:
        sql = f"""
            SELECT * FROM facts
            WHERE id IN ({placeholders}) AND active = 1 AND agent_id = ?
        """
        params = fact_ids + [agent_id]

    fact_rows = conn.execute(sql, params).fetchall()

    results = []
    for row in fact_rows:
        d = dict(row)
        # Convert L2 distance to cosine similarity (0-1, higher = better)
        # For normalized embeddings: cosine_sim = 1 - (L2_dist^2 / 2)
        dist = distances.get(row["id"], 2.0)
        d["semantic_score"] = max(0.0, 1.0 - (dist * dist / 2.0))
        results.append(d)

    results.sort(key=lambda x: x["semantic_score"], reverse=True)
    return results[:limit]


def reindex_all(conn, agent_id):
    """Rebuild embeddings for all active facts. Useful after first install."""
    all_facts = conn.execute(
        "SELECT id, fact FROM facts WHERE agent_id = ? AND active = 1",
        (agent_id,)
    ).fetchall()

    if not all_facts:
        return 0

    facts_list = [(r["id"], r["fact"]) for r in all_facts]
    embed_facts_batch(conn, facts_list)
    return len(facts_list)


def _serialize_vec(vec):
    """Serialize a float list for sqlite-vec."""
    import struct
    return struct.pack(f"{len(vec)}f", *vec)
