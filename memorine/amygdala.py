"""
Memorine amygdala — emotional weight, decay, and reinforcement.
Memories that matter stick. The rest fades away.
"""

import math
import time


def decay_factor(last_accessed, access_count, now=None):
    """Ebbinghaus forgetting curve with reinforcement.

    More access = more stable memory.
    Untouched memories fade over days.
    Heavily used memories last months.
    """
    now = now or time.time()
    days = max((now - last_accessed) / 86400, 0)
    # stability grows with repeated access (caps at ~30 day half-life)
    stability = 1 + min(access_count, 20) * 1.5
    retention = math.exp(-days / stability)
    return round(max(retention, 0.01), 4)


def effective_weight(fact_row, now=None):
    """Combine base weight, confidence, and decay into a single score."""
    decay = decay_factor(fact_row["last_accessed"], fact_row["access_count"], now)
    return round(fact_row["weight"] * fact_row["confidence"] * decay, 4)


def importance_from_error(is_error):
    """Errors get high emotional weight — pain sticks."""
    return 2.5 if is_error else 1.0


def reinforce(conn, fact_id, boost=0.1):
    """Accessing a memory reinforces it — like rehearsal."""
    now = time.time()
    conn.execute("""
        UPDATE facts SET
            last_accessed = ?,
            access_count = access_count + 1,
            weight = MIN(weight + ?, 5.0)
        WHERE id = ?
    """, (now, boost, fact_id))
    conn.commit()


def weaken(conn, fact_id, penalty=0.2):
    """Contradicted or wrong memories get weakened."""
    conn.execute("""
        UPDATE facts SET
            weight = MAX(weight - ?, 0.1),
            confidence = MAX(confidence - ?, 0.1)
        WHERE id = ?
    """, (penalty, penalty, fact_id))
    conn.commit()


def cleanup_faded(conn, agent_id=None, threshold=0.05, batch_size=500):
    """Deactivate memories that have faded below threshold.

    Processes in batches to avoid loading the entire table into memory.
    If agent_id is given, only cleans that agent's facts.
    """
    now = time.time()
    deactivated = 0
    offset = 0

    while True:
        sql = ("SELECT id, last_accessed, access_count, weight, confidence "
               "FROM facts WHERE active = 1")
        params = []
        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        sql += " LIMIT ? OFFSET ?"
        params.extend([batch_size, offset])

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            break

        batch_deactivated = 0
        for row in rows:
            ew = effective_weight(row, now)
            if ew < threshold:
                conn.execute(
                    "UPDATE facts SET active = 0 WHERE id = ?", (row["id"],)
                )
                batch_deactivated += 1

        if batch_deactivated:
            conn.commit()
        deactivated += batch_deactivated

        if len(rows) < batch_size:
            break
        offset += batch_size

    return deactivated
