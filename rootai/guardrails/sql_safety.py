"""
rootai/guardrails/sql_safety.py

Pattern-based safety check for LLM-generated SQL. Runs before every
query execution. The read-only DuckDB connection in rootai.tools.db is
the primary defense (it can't write regardless of what the SQL says).
This layer adds explicit rejection with a clear reason string that
lands in the trace, so an interviewer can see the agent's safety
posture at a glance.

Not a substitute for the connection-level lock. Belt and suspenders.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Forbidden SQL keywords. Match on word boundary so 'INSERT' matches
# but 'INSERTION' in a column name doesn't. Case-insensitive.
_FORBIDDEN_KEYWORDS = (
    "DROP",
    "DELETE",
    "INSERT",
    "UPDATE",
    "TRUNCATE",
    "ALTER",
    "CREATE",
    "REPLACE",
    "GRANT",
    "REVOKE",
    "ATTACH",
    "DETACH",
    "COPY",
    "EXPORT",
    "IMPORT",
    "PRAGMA",
)

# The only view the agent should query. If the LLM tries to hit raw_*
# tables directly, reject and force it to use the denorm.
_ALLOWED_TABLES = {"order_items_denorm"}


@dataclass
class SafetyCheck:
    """Result of running the safety check. JSON-serializable."""
    passed: bool
    reason: str | None = None


def check_sql_safety(sql: str) -> SafetyCheck:
    """
    Return SafetyCheck(passed=True) if the SQL is safe to run, else
    SafetyCheck(passed=False, reason='...').

    Checks:
    1. Query starts with SELECT or WITH (SELECT).
    2. No forbidden keywords appear as standalone words.
    3. Only allowed tables are referenced (best-effort regex match on FROM/JOIN).
    """
    if not sql or not sql.strip():
        return SafetyCheck(passed=False, reason="empty SQL")

    normalized = sql.strip().upper()

    # 1. Must start with SELECT or WITH
    if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
        return SafetyCheck(
            passed=False,
            reason=f"SQL must start with SELECT or WITH, got: {normalized[:20]}",
        )

    # 2. No forbidden keywords as standalone tokens
    for kw in _FORBIDDEN_KEYWORDS:
        pattern = r"\b" + kw + r"\b"
        if re.search(pattern, normalized):
            return SafetyCheck(
                passed=False,
                reason=f"forbidden keyword: {kw}",
            )

    # 3. Only allowed tables. Regex captures the token after FROM or JOIN.
    # This is best-effort; it can miss table names in CTEs, but the
    # read-only connection catches anything this misses.
    table_refs = re.findall(r"\b(?:FROM|JOIN)\s+([A-Z_][A-Z0-9_]*)", normalized)
    referenced = {t.lower() for t in table_refs}
    unknown_tables = referenced - _ALLOWED_TABLES

    # Allow CTE-defined temporary names: they appear in table_refs but are
    # not real tables. We detect CTEs by looking for 'WITH <name> AS'
    # patterns, and remove those names from unknown_tables.
    cte_names = {n.lower() for n in re.findall(r"\b([A-Z_][A-Z0-9_]*)\s+AS\s*\(", normalized)}
    unknown_tables -= cte_names

    if unknown_tables:
        return SafetyCheck(
            passed=False,
            reason=f"references disallowed table(s): {sorted(unknown_tables)}. Only {sorted(_ALLOWED_TABLES)} is queryable.",
        )

    return SafetyCheck(passed=True)