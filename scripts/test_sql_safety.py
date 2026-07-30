"""Quick check: sql_safety accepts real queries and rejects dangerous ones."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rootai.guardrails.sql_safety import check_sql_safety


CASES = [
    # (sql, expected_pass, description)
    ("SELECT customer_state, COUNT(*) FROM order_items_denorm GROUP BY 1", True, "basic SELECT"),
    ("WITH agg AS (SELECT customer_state, SUM(price) AS rev FROM order_items_denorm GROUP BY 1) SELECT * FROM agg", True, "CTE"),
    ("DROP TABLE order_items_denorm", False, "DROP"),
    ("DELETE FROM order_items_denorm WHERE 1=1", False, "DELETE"),
    ("SELECT * FROM raw_customers", False, "disallowed table"),
    ("", False, "empty"),
    ("INSERT INTO x VALUES (1)", False, "INSERT"),
    ("UPDATE order_items_denorm SET price = 0", False, "UPDATE"),
]

failed = 0
for sql, expected_pass, desc in CASES:
    result = check_sql_safety(sql)
    ok = (result.passed == expected_pass)
    marker = "PASS" if ok else "FAIL"
    print(f"[{marker}] {desc}: passed={result.passed}, reason={result.reason}")
    if not ok:
        failed += 1

print(f"\n{len(CASES) - failed}/{len(CASES)} checks passed")