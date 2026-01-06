import os
import sys
import time
import random

# Add the apexbase python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'apexbase', 'python'))

from apexbase import ApexClient


def _t(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    print(f"{label}: {(t1 - t0):.6f}s")
    return out


def main():
    dirpath = "./test_db_1m"
    client = ApexClient(dirpath=dirpath, drop_if_exists=False)

    # Ensure table exists
    client.use_table("default")

    # Build 1,000,000 rows like user's scenario (but do NOT store 100 times; that is 100M rows)
    def gen_rows(n: int):
        rows = []
        for i in range(n):
            rows.append({
                "title": f"Python编程指南第{i+1}部分",
                "content": f"学习Python的最佳实践 - 第{i+1}章节，包含详细的编程技巧和实例代码。",
                "number": random.randint(0, 10000),
                "col1": f"p{(i % 1000)}",
                "col2": "test_x" if (i % 10 == 0) else "nope",
                "col3": f"v{i}",
                "col4": i % 100,
            })
        return rows

    if client.count_rows() < 1_000_000:
        rows = _t("generate_1m_rows", lambda: gen_rows(1_000_000))
        _t("store_1m_rows", lambda: client.store(rows))
        _t("flush", lambda: client.flush())
    else:
        print("DB already has >= 1,000,000 rows; skipping load")

    # Warm-up
    _t("warmup_select_1", lambda: client.execute("SELECT number FROM default LIMIT 10").to_dict())

    # Filters
    _t("where_single", lambda: client.query("number > 9000", limit=1000).to_dict())
    _t("where_and", lambda: client.execute("SELECT title, number FROM default WHERE number > 9000 AND number < 9500 LIMIT 1000").to_dict())
    _t("where_or", lambda: client.execute("SELECT title, number FROM default WHERE number > 9990 OR number < 10 LIMIT 1000").to_dict())

    # LIKE
    _t("like_prefix", lambda: client.execute("SELECT title FROM default WHERE title LIKE 'Python编程指南第1%' LIMIT 1000").to_dict())

    # REGEXP + window function
    _t(
        "regexp_window",
        lambda: client.execute(
            "SELECT col3, col1, col4, row_number() OVER (PARTITION BY col1 ORDER BY col4) AS rn "
            "FROM default WHERE col2 REGEXP \"test*\" LIMIT 1000"
        ).to_dict(),
    )

    # GROUP BY
    _t(
        "group_by",
        lambda: client.execute(
            "SELECT col1, COUNT(*) AS c, AVG(number) AS avg_num FROM default GROUP BY col1"
        ).to_dict(),
    )

    client.close()


if __name__ == "__main__":
    main()
