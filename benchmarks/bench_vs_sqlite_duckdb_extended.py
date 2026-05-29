"""
Extended ApexBase benchmark diagnostics.

The default bench_vs_sqlite_duckdb.py entrypoint stays aligned with the README
public scoreboard. This wrapper runs the full diagnostic profile: file-format
extras, materialization APIs, Q/s, OLTP microbenchmarks, durable/transaction
sections, and all vector metrics.
"""

from bench_vs_sqlite_duckdb import PROFILE_EXTENDED, main


if __name__ == "__main__":
    main(default_profile=PROFILE_EXTENDED)
