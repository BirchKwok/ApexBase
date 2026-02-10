"""
ApexBase PostgreSQL-compatible Server CLI

Usage:
    apexbase-server --dir /path/to/data --port 5432

Then connect with DBeaver, psql, or any PostgreSQL client.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="apexbase-server",
        description="ApexBase PostgreSQL-compatible wire protocol server",
    )
    parser.add_argument(
        "--dir", "-d",
        default=".",
        help="Directory containing ApexBase database files (default: current directory)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5432,
        help="Port to listen on (default: 5432)",
    )

    args = parser.parse_args()

    try:
        from apexbase._core import start_pg_server
    except ImportError:
        print(
            "Error: apexbase-server requires the 'server' feature.\n"
            "If you installed via pip, please reinstall with server support:\n"
            "  pip install apexbase[server]\n"
            "Or build from source:\n"
            "  cargo build --release --bin apexbase-server --no-default-features --features server",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        start_pg_server(args.dir, args.host, args.port)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
