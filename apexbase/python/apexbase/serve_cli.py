"""
ApexBase Combined Server Launcher

Starts both the PostgreSQL Wire server and Arrow Flight gRPC server simultaneously.

Usage:
    apexbase-serve --dir /path/to/data

    apexbase-serve --dir /path/to/data \\
        --pg-port 5432 --flight-port 50051

    # Disable one server:
    apexbase-serve --dir /path/to/data --no-flight
    apexbase-serve --dir /path/to/data --no-pg

Clients:
    PG Wire:  psycopg2.connect(host="127.0.0.1", port=5432, ...)
    Flight:   pyarrow.flight.connect("grpc://127.0.0.1:50051")
"""

import argparse
import sys
import threading
import signal
import time


def _start_pg(data_dir, host, port, errors):
    try:
        from apexbase._core import start_pg_server
        start_pg_server(data_dir, host, port)
    except Exception as e:
        errors.append(("PG Wire", e))


def _start_flight(data_dir, host, port, errors):
    try:
        from apexbase._core import start_flight_server
        start_flight_server(data_dir, host, port)
    except Exception as e:
        errors.append(("Arrow Flight", e))


def main():
    parser = argparse.ArgumentParser(
        prog="apexbase-serve",
        description="ApexBase combined server: PostgreSQL Wire + Arrow Flight gRPC",
    )
    parser.add_argument(
        "--dir", "-d",
        default=".",
        help="Directory containing ApexBase database files (default: current directory)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to for both servers (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--pg-port",
        type=int,
        default=5432,
        help="PostgreSQL Wire port (default: 5432)",
    )
    parser.add_argument(
        "--flight-port",
        type=int,
        default=50051,
        help="Arrow Flight gRPC port (default: 50051)",
    )
    parser.add_argument(
        "--no-pg",
        action="store_true",
        help="Disable PostgreSQL Wire server",
    )
    parser.add_argument(
        "--no-flight",
        action="store_true",
        help="Disable Arrow Flight server",
    )

    args = parser.parse_args()

    if args.no_pg and args.no_flight:
        print("Error: both --no-pg and --no-flight specified, nothing to start.", file=sys.stderr)
        sys.exit(1)

    errors = []
    threads = []

    print("ApexBase Server")
    print(f"  Data dir: {args.dir}")

    if not args.no_pg:
        try:
            from apexbase._core import start_pg_server
        except ImportError:
            print(
                "Error: PG Wire server not available (requires 'server' feature).",
                file=sys.stderr,
            )
            sys.exit(1)

        t_pg = threading.Thread(
            target=_start_pg,
            args=(args.dir, args.host, args.pg_port, errors),
            daemon=True,
        )
        threads.append(("PG Wire", t_pg))
        print(f"  PG Wire:      postgresql://{args.host}:{args.pg_port}")

    if not args.no_flight:
        try:
            from apexbase._core import start_flight_server
        except ImportError:
            print(
                "Error: Arrow Flight server not available (requires 'flight' feature).",
                file=sys.stderr,
            )
            sys.exit(1)

        t_fl = threading.Thread(
            target=_start_flight,
            args=(args.dir, args.host, args.flight_port, errors),
            daemon=True,
        )
        threads.append(("Arrow Flight", t_fl))
        print(f"  Arrow Flight: grpc://{args.host}:{args.flight_port}")

    print("  Press Ctrl-C to stop all servers.\n")

    for _, t in threads:
        t.start()

    # Give servers a moment to start, then check for early failures
    time.sleep(0.5)
    if errors:
        for name, e in errors:
            print(f"  ERROR [{name}]: {e}", file=sys.stderr)
        sys.exit(1)

    # Block until Ctrl-C or a server thread dies unexpectedly
    try:
        while True:
            alive = [t.is_alive() for _, t in threads]
            if not any(alive):
                break
            if errors:
                for name, e in errors:
                    print(f"\nERROR [{name}]: {e}", file=sys.stderr)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping all servers...")

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
