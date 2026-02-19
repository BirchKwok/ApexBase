"""
ApexBase Arrow Flight gRPC Server CLI

Usage:
    apexbase-flight --dir /path/to/data --port 50051

Python client:
    import pyarrow.flight as fl
    client = fl.connect("grpc://127.0.0.1:50051")
    df = client.do_get(fl.Ticket(b"SELECT * FROM t")).read_all().to_pandas()
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="apexbase-flight",
        description="ApexBase Arrow Flight gRPC server — zero-copy columnar data transfer",
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
        default=50051,
        help="Port to listen on (default: 50051)",
    )

    args = parser.parse_args()

    try:
        from apexbase._core import start_flight_server
    except ImportError:
        print(
            "Error: apexbase-flight requires the 'flight' feature.\n"
            "Build from source:\n"
            "  maturin develop --release\n"
            "Or:\n"
            "  cargo build --release --bin apexbase-flight --no-default-features --features flight",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"ApexBase Arrow Flight Server")
    print(f"  Listening on: grpc://{args.host}:{args.port}")
    print(f"  Data dir:     {args.dir}")
    print(f"  Python:       import pyarrow.flight as fl; fl.connect('grpc://{args.host}:{args.port}')")
    print(f"  Press Ctrl-C to stop.\n")

    try:
        start_flight_server(args.dir, args.host, args.port)
    except KeyboardInterrupt:
        print("\nFlight server stopped.")
    except Exception as e:
        print(f"Flight server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
