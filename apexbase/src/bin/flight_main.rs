//! ApexBase Arrow Flight gRPC Server
//!
//! Usage:
//!   apexbase-flight --dir /path/to/data --port 50051
//!
//! Python client:
//!   import pyarrow.flight as flight
//!   c = flight.connect("grpc://127.0.0.1:50051")
//!   df = c.do_get(flight.Ticket(b"SELECT * FROM t")).read_pandas()

use clap::Parser;
use std::path::PathBuf;

use apexbase::flight::{self, FlightConfig};

#[derive(Parser, Debug)]
#[command(name = "apexbase-flight")]
#[command(about = "ApexBase Arrow Flight gRPC server — zero-copy columnar data transfer")]
#[command(version)]
struct Args {
    /// Directory containing ApexBase database files
    #[arg(short, long, default_value = ".")]
    dir: PathBuf,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on (default 50051 for gRPC)
    #[arg(short, long, default_value_t = 50051)]
    port: u16,
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let args = Args::parse();

    let config = FlightConfig {
        data_dir: args.dir,
        host: args.host,
        port: args.port,
    };

    if let Err(e) = flight::start_flight_server(config).await {
        eprintln!("Flight server error: {}", e);
        std::process::exit(1);
    }
}
