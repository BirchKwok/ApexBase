//! ApexBase PostgreSQL-compatible Server
//!
//! Usage:
//!   apexbase-server --dir /path/to/data --port 5432
//!
//! Then connect with DBeaver, psql, or any PostgreSQL client.

use clap::Parser;
use std::path::PathBuf;

use apexbase::server::{self, ServerConfig};

#[derive(Parser, Debug)]
#[command(name = "apexbase-server")]
#[command(about = "ApexBase PostgreSQL-compatible wire protocol server")]
#[command(version)]
struct Args {
    /// Directory containing ApexBase database files
    #[arg(short, long, default_value = ".")]
    dir: PathBuf,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 5432)]
    port: u16,
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let args = Args::parse();

    let config = ServerConfig {
        data_dir: args.dir,
        host: args.host,
        port: args.port,
    };

    if let Err(e) = server::start_server(config).await {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
}
