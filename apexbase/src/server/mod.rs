//! PostgreSQL Wire Protocol Server for ApexBase
//!
//! Enables DBeaver and other PostgreSQL-compatible clients to connect to ApexBase
//! using standard PostgreSQL wire protocol.

mod handler;
mod pg_catalog;
mod types;

pub use handler::ApexBaseHandler;

use std::path::PathBuf;
use std::sync::Arc;

use pgwire::api::PgWireHandlerFactory;
use pgwire::tokio::process_socket;
use tokio::net::TcpListener;

/// Server configuration
pub struct ServerConfig {
    /// Directory containing ApexBase database files
    pub data_dir: PathBuf,
    /// Host to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("."),
            host: "127.0.0.1".to_string(),
            port: 5432,
        }
    }
}

/// Server handler factory
pub struct ApexBaseServerFactory {
    handler: Arc<ApexBaseHandler>,
}

impl PgWireHandlerFactory for ApexBaseServerFactory {
    type StartupHandler = ApexBaseHandler;
    type SimpleQueryHandler = ApexBaseHandler;
    type ExtendedQueryHandler = ApexBaseHandler;
    type CopyHandler = ApexBaseHandler;

    fn simple_query_handler(&self) -> Arc<Self::SimpleQueryHandler> {
        self.handler.clone()
    }

    fn extended_query_handler(&self) -> Arc<Self::ExtendedQueryHandler> {
        self.handler.clone()
    }

    fn startup_handler(&self) -> Arc<Self::StartupHandler> {
        self.handler.clone()
    }

    fn copy_handler(&self) -> Arc<Self::CopyHandler> {
        self.handler.clone()
    }
}

/// Start the PostgreSQL wire protocol server
pub async fn start_server(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    let handler = Arc::new(ApexBaseHandler::new(config.data_dir.clone()));
    let factory = Arc::new(ApexBaseServerFactory {
        handler,
    });

    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr).await?;

    log::info!("ApexBase server listening on {}", addr);
    log::info!("Data directory: {}", config.data_dir.display());
    println!("ApexBase PostgreSQL-compatible server");
    println!("  Listening on: {}", addr);
    println!("  Data dir:     {}", config.data_dir.display());
    println!("  Connect with: psql -h {} -p {}", config.host, config.port);

    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok(incoming) => {
                        let factory_ref = factory.clone();
                        tokio::spawn(async move {
                            if let Err(e) = process_socket(incoming.0, None, factory_ref).await {
                                log::error!("Connection error: {:?}", e);
                            }
                        });
                    }
                    Err(e) => {
                        log::error!("Accept error: {:?}", e);
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nShutting down ApexBase server...");
                break;
            }
        }
    }

    Ok(())
}
