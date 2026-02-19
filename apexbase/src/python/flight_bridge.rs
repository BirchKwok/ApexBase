//! PyO3 bridge for the Arrow Flight gRPC server.
//!
//! Exposes `start_flight_server` as a Python function so `pip install apexbase`
//! users can launch the Flight server without a separate Rust binary.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::path::PathBuf;

use crate::flight::{self, FlightConfig};

/// Start the ApexBase Arrow Flight gRPC server (blocking).
///
/// This function blocks until the server is stopped (Ctrl-C).
///
/// Args:
///     data_dir (str): Directory containing ApexBase database files.
///     host (str): Host to bind to. Default "127.0.0.1".
///     port (int): Port to listen on. Default 50051.
#[pyfunction]
#[pyo3(signature = (data_dir, host = "127.0.0.1".to_string(), port = 50051))]
pub fn start_flight_server(py: Python<'_>, data_dir: String, host: String, port: u16) -> PyResult<()> {
    let config = FlightConfig {
        data_dir: PathBuf::from(data_dir),
        host,
        port,
    };

    py.allow_threads(|| {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tokio runtime: {}", e)))?;

        rt.block_on(async {
            flight::start_flight_server(config).await
                .map_err(|e| PyRuntimeError::new_err(format!("Flight server error: {}", e)))
        })
    })
}
