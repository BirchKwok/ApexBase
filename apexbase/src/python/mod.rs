//! Python bindings via PyO3
//!
//! ApexStorage is the storage implementation using on-demand reading.

mod bindings;
#[cfg(feature = "server")]
mod server_bridge;
#[cfg(feature = "flight")]
mod flight_bridge;

pub use bindings::ApexStorageImpl as ApexStorage;
#[cfg(feature = "server")]
pub use server_bridge::start_pg_server;
#[cfg(feature = "flight")]
pub use flight_bridge::start_flight_server;
