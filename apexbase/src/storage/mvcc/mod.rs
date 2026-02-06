//! MVCC (Multi-Version Concurrency Control) subsystem
//!
//! Enables concurrent reads and writes without blocking:
//! - Readers see a consistent snapshot at their start time
//! - Writers create new versions without disturbing readers
//! - Garbage collection removes old versions no longer visible
//!
//! Architecture:
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │              VersionStore                         │
//! │  - Manages version chains per row                │
//! │  - Each version: (begin_ts, end_ts, data)        │
//! ├──────────────────────────────────────────────────┤
//! │  SnapshotManager                                 │
//! │  - Tracks active snapshots                       │
//! │  - Provides visibility check                     │
//! ├──────────────────────────────────────────────────┤
//! │  GarbageCollector                                │
//! │  - Removes versions invisible to all snapshots   │
//! │  - Reclaims storage space                        │
//! └──────────────────────────────────────────────────┘
//! ```

pub mod version_store;
pub mod snapshot;
pub mod gc;

pub use version_store::{VersionStore, RowVersion, VersionChain};
pub use snapshot::{SnapshotManager, Snapshot, SnapshotId};
pub use gc::GarbageCollector;
