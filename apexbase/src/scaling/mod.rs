//! Horizontal Scaling Framework
//!
//! Provides abstractions for distributing data across multiple nodes/shards.
//! Designed as a pluggable framework that can be implemented with different
//! backends (local filesystem, network, cloud storage).
//!
//! Architecture:
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                    ShardRouter                            │
//! │  - Routes queries/writes to correct shard                │
//! │  - Handles cross-shard queries (scatter-gather)          │
//! │  - Manages shard topology changes                        │
//! ├──────────────────────────────────────────────────────────┤
//! │  ShardManager                                            │
//! │  - Manages shard lifecycle (create/split/merge/migrate)  │
//! │  - Monitors shard health and rebalancing                 │
//! │  - Coordinates shard metadata                            │
//! ├──────────────────────────────────────────────────────────┤
//! │  PartitionStrategy                                       │
//! │  - Hash partitioning (uniform distribution)              │
//! │  - Range partitioning (ordered data)                     │
//! │  - Custom partitioning (user-defined)                    │
//! ├──────────────────────────────────────────────────────────┤
//! │  NodeManager                                             │
//! │  - Tracks cluster membership                             │
//! │  - Health checking                                       │
//! │  - Leader election (future)                              │
//! └──────────────────────────────────────────────────────────┘
//! ```

pub mod partition;
pub mod shard;
pub mod router;
pub mod node;

pub use partition::{PartitionStrategy, PartitionKey, HashPartitioner, RangePartitioner};
pub use shard::{ShardManager, ShardId, ShardMeta, ShardStatus};
pub use router::{ShardRouter, RoutingDecision};
pub use node::{NodeManager, NodeId, NodeInfo, NodeStatus};
