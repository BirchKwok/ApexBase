//! Observable outcome of a failed storage commit. This does not strengthen
//! the table's durability mode or promise cross-table crash atomicity.

use std::{fmt, io};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitOutcome {
    /// No commit marker or data application has been attempted.
    NotCommitted,
    /// A marker/data write may have taken effect, or the transaction ID is no
    /// longer active and its prior outcome is unavailable. Reconcile first.
    Unknown,
    /// Data, indexes and transaction publication completed; maintenance failed.
    Committed,
}

impl CommitOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotCommitted => "not_committed",
            Self::Unknown => "unknown",
            Self::Committed => "committed",
        }
    }
}

/// Payload carried by `io::Error` from `Session::commit_txn`.
/// Rust callers can downcast `error.get_ref()`; Python retains RuntimeError
/// with the same stable `commit_outcome=...` marker and original cause text.
#[derive(Debug)]
pub struct CommitError {
    pub txn_id: u64,
    pub outcome: CommitOutcome,
    cause: io::Error,
}

impl CommitError {
    pub(crate) fn wrap(txn_id: u64, outcome: CommitOutcome, cause: io::Error) -> io::Error {
        io::Error::new(
            cause.kind(),
            Self {
                txn_id,
                outcome,
                cause,
            },
        )
    }
}

impl fmt::Display for CommitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let advice = match self.outcome {
            CommitOutcome::NotCommitted => {
                "retry requires a new transaction and resolving the cause"
            }
            CommitOutcome::Unknown => {
                "do not replay DML; reopen and reconcile the transaction outcome"
            }
            CommitOutcome::Committed => {
                "do not replay DML; logical commit completed, maintenance failed"
            }
        };
        write!(
            f,
            "commit_outcome={} txn_id={}: {}; {}",
            self.outcome.as_str(),
            self.txn_id,
            self.cause,
            advice
        )
    }
}

impl std::error::Error for CommitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn commit_error_preserves_kind_source_and_outcome() {
        for outcome in [
            CommitOutcome::NotCommitted,
            CommitOutcome::Unknown,
            CommitOutcome::Committed,
        ] {
            let cause = io::Error::from_raw_os_error(13);
            let kind = cause.kind();
            let error = CommitError::wrap(42, outcome, cause);
            assert_eq!(error.kind(), kind);
            let detail = error
                .get_ref()
                .unwrap()
                .downcast_ref::<CommitError>()
                .unwrap();
            assert_eq!(detail.txn_id, 42);
            assert_eq!(detail.outcome, outcome);
            assert_eq!(
                detail
                    .source()
                    .unwrap()
                    .downcast_ref::<io::Error>()
                    .unwrap()
                    .raw_os_error(),
                Some(13)
            );
            assert!(error
                .to_string()
                .contains(&format!("commit_outcome={}", outcome.as_str())));
        }
    }
}
