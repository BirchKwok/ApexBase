//! Vector distance operations for ANN / TopK queries.
//!
//! Supports:
//!   - L2 (Euclidean) distance
//!   - Cosine distance / similarity
//!   - Inner product (dot product)
//!   - L1 (Manhattan) distance
//!   - L∞ (Chebyshev) distance
//!
//! Vectors are stored as **Binary** columns (raw little-endian f32 bytes).
//! A `dim`-dimensional vector occupies exactly `dim * 4` bytes.
//!
//! The batch compute functions use Rayon for row-level parallelism; the
//! per-vector kernels are written to enable LLVM AVX-2 auto-vectorisation
//! (8×f32 per register) at opt-level=3.

use std::io;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryArray, Float64Array, Float32Array, StringArray};
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Reinterpret a byte slice as an f32 slice (zero-copy).
///
/// # Safety
/// Requires `bytes.len() % 4 == 0` and the bytes to be valid f32 LE values.
#[inline(always)]
unsafe fn bytes_to_f32(bytes: &[u8]) -> &[f32] {
    std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
}

/// Parse query-vector bytes (either raw f32 LE or raw f64 LE) to a `Vec<f32>`.
pub fn bytes_to_query_vec_f32(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 == 0 {
        let floats = unsafe { bytes_to_f32(bytes) };
        Some(floats.to_vec())
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core scalar kernels  (LLVM will auto-vectorise with opt-level=3)
// ─────────────────────────────────────────────────────────────────────────────

/// L2 squared distance (Σ(aᵢ−bᵢ)²).  No sqrt → safe for ordering.
#[inline(always)]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    // 8-element unroll for AVX2 (256-bit / 32-bit = 8 lanes)
    let c = n / 8;
    for i in 0..c {
        let base = i << 3;
        let d0 = a[base]   - b[base];
        let d1 = a[base+1] - b[base+1];
        let d2 = a[base+2] - b[base+2];
        let d3 = a[base+3] - b[base+3];
        let d4 = a[base+4] - b[base+4];
        let d5 = a[base+5] - b[base+5];
        let d6 = a[base+6] - b[base+6];
        let d7 = a[base+7] - b[base+7];
        s += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7;
    }
    for i in (c*8)..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// L2 (Euclidean) distance √(Σ(aᵢ−bᵢ)²).
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// L1 (Manhattan) distance Σ|aᵢ−bᵢ|.
#[inline(always)]
pub fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    let c = n / 8;
    for i in 0..c {
        let base = i << 3;
        s += (a[base]   - b[base]).abs()
           + (a[base+1] - b[base+1]).abs()
           + (a[base+2] - b[base+2]).abs()
           + (a[base+3] - b[base+3]).abs()
           + (a[base+4] - b[base+4]).abs()
           + (a[base+5] - b[base+5]).abs()
           + (a[base+6] - b[base+6]).abs()
           + (a[base+7] - b[base+7]).abs();
    }
    for i in (c*8)..n {
        s += (a[i] - b[i]).abs();
    }
    s
}

/// L∞ (Chebyshev) distance max|aᵢ−bᵢ|.
#[inline(always)]
pub fn linf_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut m = 0.0f32;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if d > m { m = d; }
    }
    m
}

/// Dot product Σ aᵢ·bᵢ.
#[inline(always)]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    let c = n / 8;
    for i in 0..c {
        let base = i << 3;
        s += a[base]*b[base] + a[base+1]*b[base+1] + a[base+2]*b[base+2] + a[base+3]*b[base+3]
           + a[base+4]*b[base+4] + a[base+5]*b[base+5] + a[base+6]*b[base+6] + a[base+7]*b[base+7];
    }
    for i in (c*8)..n {
        s += a[i] * b[i];
    }
    s
}

/// Cosine similarity cos(a,b) = dot(a,b) / (‖a‖·‖b‖).
/// Returns 0.0 if either vector is zero.
#[inline(always)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = inner_product(a, b);
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Cosine distance = 1 − cosine_similarity.
#[inline(always)]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch (Arrow array) distance computation
// ─────────────────────────────────────────────────────────────────────────────

/// Distance metric selector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    L2,
    L2Squared,
    L1,
    LInf,
    InnerProduct,
    NegInnerProduct,
    CosineSimilarity,
    CosineDistance,
}

impl DistanceMetric {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "L2" | "EUCLIDEAN" | "ARRAY_DISTANCE" | "L2_DISTANCE" => Some(Self::L2),
            "L2_SQUARED" | "SQUARED_L2" => Some(Self::L2Squared),
            "L1" | "MANHATTAN" | "L1_DISTANCE" => Some(Self::L1),
            "LINF" | "CHEBYSHEV" | "LINF_DISTANCE" => Some(Self::LInf),
            "DOT" | "INNER_PRODUCT" | "ARRAY_INNER_PRODUCT" => Some(Self::InnerProduct),
            "NEG_INNER_PRODUCT" | "NEGATIVE_INNER_PRODUCT" | "ARRAY_NEGATIVE_INNER_PRODUCT" => Some(Self::NegInnerProduct),
            "COSINE" | "COSINE_SIMILARITY" | "ARRAY_COSINE_SIMILARITY" => Some(Self::CosineSimilarity),
            "COSINE_DISTANCE" | "ARRAY_COSINE_DISTANCE" => Some(Self::CosineDistance),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::L2          => l2_distance(a, b),
            Self::L2Squared   => l2_squared(a, b),
            Self::L1          => l1_distance(a, b),
            Self::LInf        => linf_distance(a, b),
            Self::InnerProduct => inner_product(a, b),
            Self::NegInnerProduct => -inner_product(a, b),
            Self::CosineSimilarity => cosine_similarity(a, b),
            Self::CosineDistance   => cosine_distance(a, b),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extract query vector from various Arrow array types
// ─────────────────────────────────────────────────────────────────────────────

/// Try to extract a `Vec<f32>` query vector from an Arrow array.
///
/// Accepts:
/// - `BinaryArray`  (raw f32 LE bytes, first non-null row)
/// - `Float32Array` (first non-null row — rare, used for inter-column distance)  
/// - `Float64Array` (first non-null row, coerced to f32)
/// - `StringArray`  (JSON array literal "[1.0, 2.0, 3.0]", first non-null row)
pub fn extract_query_vector(arr: &dyn Array) -> io::Result<Vec<f32>> {
    if let Some(ba) = arr.as_any().downcast_ref::<BinaryArray>() {
        for i in 0..ba.len() {
            if !ba.is_null(i) {
                let bytes = ba.value(i);
                if bytes.len() % 4 == 0 && !bytes.is_empty() {
                    let floats = unsafe { bytes_to_f32(bytes) };
                    return Ok(floats.to_vec());
                }
            }
        }
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Empty or invalid binary vector"));
    }

    if let Some(fa) = arr.as_any().downcast_ref::<Float32Array>() {
        for i in 0..fa.len() {
            if !fa.is_null(i) {
                // Whole column is the query vector (unusual — per-row mode)
                let vals: Vec<f32> = (0..fa.len())
                    .filter(|&j| !fa.is_null(j))
                    .map(|j| fa.value(j))
                    .collect();
                return Ok(vals);
            }
        }
    }

    if let Some(da) = arr.as_any().downcast_ref::<Float64Array>() {
        let vals: Vec<f32> = (0..da.len())
            .filter(|&j| !da.is_null(j))
            .map(|j| da.value(j) as f32)
            .collect();
        if !vals.is_empty() {
            return Ok(vals);
        }
    }

    if let Some(sa) = arr.as_any().downcast_ref::<StringArray>() {
        for i in 0..sa.len() {
            if !sa.is_null(i) {
                let s = sa.value(i).trim();
                // Try JSON array "[1.0, 2.0, …]"
                if s.starts_with('[') && s.ends_with(']') {
                    let inner = &s[1..s.len()-1];
                    let parsed: Result<Vec<f32>, _> = inner
                        .split(',')
                        .map(|tok| tok.trim().parse::<f32>())
                        .collect();
                    if let Ok(v) = parsed {
                        if !v.is_empty() {
                            return Ok(v);
                        }
                    }
                }
            }
        }
    }

    Err(io::Error::new(io::ErrorKind::InvalidInput, "Cannot extract query vector from argument"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch distance computation (Arrow in → Arrow out)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `metric(row_vec, query)` for every row in a Binary column.
///
/// Returns a `Float64Array` of the same length with NULLs for NULL/invalid rows.
pub fn batch_distance(
    col: &dyn Array,
    query: &[f32],
    metric: DistanceMetric,
) -> io::Result<ArrayRef> {
    // ── Binary column path (primary: raw f32 bytes) ──────────────────────────
    if let Some(ba) = col.as_any().downcast_ref::<BinaryArray>() {
        let dim = query.len();
        let expected_bytes = dim * 4;
        let distances: Vec<Option<f64>> = (0..ba.len())
            .into_par_iter()
            .map(|i| {
                if ba.is_null(i) { return None; }
                let bytes = ba.value(i);
                if bytes.len() != expected_bytes { return None; }
                // SAFETY: length is exactly dim*4 and aligned to 4 bytes
                let vec = unsafe { bytes_to_f32(bytes) };
                Some(metric.compute(vec, query) as f64)
            })
            .collect();
        return Ok(Arc::new(Float64Array::from(distances)) as ArrayRef);
    }

    // ── String column path ("[1.0, 2.0, …]" per row) ─────────────────────────
    if let Some(sa) = col.as_any().downcast_ref::<StringArray>() {
        let distances: Vec<Option<f64>> = (0..sa.len())
            .into_par_iter()
            .map(|i| {
                if sa.is_null(i) { return None; }
                let s = sa.value(i).trim();
                if !(s.starts_with('[') && s.ends_with(']')) { return None; }
                let inner = &s[1..s.len()-1];
                let parsed: Result<Vec<f32>, _> = inner
                    .split(',')
                    .map(|t| t.trim().parse::<f32>())
                    .collect();
                let vec = parsed.ok()?;
                if vec.len() != query.len() { return None; }
                Some(metric.compute(&vec, query) as f64)
            })
            .collect();
        return Ok(Arc::new(Float64Array::from(distances)) as ArrayRef);
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "array_distance: first argument must be a Binary (vector) or String column",
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// TopK helper: partial sort with a binary max-heap
// ─────────────────────────────────────────────────────────────────────────────

/// Given a `Float64Array` of distances (some possibly NULL),
/// return the indices of the `k` smallest non-null values in ascending order.
///
/// This is O(n log k) — much faster than full sort for small k.
pub fn topk_indices_asc(distances: &Float64Array, k: usize) -> Vec<usize> {
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    if k == 0 { return vec![]; }

    // Max-heap of (dist_bits, idx) — we keep the k smallest
    #[derive(Copy, Clone)]
    struct Entry(u64, usize); // (distance bits for ordering, original row idx)

    impl PartialEq for Entry { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
    impl Eq for Entry {}
    impl PartialOrd for Entry { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
    impl Ord for Entry { fn cmp(&self, o: &Self) -> Ordering { self.0.cmp(&o.0) } }

    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k + 1);

    for i in 0..distances.len() {
        if distances.is_null(i) { continue; }
        let d = distances.value(i);
        let bits = d.to_bits();
        if heap.len() < k {
            heap.push(Entry(bits, i));
        } else if let Some(&Entry(top_bits, _)) = heap.peek() {
            if bits < top_bits {
                heap.pop();
                heap.push(Entry(bits, i));
            }
        }
    }

    let mut result: Vec<(f64, usize)> = heap.into_iter()
        .map(|Entry(b, i)| (f64::from_bits(b), i))
        .collect();
    result.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    result.into_iter().map(|(_, i)| i).collect()
}

/// Given a `Float64Array` of distances (some possibly NULL),
/// return the indices of the `k` *largest* non-null values in descending order.
/// (Used for cosine_similarity / inner_product ORDER BY … DESC LIMIT k.)
pub fn topk_indices_desc(distances: &Float64Array, k: usize) -> Vec<usize> {
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    if k == 0 { return vec![]; }

    #[derive(Copy, Clone)]
    struct Entry(u64, usize); // negated bits to turn max-heap into min-heap

    impl PartialEq for Entry { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
    impl Eq for Entry {}
    impl PartialOrd for Entry { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
    impl Ord for Entry { fn cmp(&self, o: &Self) -> Ordering { self.0.cmp(&o.0) } }

    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k + 1);

    for i in 0..distances.len() {
        if distances.is_null(i) { continue; }
        let d = distances.value(i);
        // negate to turn max into min ordering in the heap
        let neg_bits = (-d).to_bits();
        if heap.len() < k {
            heap.push(Entry(neg_bits, i));
        } else if let Some(&Entry(top_bits, _)) = heap.peek() {
            if neg_bits < top_bits {
                heap.pop();
                heap.push(Entry(neg_bits, i));
            }
        }
    }

    let mut result: Vec<(f64, usize)> = heap.into_iter()
        .map(|Entry(b, i)| (-f64::from_bits(b), i))
        .collect();
    result.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    result.into_iter().map(|(_, i)| i).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Python-side vector encoding helpers (used by bindings.rs)
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a slice of f32 values as raw LE bytes.
pub fn encode_f32_vec(floats: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(floats.len() * 4);
    for &f in floats {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Decode raw LE bytes to a Vec<f32>.
pub fn decode_f32_vec(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 { return None; }
    let len = bytes.len() / 4;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 0.0, 0.0];
        assert!((l2_distance(&a, &b) - 1.0).abs() < 1e-6);

        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let expected = ((3.0f32*3.0 + 3.0*3.0 + 3.0*3.0) as f32).sqrt();
        assert!((l2_distance(&a, &b) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        assert!((inner_product(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_l1_distance() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        assert!((l1_distance(&a, &b) - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_encode_decode() {
        let v = vec![1.0f32, 2.0, 3.0];
        let encoded = encode_f32_vec(&v);
        let decoded = decode_f32_vec(&encoded).unwrap();
        assert_eq!(decoded, v);
    }

    #[test]
    fn test_topk_indices() {
        let dists = Float64Array::from(vec![Some(5.0), Some(1.0), Some(3.0), Some(2.0), None]);
        let idx = topk_indices_asc(&dists, 3);
        assert_eq!(idx, vec![1, 3, 2]); // distances 1.0, 2.0, 3.0
    }

    #[test]
    fn test_batch_distance() {
        // Build a binary column with two 3-dim f32 vectors
        let v1 = encode_f32_vec(&[1.0, 0.0, 0.0]);
        let v2 = encode_f32_vec(&[0.0, 1.0, 0.0]);
        let col: BinaryArray = vec![Some(v1.as_slice()), Some(v2.as_slice())].into();
        let query = vec![1.0f32, 0.0, 0.0];

        let out = batch_distance(&col, &query, DistanceMetric::L2).unwrap();
        let fa = out.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!((fa.value(0)).abs() < 1e-5);          // distance to itself
        assert!((fa.value(1) - 1.4142135).abs() < 1e-4); // √2
    }
}
