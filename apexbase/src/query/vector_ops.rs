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

/// Cosine similarity with **pre-computed query norm** `nb = ‖b‖`.
///
/// Fuses the dot-product and ‖a‖ computation into a single 8-way unrolled pass,
/// avoiding the extra full pass over `a` that the two-call version would need.
/// `nb` should be computed once per query (not once per row).
#[inline(always)]
pub fn cosine_similarity_fused(a: &[f32], b: &[f32], nb: f32) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na2 = 0.0f32;
    let c = n / 8;
    for i in 0..c {
        let base = i << 3;
        dot += a[base]*b[base] + a[base+1]*b[base+1] + a[base+2]*b[base+2] + a[base+3]*b[base+3]
             + a[base+4]*b[base+4] + a[base+5]*b[base+5] + a[base+6]*b[base+6] + a[base+7]*b[base+7];
        na2 += a[base]*a[base] + a[base+1]*a[base+1] + a[base+2]*a[base+2] + a[base+3]*a[base+3]
             + a[base+4]*a[base+4] + a[base+5]*a[base+5] + a[base+6]*a[base+6] + a[base+7]*a[base+7];
    }
    for i in (c*8)..n {
        dot += a[i] * b[i];
        na2 += a[i] * a[i];
    }
    let na = na2.sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Cosine distance = 1 − cosine_similarity.
#[inline(always)]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

// ─────────────────────────────────────────────────────────────────────────────
// DistanceComputer: precomputes per-query state to avoid N recomputations
// ─────────────────────────────────────────────────────────────────────────────

/// Holds the query vector and any precomputed values (e.g. query norm for cosine).
///
/// Construct once per query; call `compute(row)` N times.
/// This avoids recomputing the query norm O(n) times in batch/TopK scans.
pub struct DistanceComputer {
    pub metric: DistanceMetric,
    pub query:  Vec<f32>,
    /// Pre-computed ‖query‖ for cosine metrics (0.0 for others).
    query_norm: f32,
}

impl DistanceComputer {
    pub fn new(metric: DistanceMetric, query: Vec<f32>) -> Self {
        let query_norm = match metric {
            DistanceMetric::CosineSimilarity | DistanceMetric::CosineDistance => {
                query.iter().map(|x| x * x).sum::<f32>().sqrt()
            }
            _ => 0.0,
        };
        Self { metric, query, query_norm }
    }

    #[inline(always)]
    pub fn compute(&self, a: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::L2             => l2_distance(a, &self.query),
            DistanceMetric::L2Squared      => l2_squared(a, &self.query),
            DistanceMetric::L1             => l1_distance(a, &self.query),
            DistanceMetric::LInf           => linf_distance(a, &self.query),
            DistanceMetric::InnerProduct   => inner_product(a, &self.query),
            DistanceMetric::NegInnerProduct => -inner_product(a, &self.query),
            DistanceMetric::CosineSimilarity =>
                cosine_similarity_fused(a, &self.query, self.query_norm),
            DistanceMetric::CosineDistance =>
                1.0 - cosine_similarity_fused(a, &self.query, self.query_norm),
        }
    }
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
// Fused TopK: single-pass compute + heap, O(n log k)
// ─────────────────────────────────────────────────────────────────────────────

/// Single-pass TopK directly on a `BinaryArray` of stored vectors.
///
/// Fuses SIMD distance computation with max-heap maintenance in **one**
/// sequential pass — no intermediate `Float64Array` allocation for all rows.
/// O(n log k) time, O(k) extra space.
///
/// Returns `Vec<(row_index, f32_distance)>` sorted **ascending** (nearest first).
pub fn topk_heap_direct(
    col: &BinaryArray,
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<(usize, f32)> {
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    if k == 0 || col.len() == 0 || query.is_empty() {
        return vec![];
    }
    let k_capped = k.min(col.len());
    let expected_bytes = query.len() * 4;

    // Max-heap of (f32_bits, row_idx).
    // IEEE-754 positive f32 bit patterns sort identically to the float values,
    // so direct bit comparison is correct for non-NaN, non-negative distances.
    #[derive(Copy, Clone)]
    struct Entry(u32, usize);
    impl PartialEq  for Entry { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
    impl Eq         for Entry {}
    impl PartialOrd for Entry {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    impl Ord for Entry { fn cmp(&self, o: &Self) -> Ordering { self.0.cmp(&o.0) } }

    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k_capped + 1);

    for i in 0..col.len() {
        if col.is_null(i) { continue; }
        let bytes = col.value(i);
        if bytes.len() != expected_bytes { continue; }
        // SAFETY: length == query.len()*4, bytes are valid LE f32 values.
        let vec = unsafe { bytes_to_f32(bytes) };
        let dist = metric.compute(vec, query);
        if dist.is_nan() { continue; }
        let bits = dist.to_bits();
        if heap.len() < k_capped {
            heap.push(Entry(bits, i));
        } else if let Some(&Entry(top_bits, _)) = heap.peek() {
            if bits < top_bits {
                heap.pop();
                heap.push(Entry(bits, i));
            }
        }
    }

    let mut result: Vec<(usize, f32)> = heap
        .into_iter()
        .map(|Entry(b, i)| (i, f32::from_bits(b)))
        .collect();
    result.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel TopK: Rayon fold+reduce, O(n/T log k) per thread
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel TopK on a `BinaryArray` using Rayon.
///
/// Uses `DistanceComputer` which has pre-computed query-norm for cosine metrics.
/// Splits the array into exactly T = num_threads contiguous chunks; each thread
/// maintains one local max-heap of size k, then T heaps are merged.
/// Inner loop uses raw buffer pointers to avoid per-row `col.value(i)` overhead.
///
/// Returns `Vec<(row_index, f32_distance)>` sorted **ascending** (nearest first).
pub fn topk_heap_direct_parallel(
    col: &BinaryArray,
    computer: &DistanceComputer,
    k: usize,
) -> Vec<(usize, f32)> {
    use rayon::prelude::*;
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    let n = col.len();
    if k == 0 || n == 0 || computer.query.is_empty() {
        return vec![];
    }
    let k_capped = k.min(n);
    let dim = computer.query.len();
    let expected_bytes = dim * 4;

    #[derive(Copy, Clone)]
    struct Entry(u32, usize);
    impl PartialEq  for Entry { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
    impl Eq         for Entry {}
    impl PartialOrd for Entry { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
    impl Ord        for Entry { fn cmp(&self, o: &Self) -> Ordering { self.0.cmp(&o.0) } }

    // Grab raw buffer slices ONCE — avoid Arc deref + bounds checks inside the hot loop.
    // SAFETY: these references live for the duration of the function, col is immutable.
    let values: &[u8]  = col.values().as_slice();
    let offsets: &[i32] = col.offsets().as_ref(); // len = n+1
    // Null bitmap: Option<&[u8]> pointing into Arrow's validity buffer.
    let null_bytes: Option<&[u8]> = col.nulls().map(|nb| nb.buffer().as_slice());

    // Capture raw pointers as usize so they are Send across Rayon threads.
    // SAFETY: values, offsets, null_bytes are all derived from `col` which is
    //         immutable and outlives the parallel section below.
    let val_ptr  = values.as_ptr() as usize;
    let val_len  = values.len();
    let off_ptr  = offsets.as_ptr() as usize;
    let null_ptr: Option<(usize, usize)> = null_bytes.map(|nb| (nb.as_ptr() as usize, nb.len()));

    // Split into exactly T chunks (one per Rayon worker thread).
    let t = rayon::current_num_threads().max(1);
    let chunk_size = (n + t - 1) / t;

    let per_chunk: Vec<Vec<(usize, f32)>> = (0..t)
        .into_par_iter()
        .map(|tid| {
            let start = tid * chunk_size;
            if start >= n { return vec![]; }
            let end = (start + chunk_size).min(n);

            // SAFETY: pointers are valid for `col`'s lifetime (enforced by borrow above).
            let values  = unsafe { std::slice::from_raw_parts(val_ptr  as *const u8,  val_len) };
            let offsets = unsafe { std::slice::from_raw_parts(off_ptr  as *const i32, n + 1)  };

            let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k_capped + 1);

            for i in start..end {
                // Null check via raw bitmap (bit i).
                if let Some((nb_ptr, nb_len)) = null_ptr {
                    let byte_idx = i / 8;
                    if byte_idx < nb_len {
                        // SAFETY: byte_idx < nb_len.
                        let byte = unsafe { *(nb_ptr as *const u8).add(byte_idx) };
                        if (byte >> (i & 7)) & 1 == 0 { continue; }
                    }
                }

                // SAFETY: offsets has n+1 elements; i and i+1 are within bounds.
                let off_s = unsafe { *offsets.get_unchecked(i)   } as usize;
                let off_e = unsafe { *offsets.get_unchecked(i+1) } as usize;
                if off_e - off_s != expected_bytes { continue; }

                // SAFETY: off_s..off_e is within values (enforced by BinaryArray invariants).
                let bytes = unsafe { values.get_unchecked(off_s..off_e) };
                let vec   = unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const f32, dim)
                };

                let dist = computer.compute(vec);
                if dist.is_nan() { continue; }

                let bits = dist.to_bits();
                if heap.len() < k_capped {
                    heap.push(Entry(bits, i));
                } else if let Some(&Entry(top, _)) = heap.peek() {
                    if bits < top {
                        heap.pop();
                        heap.push(Entry(bits, i));
                    }
                }
            }
            heap.into_iter().map(|Entry(b, i)| (i, f32::from_bits(b))).collect()
        })
        .collect();

    // Merge T small top-k lists into final top-k (T is small, e.g. 8-16).
    let mut final_heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k_capped + 1);
    for chunk in per_chunk {
        for (idx, dist) in chunk {
            let bits = dist.to_bits();
            if final_heap.len() < k_capped {
                final_heap.push(Entry(bits, idx));
            } else if let Some(&Entry(top, _)) = final_heap.peek() {
                if bits < top {
                    final_heap.pop();
                    final_heap.push(Entry(bits, idx));
                }
            }
        }
    }

    let mut result: Vec<(usize, f32)> = final_heap
        .into_iter()
        .map(|Entry(b, i)| (i, f32::from_bits(b)))
        .collect();
    result.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    result
}

/// Parallel TopK on a `FixedSizeListArray<Float32>` using Rayon.
///
/// Compared to the `BinaryArray` version, the inner loop is simpler:
/// the float data is stored contiguously with stride = `dim`, so there
/// are no per-row offset lookups.  This eliminates one pointer dereference
/// and the `off_e - off_s != expected_bytes` size-check per row.
///
/// Returns `Vec<(row_index, f32_distance)>` sorted **ascending** (nearest first).
pub fn topk_heap_direct_parallel_fixed(
    col: &arrow::array::FixedSizeListArray,
    computer: &DistanceComputer,
    k: usize,
) -> Vec<(usize, f32)> {
    use rayon::prelude::*;
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    let n = col.len();
    if k == 0 || n == 0 || computer.query.is_empty() {
        return vec![];
    }
    let k_capped = k.min(n);
    let dim = computer.query.len();

    // The values child is a flat Float32Array with n*dim elements.
    let values_child = col.values();
    let float_arr = values_child
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .expect("FixedSizeList child must be Float32Array");
    let floats: &[f32] = float_arr.values().as_ref();
    // floats has length n * dim (no nulls expected for vector columns)

    let float_ptr = floats.as_ptr() as usize;
    let float_len = floats.len();

    #[derive(Copy, Clone)]
    struct Entry(u32, usize);
    impl PartialEq  for Entry { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
    impl Eq         for Entry {}
    impl PartialOrd for Entry { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
    impl Ord        for Entry { fn cmp(&self, o: &Self) -> Ordering { self.0.cmp(&o.0) } }

    let t = rayon::current_num_threads().max(1);
    let chunk_size = (n + t - 1) / t;

    let per_chunk: Vec<Vec<(usize, f32)>> = (0..t)
        .into_par_iter()
        .map(|tid| {
            let start = tid * chunk_size;
            if start >= n { return vec![]; }
            let end = (start + chunk_size).min(n);

            // SAFETY: float_ptr is valid for `col`'s lifetime (borrow above).
            let floats = unsafe { std::slice::from_raw_parts(float_ptr as *const f32, float_len) };
            let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k_capped + 1);

            for i in start..end {
                let off = i * dim;
                if off + dim > float_len { break; }
                // SAFETY: bounds checked above.
                let vec = unsafe { floats.get_unchecked(off..off + dim) };
                let dist = computer.compute(vec);
                if dist.is_nan() { continue; }
                let bits = dist.to_bits();
                if heap.len() < k_capped {
                    heap.push(Entry(bits, i));
                } else if let Some(&Entry(top, _)) = heap.peek() {
                    if bits < top {
                        heap.pop();
                        heap.push(Entry(bits, i));
                    }
                }
            }
            heap.into_iter().map(|Entry(b, i)| (i, f32::from_bits(b))).collect()
        })
        .collect();

    // Merge T small top-k lists.
    let mut final_heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k_capped + 1);
    for chunk in per_chunk {
        for (idx, dist) in chunk {
            let bits = dist.to_bits();
            if final_heap.len() < k_capped {
                final_heap.push(Entry(bits, idx));
            } else if let Some(&Entry(top, _)) = final_heap.peek() {
                if bits < top {
                    final_heap.pop();
                    final_heap.push(Entry(bits, idx));
                }
            }
        }
    }

    let mut result: Vec<(usize, f32)> = final_heap
        .into_iter()
        .map(|Entry(b, i)| (i, f32::from_bits(b)))
        .collect();
    result.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    result
}

/// Core parallel TopK on a raw contiguous `&[f32]` (no Arrow, no allocation).
/// `floats` has length `n_rows * dim`. Zero-copy when called with an mmap slice.
/// Returns `Vec<(row_index, f32_distance)>` sorted ascending (nearest first).
pub fn topk_heap_on_floats(
    floats: &[f32],
    n_rows: usize,
    dim: usize,
    computer: &DistanceComputer,
    k: usize,
) -> Vec<(usize, f32)> {
    use rayon::prelude::*;
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    if k == 0 || n_rows == 0 || dim == 0 || computer.query.is_empty() {
        return vec![];
    }
    let k_capped = k.min(n_rows);

    #[derive(Copy, Clone)]
    struct Entry(u32, usize);
    impl PartialEq  for Entry { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
    impl Eq         for Entry {}
    impl PartialOrd for Entry { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
    impl Ord        for Entry { fn cmp(&self, o: &Self) -> Ordering { self.0.cmp(&o.0) } }

    let float_ptr = floats.as_ptr() as usize;
    let float_len = floats.len();
    let t = rayon::current_num_threads().max(1);
    let chunk_size = (n_rows + t - 1) / t;

    let per_chunk: Vec<Vec<(usize, f32)>> = (0..t)
        .into_par_iter()
        .map(|tid| {
            let start = tid * chunk_size;
            if start >= n_rows { return vec![]; }
            let end = (start + chunk_size).min(n_rows);
            let floats = unsafe { std::slice::from_raw_parts(float_ptr as *const f32, float_len) };
            let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k_capped + 1);
            for i in start..end {
                let off = i * dim;
                if off + dim > float_len { break; }
                let vec = unsafe { floats.get_unchecked(off..off + dim) };
                let dist = computer.compute(vec);
                if dist.is_nan() { continue; }
                let bits = dist.to_bits();
                if heap.len() < k_capped {
                    heap.push(Entry(bits, i));
                } else if let Some(&Entry(top, _)) = heap.peek() {
                    if bits < top { heap.pop(); heap.push(Entry(bits, i)); }
                }
            }
            heap.into_iter().map(|Entry(b, i)| (i, f32::from_bits(b))).collect()
        })
        .collect();

    let mut final_heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k_capped + 1);
    for chunk in per_chunk {
        for (idx, dist) in chunk {
            let bits = dist.to_bits();
            if final_heap.len() < k_capped {
                final_heap.push(Entry(bits, idx));
            } else if let Some(&Entry(top, _)) = final_heap.peek() {
                if bits < top { final_heap.pop(); final_heap.push(Entry(bits, idx)); }
            }
        }
    }
    let mut result: Vec<(usize, f32)> = final_heap
        .into_iter()
        .map(|Entry(b, i)| (i, f32::from_bits(b)))
        .collect();
    result.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    result
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
