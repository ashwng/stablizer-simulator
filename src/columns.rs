//! Persistent column-packed tableau for unitary gates.
//!
//! Each bit is one row, so the AG update equations operate on 64 rows per
//! word. AVX2 handles four words at once; other CPUs use portable word loops.
//! Measurement works directly on columns with O(n) temporary storage. Explicit
//! row views use 64x64 bit transposes and include the scratch row.

use crate::{circuit::Gate, rows::Rows};

#[derive(Clone, Debug)]
pub(crate) struct Columns {
    stride: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    r: Vec<u64>,
}

impl Columns {
    fn zero(nq: usize, nrows: usize) -> Self {
        let stride = nrows.div_ceil(256) * 4;
        Self {
            stride,
            x: vec![0; nq * stride],
            z: vec![0; nq * stride],
            r: vec![0; stride],
        }
    }

    /// Construct the identity directly, without allocating a row tableau.
    pub(crate) fn identity(n: usize) -> Self {
        let mut out = Self::zero(n, 2 * n + 1);
        for q in 0..n {
            out.x[q * out.stride + q / 64] |= 1 << (q % 64);
            out.z[q * out.stride + (n + q) / 64] |= 1 << ((n + q) % 64);
        }
        out
    }

    #[inline]
    pub(crate) fn x(&self, row: usize, q: usize) -> bool {
        self.x[q * self.stride + row / 64] >> (row % 64) & 1 != 0
    }

    #[inline]
    pub(crate) fn z(&self, row: usize, q: usize) -> bool {
        self.z[q * self.stride + row / 64] >> (row % 64) & 1 != 0
    }

    #[inline]
    pub(crate) fn r(&self, row: usize) -> bool {
        self.r[row / 64] >> (row % 64) & 1 != 0
    }

    /// Find an anticommuting stabilizer without inspecting individual bits.
    pub(crate) fn random_pivot(&self, n: usize, q: usize) -> Option<usize> {
        let column = &self.x[q * self.stride..(q + 1) * self.stride];
        for k in n / 64..(2 * n).div_ceil(64) {
            let bits = column[k] & range_mask(k, n, 2 * n);
            if bits != 0 {
                return Some(64 * k + bits.trailing_zeros() as usize);
            }
        }
        None
    }

    /// Multiply the pivot into all anticommuting rows in parallel, then replace
    /// it by the measured Z. Each phase bit represents one row, just like x/z.
    pub(crate) fn measure_random(
        &mut self,
        n: usize,
        q: usize,
        pivot: usize,
        choose: impl FnOnce() -> bool,
    ) -> Result<bool, ()> {
        let partner = pivot - n;
        let (pk, pm) = (pivot / 64, 1u64 << (pivot % 64));
        let (dk, dm) = (partner / 64, 1u64 << (partner % 64));
        // Keep only words containing target rows. Sparse measurements need not
        // stream through the full tableau. Never include scratch or padding.
        let mut targets = Vec::new();
        for k in 0..(2 * n).div_ceil(64) {
            let mut mask = self.x[q * self.stride + k] & range_mask(k, 0, 2 * n);
            if k == pk {
                mask &= !pm;
            }
            if k == dk {
                mask &= !dm;
            }
            if mask != 0 {
                targets.push((k, mask, 0u64, 0u64));
            }
        }
        let pivot_sign = self.r(pivot);
        for col in 0..n {
            let start = col * self.stride;
            let px = self.x[start + pk] & pm != 0;
            let pz = self.z[start + pk] & pm != 0;
            if px || pz {
                for &mut (k, mask, ref mut low, ref mut high) in &mut targets {
                    let x = self.x[start + k];
                    let z = self.z[start + k];
                    // Local Pauli product has phase +i or -i exactly when the
                    // Paulis anticommute. Add +/-1 modulo four using two bit
                    // planes, preserving the carry that determines the sign.
                    let (odd, minus) = match (px, pz) {
                        (true, true) => (x ^ z, x & !z),
                        (true, false) => (z, !x & z),
                        (false, true) => (x, x & z),
                        _ => unreachable!(),
                    };
                    let odd = odd & mask;
                    *high ^= (*low & odd) ^ (minus & mask);
                    *low ^= odd;
                    if px {
                        self.x[start + k] ^= mask;
                    }
                    if pz {
                        self.z[start + k] ^= mask;
                    }
                }
            }
            // Original pivot -> paired destabilizer; pivot -> identity.
            self.x[start + dk] = (self.x[start + dk] & !dm) | if px { dm } else { 0 };
            self.z[start + dk] = (self.z[start + dk] & !dm) | if pz { dm } else { 0 };
            self.x[start + pk] &= !pm;
            self.z[start + pk] &= !pm;
        }
        for (k, mask, low, high) in targets {
            if low != 0 {
                return Err(());
            }
            self.r[k] ^= high ^ if pivot_sign { mask } else { 0 };
        }
        let outcome = choose();
        self.r[dk] = (self.r[dk] & !dm) | if pivot_sign { dm } else { 0 };
        self.r[pk] = (self.r[pk] & !pm) | if outcome { pm } else { 0 };
        self.z[q * self.stride + pk] |= pm;
        Ok(outcome)
    }

    /// Sign of the product of stabilizers selected by destabilizer X bits.
    /// Use P(x,z) = i^(x*z) X^x Z^z. Product phase is the sum of the Y
    /// counts plus twice the number of Z-before-X crossings, minus the output
    /// Y count. Prefix XOR computes the crossing parity for 64 rows at once.
    pub(crate) fn determinate_outcome(&self, n: usize, q: usize) -> Result<bool, ()> {
        let mut masks = vec![0u64; self.stride];
        for k in 0..n.div_ceil(64) {
            let mut bits = self.x[q * self.stride + k] & range_mask(k, 0, n);
            while bits != 0 {
                let row = n + k * 64 + bits.trailing_zeros() as usize;
                masks[row / 64] |= 1 << (row % 64);
                bits &= bits - 1;
            }
        }
        let selected: Vec<_> = masks
            .into_iter()
            .enumerate()
            .filter(|&(_, m)| m != 0)
            .collect();
        if selected.is_empty() {
            return Ok(false);
        }
        if selected.len() == 1 && selected[0].1.is_power_of_two() {
            return Ok(self.r(selected[0].0 * 64 + selected[0].1.trailing_zeros() as usize));
        }
        let mut sign = 0;
        for &(k, mask) in &selected {
            sign ^= (self.r[k] & mask).count_ones() & 1;
        }
        let mut phase = 2 * sign;
        for col in 0..n {
            let start = col * self.stride;
            let (mut xp, mut zp) = (0, 0);
            for &(k, mask) in &selected {
                let x = self.x[start + k] & mask;
                let z = self.z[start + k] & mask;
                let mut prefix = z;
                for shift in [1, 2, 4, 8, 16, 32] {
                    prefix ^= prefix << shift;
                }
                let earlier_z = (prefix << 1) ^ 0u64.wrapping_sub(zp as u64);
                phase = (phase + (x & z).count_ones() + 2 * ((earlier_z & x).count_ones() & 1)) & 3;
                xp ^= x.count_ones() & 1;
                zp ^= z.count_ones() & 1;
            }
            phase = (phase + 4 - (xp & zp)) & 3;
        }
        if phase & 1 != 0 {
            return Err(());
        }
        Ok(phase == 2)
    }

    #[cfg(test)]
    fn from_rows(rows: &Rows) -> Self {
        let mut out = Self::zero(rows.nq, rows.nrows);
        out.load_rows(rows);
        out
    }

    /// Refill existing buffers after a row mutation; dimensions do not change.
    pub(crate) fn load_rows(&mut self, rows: &Rows) {
        transpose_into(
            &rows.x,
            rows.nrows,
            rows.nq,
            rows.w,
            &mut self.x,
            self.stride,
        );
        transpose_into(
            &rows.z,
            rows.nrows,
            rows.nq,
            rows.w,
            &mut self.z,
            self.stride,
        );
        self.r.fill(0);
        for (i, &r) in rows.r.iter().enumerate() {
            self.r[i / 64] |= (r as u64) << (i % 64);
        }
    }

    pub(crate) fn write_rows(&self, rows: &mut Rows) {
        transpose_into(
            &self.x,
            rows.nq,
            rows.nrows,
            self.stride,
            &mut rows.x,
            rows.w,
        );
        transpose_into(
            &self.z,
            rows.nq,
            rows.nrows,
            self.stride,
            &mut rows.z,
            rows.w,
        );
        for (i, r) in rows.r.iter_mut().enumerate() {
            *r = ((self.r[i / 64] >> (i % 64)) & 1) as u8;
        }
    }

    pub(crate) fn apply(&mut self, gates: &[Gate]) {
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        self.apply_with_backend(gates, avx2);
    }

    fn apply_with_backend(&mut self, gates: &[Gate], avx2: bool) {
        let w = self.stride;
        for gate in gates {
            match *gate {
                Gate::H(q) | Gate::P(q) => {
                    let x = &mut self.x[q * w..(q + 1) * w];
                    let z = &mut self.z[q * w..(q + 1) * w];
                    let hadamard = matches!(gate, Gate::H(_));
                    #[cfg(target_arch = "x86_64")]
                    if avx2 {
                        // SAFETY: caller detected AVX2; slices contain full
                        // four-word vectors with equal lengths and no aliasing.
                        unsafe { avx2::single(x, z, &mut self.r, hadamard) };
                        continue;
                    }
                    single_words(x, z, &mut self.r, hadamard);
                }
                Gate::Cnot(a, b) => {
                    let (xa, xb) = two_columns(&mut self.x, w, a, b);
                    let (za, zb) = two_columns(&mut self.z, w, a, b);
                    #[cfg(target_arch = "x86_64")]
                    if avx2 {
                        // SAFETY: same CPU and slice invariants as above.
                        unsafe { avx2::cnot(xa, xb, za, zb, &mut self.r) };
                        continue;
                    }
                    cnot_words(xa, xb, za, zb, &mut self.r);
                }
                Gate::Measure(_) => unreachable!("unitary batch contains a measurement"),
            }
        }
        // Keep the same signature on platforms without the AVX2 implementation.
        let _ = avx2;
    }
}

/// Bits of one word belonging to the half-open row interval [start, end).
fn range_mask(word: usize, start: usize, end: usize) -> u64 {
    let lo = start.saturating_sub(word * 64).min(64);
    let hi = end.saturating_sub(word * 64).min(64);
    let below = |bit: usize| {
        if bit == 64 {
            u64::MAX
        } else {
            (1u64 << bit) - 1
        }
    };
    below(hi) & !below(lo)
}

fn two_columns(data: &mut [u64], w: usize, a: usize, b: usize) -> (&mut [u64], &mut [u64]) {
    assert_ne!(a, b);
    if a < b {
        let (before, after) = data.split_at_mut(b * w);
        (&mut before[a * w..(a + 1) * w], &mut after[..w])
    } else {
        let (before, after) = data.split_at_mut(a * w);
        (&mut after[..w], &mut before[b * w..(b + 1) * w])
    }
}

fn single_words(x: &mut [u64], z: &mut [u64], r: &mut [u64], hadamard: bool) {
    for ((x, z), r) in x.iter_mut().zip(z).zip(r) {
        *r ^= *x & *z;
        if hadamard {
            std::mem::swap(x, z);
        } else {
            *z ^= *x;
        }
    }
}

fn cnot_words(xa: &[u64], xb: &mut [u64], za: &mut [u64], zb: &[u64], r: &mut [u64]) {
    for ((((xa, xb), za), zb), r) in xa.iter().zip(xb).zip(za).zip(zb).zip(r) {
        *r ^= *xa & *zb & !(*xb ^ *za);
        *xb ^= *xa;
        *za ^= *zb;
    }
}

/// Transpose a rectangular packed-bit matrix with independently padded strides.
fn transpose_into(src: &[u64], nrows: usize, ncols: usize, sw: usize, dst: &mut [u64], dw: usize) {
    for q0 in (0..ncols).step_by(64) {
        for r0 in (0..nrows).step_by(64) {
            let mut tile = [0; 64];
            for (i, word) in tile.iter_mut().take((nrows - r0).min(64)).enumerate() {
                *word = src[(r0 + i) * sw + q0 / 64];
            }
            transpose_tile(&mut tile);
            for (j, &word) in tile.iter().take((ncols - q0).min(64)).enumerate() {
                dst[(q0 + j) * dw + r0 / 64] = word;
            }
        }
    }
}

fn transpose_tile(tile: &mut [u64; 64]) {
    for (shift, mask) in [
        (1, 0x5555_5555_5555_5555),
        (2, 0x3333_3333_3333_3333),
        (4, 0x0f0f_0f0f_0f0f_0f0f),
        (8, 0x00ff_00ff_00ff_00ff),
        (16, 0x0000_ffff_0000_ffff),
        (32, 0x0000_0000_ffff_ffff),
    ] {
        for base in (0..64).step_by(2 * shift) {
            for offset in 0..shift {
                let a = base + offset;
                let b = a + shift;
                let swap = ((tile[a] >> shift) ^ tile[b]) & mask;
                tile[a] ^= swap << shift;
                tile[b] ^= swap;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Caller must detect AVX2 and supply equally sized slices, multiples of 4.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn single(x: &mut [u64], z: &mut [u64], r: &mut [u64], hadamard: bool) {
        for ((x, z), r) in x
            .chunks_exact_mut(4)
            .zip(z.chunks_exact_mut(4))
            .zip(r.chunks_exact_mut(4))
        {
            // SAFETY: each chunk is four u64s (32 bytes); loadu/storeu permit
            // unaligned Vec storage. Mutable slices guarantee disjoint writes.
            unsafe {
                let xv = _mm256_loadu_si256(x.as_ptr().cast());
                let zv = _mm256_loadu_si256(z.as_ptr().cast());
                let rv = _mm256_loadu_si256(r.as_ptr().cast());
                _mm256_storeu_si256(
                    r.as_mut_ptr().cast(),
                    _mm256_xor_si256(rv, _mm256_and_si256(xv, zv)),
                );
                if hadamard {
                    _mm256_storeu_si256(x.as_mut_ptr().cast(), zv);
                    _mm256_storeu_si256(z.as_mut_ptr().cast(), xv);
                } else {
                    _mm256_storeu_si256(z.as_mut_ptr().cast(), _mm256_xor_si256(zv, xv));
                }
            }
        }
    }

    /// Caller must detect AVX2 and supply equally sized slices, multiples of 4.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn cnot(
        xa: &[u64],
        xb: &mut [u64],
        za: &mut [u64],
        zb: &[u64],
        r: &mut [u64],
    ) {
        let chunks = xa
            .chunks_exact(4)
            .zip(xb.chunks_exact_mut(4))
            .zip(za.chunks_exact_mut(4))
            .zip(zb.chunks_exact(4))
            .zip(r.chunks_exact_mut(4));
        for ((((xa, xb), za), zb), r) in chunks {
            // SAFETY: all loads/stores address complete disjoint 32-byte chunks.
            unsafe {
                let a = _mm256_loadu_si256(xa.as_ptr().cast());
                let b = _mm256_loadu_si256(xb.as_ptr().cast());
                let c = _mm256_loadu_si256(za.as_ptr().cast());
                let d = _mm256_loadu_si256(zb.as_ptr().cast());
                let sign = _mm256_loadu_si256(r.as_ptr().cast());
                let flip = _mm256_andnot_si256(_mm256_xor_si256(b, c), _mm256_and_si256(a, d));
                _mm256_storeu_si256(r.as_mut_ptr().cast(), _mm256_xor_si256(sign, flip));
                _mm256_storeu_si256(xb.as_mut_ptr().cast(), _mm256_xor_si256(b, a));
                _mm256_storeu_si256(za.as_mut_ptr().cast(), _mm256_xor_si256(c, d));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    #[test]
    fn transpose_matches_individual_bits() {
        let mut rng = Rng::seed_from_u64(42);
        let original = std::array::from_fn(|_| rng.next_u64());
        let mut tile = original;
        transpose_tile(&mut tile);
        for i in 0..64 {
            for j in 0..64 {
                assert_eq!((tile[j] >> i) & 1, (original[i] >> j) & 1);
            }
        }
        transpose_tile(&mut tile);
        assert_eq!(tile, original);
    }

    #[test]
    fn reloading_rows_reuses_columns_and_resets_signs() {
        let mut rows = Rows::new(65, 131);
        rows.set_r(130, true);
        rows.set_x(130, 64, true);
        let mut columns = Columns::from_rows(&rows);
        let buffers = (columns.x.as_ptr(), columns.z.as_ptr(), columns.r.as_ptr());
        columns.apply(&[Gate::P(64), Gate::H(64), Gate::Cnot(64, 0)]);
        rows.zero_row(130);
        rows.set_z(0, 64, true);
        columns.load_rows(&rows);
        assert_eq!(
            buffers,
            (columns.x.as_ptr(), columns.z.as_ptr(), columns.r.as_ptr())
        );
        let mut actual = Rows::new(65, 131);
        columns.write_rows(&mut actual);
        assert!(rows.rows_equal(&actual, 131));
    }

    #[test]
    fn both_backends_match_scalar_gates_on_arbitrary_rows() {
        // Exercise signs, scratch rows, non-square matrices, word boundaries,
        // vector boundaries, both CNOT directions, and zero padding.
        for n in [1, 2, 31, 32, 63, 64, 65, 127, 128, 129, 257] {
            for nrows in [1, 63, 64, 65, 255, 256, 257, 2 * n + 1] {
                let mut rng = Rng::seed_from_u64((n * 1000 + nrows) as u64);
                let mut rows = Rows::new(n, nrows);
                for i in 0..nrows {
                    rows.set_r(i, rng.bit());
                    for q in 0..n {
                        rows.set_x(i, q, rng.bit());
                        rows.set_z(i, q, rng.bit());
                    }
                }
                let mut packed = Columns::from_rows(&rows);
                let mut restored = rows.clone();
                packed.write_rows(&mut restored);
                assert!(rows.rows_equal(&restored, nrows));
                let mut gates = Vec::new();
                let mut expected = rows.clone();
                for _ in 0..300 {
                    let a = rng.below(n);
                    let gate = match rng.below(if n > 1 { 3 } else { 2 }) {
                        0 => Gate::H(a),
                        1 => Gate::P(a),
                        _ => Gate::Cnot(a, (a + 1 + rng.below(n - 1)) % n),
                    };
                    match gate {
                        Gate::H(q) => expected.h(q),
                        Gate::P(q) => expected.p(q),
                        Gate::Cnot(a, b) => expected.cnot(a, b),
                        _ => unreachable!(),
                    }
                    gates.push(gate);
                }
                packed.apply_with_backend(&gates, false);
                packed.write_rows(&mut restored);
                assert!(
                    expected.rows_equal(&restored, nrows),
                    "portable n={n}, rows={nrows}"
                );
                #[cfg(target_arch = "x86_64")]
                if std::is_x86_feature_detected!("avx2") {
                    let mut packed = Columns::from_rows(&rows);
                    packed.apply_with_backend(&gates, true);
                    packed.write_rows(&mut restored);
                    assert!(
                        expected.rows_equal(&restored, nrows),
                        "AVX2 n={n}, rows={nrows}"
                    );
                }
            }
        }
    }
}
