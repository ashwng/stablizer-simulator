//! Bit-packed rows of a tableau: the storage shared by the pure tableau
//! (Section III), the mixed-state tableau (Section VII A) and the history oracle (Section V).
//!
//! Row i is the Pauli (−1)^{r_i} · P, with the x and z bits of P packed into
//! `w` words each. Rows are contiguous (row-major), as in CHP, so `rowsum`
//! is a word-wise XOR plus a popcount.

use crate::pauli::{anticommute_words, g_sum, locate, num_words, PauliString};

#[derive(Clone, Debug)]
pub struct Rows {
    pub(crate) nq: usize,
    pub(crate) nrows: usize,
    pub(crate) w: usize,
    pub(crate) x: Vec<u64>,
    pub(crate) z: Vec<u64>,
    pub(crate) r: Vec<u8>,
}

impl Rows {
    pub fn new(nq: usize, nrows: usize) -> Self {
        let w = num_words(nq);
        Rows {
            nq,
            nrows,
            w,
            x: vec![0; nrows * w],
            z: vec![0; nrows * w],
            r: vec![0; nrows],
        }
    }

    #[inline]
    pub fn xs(&self, i: usize) -> &[u64] {
        &self.x[i * self.w..(i + 1) * self.w]
    }

    #[inline]
    pub fn zs(&self, i: usize) -> &[u64] {
        &self.z[i * self.w..(i + 1) * self.w]
    }

    #[inline]
    pub fn x(&self, i: usize, q: usize) -> bool {
        let (k, m) = locate(q);
        self.x[i * self.w + k] & m != 0
    }

    #[inline]
    pub fn z(&self, i: usize, q: usize) -> bool {
        let (k, m) = locate(q);
        self.z[i * self.w + k] & m != 0
    }

    #[inline]
    pub fn r(&self, i: usize) -> bool {
        self.r[i] != 0
    }

    #[inline]
    pub fn set_x(&mut self, i: usize, q: usize, v: bool) {
        let (k, m) = locate(q);
        let word = &mut self.x[i * self.w + k];
        *word = if v { *word | m } else { *word & !m };
    }

    #[inline]
    pub fn set_z(&mut self, i: usize, q: usize, v: bool) {
        let (k, m) = locate(q);
        let word = &mut self.z[i * self.w + k];
        *word = if v { *word | m } else { *word & !m };
    }

    #[inline]
    pub fn set_r(&mut self, i: usize, v: bool) {
        self.r[i] = v as u8;
    }

    pub fn zero_row(&mut self, i: usize) {
        let w = self.w;
        self.x[i * w..(i + 1) * w].fill(0);
        self.z[i * w..(i + 1) * w].fill(0);
        self.r[i] = 0;
    }

    pub fn copy_row(&mut self, dst: usize, src: usize) {
        let w = self.w;
        self.x.copy_within(src * w..(src + 1) * w, dst * w);
        self.z.copy_within(src * w..(src + 1) * w, dst * w);
        self.r[dst] = self.r[src];
    }

    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for k in 0..self.w {
            self.x.swap(a * self.w + k, b * self.w + k);
            self.z.swap(a * self.w + k, b * self.w + k);
        }
        self.r.swap(a, b);
    }

    /// Row i as a Pauli with phase 0 or 2.
    pub fn row_pauli(&self, i: usize) -> PauliString {
        PauliString::from_words(
            self.nq,
            self.xs(i).to_vec(),
            self.zs(i).to_vec(),
            2 * self.r[i],
        )
    }

    /// Overwrites row i. The Pauli must be Hermitian (phase ±1).
    pub fn set_row_pauli(&mut self, i: usize, p: &PauliString) {
        debug_assert!(p.is_hermitian());
        let w = self.w;
        self.x[i * w..(i + 1) * w].copy_from_slice(p.x_words());
        self.z[i * w..(i + 1) * w].copy_from_slice(p.z_words());
        self.r[i] = p.is_negative() as u8;
    }

    /// True iff rows i and j anticommute (symplectic product 1).
    pub fn anticommute(&self, i: usize, j: usize) -> bool {
        anticommute_words(self.xs(i), self.zs(i), self.xs(j), self.zs(j))
    }

    pub fn anticommutes_with(&self, i: usize, p: &PauliString) -> bool {
        anticommute_words(self.xs(i), self.zs(i), p.x_words(), p.z_words())
    }

    /// rowsum(h, i) of Section III: row h ← row i · row h, with
    /// r_h set from 2 r_h + 2 r_i + Σ_j g(x_ij, z_ij, x_hj, z_hj) mod 4.
    ///
    /// The sum is odd exactly when the rows anticommute; the product then has
    /// phase ±i, which one sign bit cannot hold. In that case row h is left
    /// unchanged and the sum mod 4 is returned as the error.
    pub fn rowsum(&mut self, h: usize, i: usize) -> Result<(), u8> {
        let w = self.w;
        let (hs, is) = (h * w, i * w);
        let mut s = 2 * (self.r[h] as i64 + self.r[i] as i64);
        s += g_sum(
            &self.x[is..is + w],
            &self.z[is..is + w],
            &self.x[hs..hs + w],
            &self.z[hs..hs + w],
        );
        let s = s.rem_euclid(4);
        if s % 2 == 1 {
            return Err(s as u8);
        }
        self.r[h] = (s / 2) as u8;
        for k in 0..w {
            self.x[hs + k] ^= self.x[is + k];
            self.z[hs + k] ^= self.z[is + k];
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Gates (Section III), applied to every row.
    // ------------------------------------------------------------------

    /// CNOT from control a to target b:
    /// r_i ⊕= x_ia z_ib (x_ib ⊕ z_ia ⊕ 1), x_ib ⊕= x_ia, z_ia ⊕= z_ib.
    pub fn cnot(&mut self, a: usize, b: usize) {
        let (ka, ma) = locate(a);
        let (kb, mb) = locate(b);
        let w = self.w;
        for i in 0..self.nrows {
            let xa = self.x[i * w + ka] & ma != 0;
            let za = self.z[i * w + ka] & ma != 0;
            let xb = self.x[i * w + kb] & mb != 0;
            let zb = self.z[i * w + kb] & mb != 0;
            self.r[i] ^= (xa & zb & !(xb ^ za)) as u8;
            self.x[i * w + kb] ^= (0u64.wrapping_sub(xa as u64)) & mb;
            self.z[i * w + ka] ^= (0u64.wrapping_sub(zb as u64)) & ma;
        }
    }

    /// Hadamard on qubit a: r_i ⊕= x_ia z_ia, then swap x_ia and z_ia.
    pub fn h(&mut self, a: usize) {
        let (k, m) = locate(a);
        let w = self.w;
        for i in 0..self.nrows {
            let xi = self.x[i * w + k];
            let zi = self.z[i * w + k];
            self.r[i] ^= ((xi & zi & m) != 0) as u8;
            self.x[i * w + k] = (xi & !m) | (zi & m);
            self.z[i * w + k] = (zi & !m) | (xi & m);
        }
    }

    /// Phase on qubit a: r_i ⊕= x_ia z_ia, then z_ia ⊕= x_ia.
    pub fn p(&mut self, a: usize) {
        let (k, m) = locate(a);
        let w = self.w;
        for i in 0..self.nrows {
            let xa = self.x[i * w + k] & m;
            let za = self.z[i * w + k] & m;
            self.r[i] ^= (xa & za != 0) as u8;
            self.z[i * w + k] ^= xa;
        }
    }

    /// Rows 0..count of `self` and `other` are identical (bits and signs).
    pub fn rows_equal(&self, other: &Rows, count: usize) -> bool {
        self.nq == other.nq
            && self.x[..count * self.w] == other.x[..count * other.w]
            && self.z[..count * self.w] == other.z[..count * other.w]
            && self.r[..count] == other.r[..count]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_bits_stay_zero() {
        let n = 70;
        let mut rows = Rows::new(n, 3);
        let y = PauliString::parse(&"Y".repeat(n)).unwrap();
        rows.set_row_pauli(0, &y);
        rows.set_row_pauli(1, &y);
        rows.rowsum(1, 0).unwrap();
        for a in 0..n {
            rows.h(a);
            rows.p(a);
        }
        rows.cnot(0, n - 1);
        let pad = !((1u64 << (n % 64)) - 1);
        for i in 0..3 {
            assert_eq!(rows.xs(i)[1] & pad, 0);
            assert_eq!(rows.zs(i)[1] & pad, 0);
        }
    }

    #[test]
    fn rowsum_rejects_anticommuting_rows() {
        let mut rows = Rows::new(2, 2);
        rows.set_row_pauli(0, &PauliString::parse("XI").unwrap());
        rows.set_row_pauli(1, &PauliString::parse("ZI").unwrap());
        assert!(rows.rowsum(1, 0).is_err());
        assert_eq!(rows.row_pauli(1).to_string(), "+ZI");
    }
}
