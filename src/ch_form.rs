//! Phase-sensitive stabilizer states in CH-form (arXiv:1808.00128, Section 4.1).
//!
//! This module defines the API and initializes |0…0⟩. Gate updates, amplitude
//! evaluation and projection are placeholders that return [`crate::Error::Value`].

use crate::error::{index_err, value_err, Result};
use crate::gf2::BitMatrix;
use num_complex::Complex64;

/// A stabilizer vector |φ⟩ = ω U_C H(v) |s⟩, including its global scalar.
///
/// The C-type Clifford U_C fixes |0…0⟩, with inverse conjugation defined by
///
/// ```text
/// U_C⁻¹ Z_p U_C = ∏_j Z_j^{G[p,j]}
/// U_C⁻¹ X_p U_C = i^{γ[p]} ∏_j X_j^{F[p,j]} Z_j^{M[p,j]}.
/// ```
///
/// The phase vector γ is modulo four. H(v) applies H where v is true, and s
/// is a computational basis string. The scalar ω includes both phase and
/// normalization; projection may produce an unnormalized or zero vector.
/// Qubits use the same zero-based indexing as [`crate::Tableau`].
///
/// This is an API scaffold: only initialization and inspection are implemented.
#[derive(Clone, Debug, PartialEq)]
pub struct ChForm {
    f: BitMatrix,
    g: BitMatrix,
    m: BitMatrix,
    gamma: Vec<u8>,
    v: Vec<bool>,
    s: Vec<bool>,
    omega: Complex64,
}

impl ChForm {
    /// Initializes |0…0⟩ on `n` qubits. Returns an error if `n == 0`.
    pub fn zero(n: usize) -> Result<Self> {
        if n == 0 {
            return value_err("n must be positive");
        }
        Ok(Self {
            f: BitMatrix::identity(n),
            g: BitMatrix::identity(n),
            m: BitMatrix::zeros(n, n),
            gamma: vec![0; n],
            v: vec![false; n],
            s: vec![false; n],
            omega: Complex64::new(1.0, 0.0),
        })
    }

    pub fn num_qubits(&self) -> usize {
        self.s.len()
    }

    /// X exponents in U_C⁻¹ X_p U_C.
    pub fn f(&self) -> &BitMatrix {
        &self.f
    }

    /// Z exponents in U_C⁻¹ Z_p U_C.
    pub fn g(&self) -> &BitMatrix {
        &self.g
    }

    /// Z exponents in U_C⁻¹ X_p U_C.
    pub fn m(&self) -> &BitMatrix {
        &self.m
    }

    /// Phase exponents γ modulo four in U_C⁻¹ X_p U_C.
    pub fn gamma(&self) -> &[u8] {
        &self.gamma
    }

    /// The mask v specifying the Hadamard layer H(v).
    pub fn hadamard_mask(&self) -> &[bool] {
        &self.v
    }

    /// The computational basis string s, indexed by qubit.
    pub fn basis_bits(&self) -> &[bool] {
        &self.s
    }

    /// The global scalar ω, including normalization.
    pub fn omega(&self) -> Complex64 {
        self.omega
    }

    /// Applies a Hadamard gate. Not implemented yet; leaves the state unchanged.
    pub fn h(&mut self, q: usize) -> Result<()> {
        self.check_qubit(q)?;
        value_err("ChForm::h is not implemented yet")
    }

    /// Applies S = diag(1, i), called P in the tableau API.
    /// Not implemented yet; leaves the state unchanged.
    pub fn s(&mut self, q: usize) -> Result<()> {
        self.check_qubit(q)?;
        value_err("ChForm::s is not implemented yet")
    }

    /// Applies controlled-X from `control` to `target` (distinct qubits).
    /// Not implemented yet; leaves the state unchanged.
    pub fn cx(&mut self, control: usize, target: usize) -> Result<()> {
        self.check_pair(control, target)?;
        value_err("ChForm::cx is not implemented yet")
    }

    /// Applies controlled-Z to distinct qubits.
    /// Not implemented yet; leaves the state unchanged.
    pub fn cz(&mut self, a: usize, b: usize) -> Result<()> {
        self.check_pair(a, b)?;
        value_err("ChForm::cz is not implemented yet")
    }

    /// Returns ⟨bits|φ⟩ including ω. `bits[q]` is qubit q, and the slice must
    /// contain exactly `num_qubits()` bits. Not implemented yet.
    pub fn amplitude(&self, bits: &[bool]) -> Result<Complex64> {
        if bits.len() != self.num_qubits() {
            return value_err(format!(
                "expected {} basis bits, got {}",
                self.num_qubits(),
                bits.len()
            ));
        }
        value_err("ChForm::amplitude is not implemented yet")
    }

    /// Applies (I + (−1)^outcome Z_q)/2 without normalizing. `false` selects
    /// |0⟩ and `true` selects |1⟩. An impossible branch will be represented by
    /// ω = 0 rather than an error. Not implemented yet; leaves the state unchanged.
    pub fn project_z(&mut self, q: usize, _outcome: bool) -> Result<()> {
        self.check_qubit(q)?;
        value_err("ChForm::project_z is not implemented yet")
    }

    fn check_qubit(&self, q: usize) -> Result<()> {
        if q >= self.num_qubits() {
            return index_err(format!("qubit {q} out of range [0, {})", self.num_qubits()));
        }
        Ok(())
    }

    fn check_pair(&self, a: usize, b: usize) -> Result<()> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        if a == b {
            return value_err("controlled gate qubits must differ");
        }
        Ok(())
    }
}
