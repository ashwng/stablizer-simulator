//! The tableau algorithm of Section III.
//!
//! A state on n qubits is a 2n × (2n+1) binary tableau (plus one scratch
//! row). Rows 0..n are the destabilizers, rows n..2n the stabilizers, and
//! row 2n is the scratch space used by determinate measurements.
//!
//! Proposition 3 gives the invariants: stabilizers commute, destabilizers
//! commute, and destabilizer i anticommutes with stabilizer i and commutes
//! with every other stabilizer. Gates take O(n) time; both kinds of
//! measurement take O(n²).
//!
//! Unitary updates and measurement use persistent column-packed storage. Other
//! row algorithms lazily obtain a synchronized row view; mutable access
//! invalidates the columns until the next gate. Read-only views do not force a repack.

use crate::circuit::Gate;
use crate::columns::Columns;
use crate::error::{index_err, value_err, Error, Result};
use crate::gf2::symplectic_completion;
use crate::pauli::PauliString;
use crate::rng::Rng;
use crate::rows::Rows;
use std::fmt;
use std::sync::{Mutex, OnceLock};

/// The outcome bit (true = |1⟩) and whether it was determinate (Case II).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeasureResult {
    pub outcome: bool,
    pub deterministic: bool,
}

#[derive(Debug)]
pub struct Tableau {
    n: usize,
    columns: Columns,
    // Columns are authoritative unless a mutable row view was requested.
    columns_current: bool,
    rows: OnceLock<Rows>,
    // A discarded read cache becomes the next conversion's destination.
    // Locking occurs only when lazily constructing a view through &self;
    // mutations use Mutex::get_mut, so the gate loop never locks.
    spare_rows: Mutex<Option<Rows>>,
}

impl Clone for Tableau {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            columns: self.columns.clone(),
            columns_current: self.columns_current,
            // Copy authoritative rows only; derived read caches are disposable.
            rows: if self.columns_current {
                OnceLock::new()
            } else {
                OnceLock::from(self.rows.get().expect("current row view").clone())
            },
            spare_rows: Mutex::new(None),
        }
    }
}

impl Tableau {
    /// The standard initial tableau for |0…0⟩: destabilizer i is X_i and
    /// stabilizer i is Z_i.
    ///
    /// # Panics
    /// If `n == 0`.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "a tableau needs at least one qubit");
        Tableau {
            n,
            columns: Columns::identity(n),
            columns_current: true,
            rows: OnceLock::new(),
            spare_rows: Mutex::new(None),
        }
    }

    /// A tableau whose stabilizer generators are exactly `stabs` (n
    /// independent, commuting, Hermitian Paulis). Destabilizers are found by
    /// symplectic completion.
    pub fn from_stabilizers(stabs: &[PauliString]) -> Result<Self> {
        let n = stabs.len();
        if n == 0 {
            return value_err("need at least one stabilizer");
        }
        let (destabs, _, _) = symplectic_completion(n, stabs)?;
        let mut t = Tableau::new(n);
        let rows = t.rows_mut();
        for i in 0..n {
            rows.set_row_pauli(i, &destabs[i]);
            rows.set_row_pauli(n + i, &stabs[i]);
        }
        Ok(t)
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    pub fn num_words(&self) -> usize {
        crate::pauli::num_words(self.n)
    }

    pub(crate) fn rows(&self) -> &Rows {
        self.rows.get_or_init(|| {
            debug_assert!(self.columns_current);
            let mut rows = self
                .spare_rows
                .lock()
                .expect("row buffer lock")
                .take()
                .unwrap_or_else(|| Rows::new(self.n, 2 * self.n + 1));
            self.columns.write_rows(&mut rows);
            rows
        })
    }

    pub(crate) fn rows_mut(&mut self) -> &mut Rows {
        self.rows();
        self.columns_current = false;
        self.rows.get_mut().expect("current row view")
    }

    fn columns_mut(&mut self) -> &mut Columns {
        if !self.columns_current {
            self.columns
                .load_rows(self.rows.get().expect("current row view"));
            self.columns_current = true;
        }
        if let Some(rows) = self.rows.take() {
            *self.spare_rows.get_mut().expect("row buffer lock") = Some(rows);
        }
        &mut self.columns
    }

    /// Circuit owns validated gates; unlike individual gate methods, this can
    /// avoid repeated dispatch and validation without imposing a size threshold.
    pub(crate) fn apply_unitaries(&mut self, gates: &[Gate]) {
        if !gates.is_empty() {
            self.columns_mut().apply(gates);
        }
    }

    pub(crate) fn check_qubit(&self, q: usize) -> Result<()> {
        if q >= self.n {
            return index_err(format!("qubit {q} out of range [0, {})", self.n));
        }
        Ok(())
    }

    fn check_row(&self, row: usize) -> Result<()> {
        if row >= 2 * self.n {
            return index_err(format!("row {row} out of range [0, {})", 2 * self.n));
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Bit accessors (row in 0..2n, qubit in 0..n).
    // ------------------------------------------------------------------

    pub fn get_x(&self, row: usize, q: usize) -> Result<bool> {
        self.check_row(row)?;
        self.check_qubit(q)?;
        Ok(if self.columns_current {
            self.columns.x(row, q)
        } else {
            self.rows().x(row, q)
        })
    }

    pub fn get_z(&self, row: usize, q: usize) -> Result<bool> {
        self.check_row(row)?;
        self.check_qubit(q)?;
        Ok(if self.columns_current {
            self.columns.z(row, q)
        } else {
            self.rows().z(row, q)
        })
    }

    pub fn get_r(&self, row: usize) -> Result<bool> {
        self.check_row(row)?;
        Ok(if self.columns_current {
            self.columns.r(row)
        } else {
            self.rows().r(row)
        })
    }

    pub fn set_x(&mut self, row: usize, q: usize, v: bool) -> Result<()> {
        self.check_row(row)?;
        self.check_qubit(q)?;
        self.rows_mut().set_x(row, q, v);
        Ok(())
    }

    pub fn set_z(&mut self, row: usize, q: usize, v: bool) -> Result<()> {
        self.check_row(row)?;
        self.check_qubit(q)?;
        self.rows_mut().set_z(row, q, v);
        Ok(())
    }

    pub fn set_r(&mut self, row: usize, v: bool) -> Result<()> {
        self.check_row(row)?;
        self.rows_mut().set_r(row, v);
        Ok(())
    }

    /// Row as a signed Pauli.
    pub fn row(&self, row: usize) -> Result<PauliString> {
        self.check_row(row)?;
        Ok(self.rows().row_pauli(row))
    }

    /// Overwrites a row. This can break the Proposition 3 invariants, so it is
    /// meant for tests and hand-built examples.
    pub fn set_row(&mut self, row: usize, p: &PauliString) -> Result<()> {
        self.check_row(row)?;
        if p.num_qubits() != self.n {
            return value_err(format!(
                "expected {} Pauli letters, got {}",
                self.n,
                p.num_qubits()
            ));
        }
        if !p.is_hermitian() {
            return value_err(format!("{p} has phase ±i, which a row cannot store"));
        }
        self.rows_mut().set_row_pauli(row, p);
        Ok(())
    }

    pub fn destabilizers(&self) -> Vec<PauliString> {
        let rows = self.rows();
        (0..self.n).map(|i| rows.row_pauli(i)).collect()
    }

    pub fn stabilizers(&self) -> Vec<PauliString> {
        let rows = self.rows();
        (self.n..2 * self.n).map(|i| rows.row_pauli(i)).collect()
    }

    /// rowsum(h, i): row h ← row i · row h. Fails, leaving row h unchanged,
    /// if the rows anticommute.
    pub fn rowsum(&mut self, h: usize, i: usize) -> Result<()> {
        self.check_row(h)?;
        self.check_row(i)?;
        self.rows_mut().rowsum(h, i).map_err(|s| {
            Error::Value(format!(
                "rowsum({h}, {i}): phase sum {s} mod 4 is odd -- rows anticommute"
            ))
        })
    }

    // ------------------------------------------------------------------
    // Gates
    // ------------------------------------------------------------------

    pub fn cnot(&mut self, a: usize, b: usize) -> Result<()> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        if a == b {
            return value_err("CNOT control and target must differ");
        }
        self.columns_mut().apply(&[Gate::Cnot(a, b)]);
        Ok(())
    }

    pub fn h(&mut self, a: usize) -> Result<()> {
        self.check_qubit(a)?;
        self.columns_mut().apply(&[Gate::H(a)]);
        Ok(())
    }

    pub fn p(&mut self, a: usize) -> Result<()> {
        self.check_qubit(a)?;
        self.columns_mut().apply(&[Gate::P(a)]);
        Ok(())
    }

    /// Applies a unitary gate; measurement gates are rejected.
    pub fn apply_unitary(&mut self, gate: &Gate) -> Result<()> {
        match *gate {
            Gate::Cnot(a, b) => self.cnot(a, b),
            Gate::H(a) => self.h(a),
            Gate::P(a) => self.p(a),
            Gate::Measure(_) => value_err("expected a unitary gate, got a measurement"),
        }
    }

    /// Applies any gate; returns the result for a measurement.
    pub fn apply(&mut self, gate: &Gate, rng: &mut Rng) -> Result<Option<MeasureResult>> {
        match *gate {
            Gate::Measure(a) => self.measure(a, rng).map(Some),
            _ => self.apply_unitary(gate).map(|_| None),
        }
    }

    // ------------------------------------------------------------------
    // Measurement
    // ------------------------------------------------------------------

    /// The first stabilizer row p with x_pa = 1, if any (Case I).
    fn random_pivot(&self, a: usize) -> Option<usize> {
        if self.columns_current {
            self.columns.random_pivot(self.n, a)
        } else {
            let rows = self.rows();
            (self.n..2 * self.n).find(|&p| rows.x(p, a))
        }
    }

    /// Whether measuring qubit `a` has a determinate outcome. O(n).
    pub fn is_deterministic(&self, a: usize) -> Result<bool> {
        self.check_qubit(a)?;
        Ok(self.random_pivot(a).is_none())
    }

    /// Case II: Z_a = ±∏ c_h R_{h+n} with c_h = x_ha, so sum those
    /// stabilizers and read the product's sign. Only the row fallback uses the
    /// scratch row; columns accumulate the phase with packed parity operations.
    fn determinate_outcome(&mut self, a: usize) -> Result<bool> {
        if self.columns_current {
            return self
                .columns
                .determinate_outcome(self.n, a)
                .map_err(|_| broken_invariants());
        }
        let n = self.n;
        let scratch = 2 * n;
        let rows = self.rows_mut();
        rows.zero_row(scratch);
        for i in 0..n {
            if rows.x(i, a) {
                rows.rowsum(scratch, i + n)
                    .map_err(|_| broken_invariants())?;
            }
        }
        Ok(rows.r(scratch))
    }

    /// The outcome that measuring `a` would give, if it is determinate,
    /// without changing the state.
    pub fn peek(&mut self, a: usize) -> Result<Option<bool>> {
        self.check_qubit(a)?;
        if self.random_pivot(a).is_some() {
            return Ok(None);
        }
        self.determinate_outcome(a).map(Some)
    }

    fn measure_with(&mut self, a: usize, choose: impl FnOnce() -> bool) -> Result<MeasureResult> {
        self.check_qubit(a)?;
        match self.random_pivot(a) {
            Some(p) => self.collapse_random(a, p, choose),
            None => Ok(MeasureResult {
                outcome: self.determinate_outcome(a)?,
                deterministic: true,
            }),
        }
    }

    fn collapse_random(
        &mut self,
        a: usize,
        p: usize,
        choose: impl FnOnce() -> bool,
    ) -> Result<MeasureResult> {
        let n = self.n;
        let outcome = if self.columns_current {
            self.columns_mut()
                .measure_random(n, a, p, choose)
                .map_err(|_| broken_invariants())?
        } else {
            let rows = self.rows_mut();
            // The paired destabilizer is overwritten, so skip its ±i product.
            for i in 0..2 * n {
                if i != p && i != p - n && rows.x(i, a) {
                    rows.rowsum(i, p).map_err(|_| broken_invariants())?;
                }
            }
            rows.copy_row(p - n, p);
            rows.zero_row(p);
            let outcome = choose();
            rows.set_r(p, outcome);
            rows.set_z(p, a, true);
            outcome
        };
        Ok(MeasureResult {
            outcome,
            deterministic: false,
        })
    }

    /// Measures qubit `a` in the standard basis (Section III, Cases I and II).
    pub fn measure(&mut self, a: usize, rng: &mut Rng) -> Result<MeasureResult> {
        self.measure_with(a, || rng.bit())
    }

    /// Measures qubit `a`, choosing `outcome` if the result is random
    /// (post-selection). Fails, without changing the state, if the outcome
    /// is determinate and differs from `outcome`.
    pub fn measure_forced(&mut self, a: usize, outcome: bool) -> Result<MeasureResult> {
        self.check_qubit(a)?;
        if let Some(p) = self.random_pivot(a) {
            return self.collapse_random(a, p, || outcome);
        }
        let det = self.determinate_outcome(a)?;
        if det != outcome {
            return value_err(format!(
                "qubit {a} is determinately {}, cannot force {}",
                det as u8, outcome as u8
            ));
        }
        Ok(MeasureResult {
            outcome: det,
            deterministic: true,
        })
    }

    // ------------------------------------------------------------------
    // Inspection
    // ------------------------------------------------------------------

    /// Checks the invariants of Proposition 3: rows i and j anticommute
    /// exactly when they are a destabilizer/stabilizer pair (|i − j| = n).
    pub fn check_invariants(&self) -> Result<()> {
        let n = self.n;
        let rows = self.rows();
        for i in 0..2 * n {
            for j in i + 1..2 * n {
                let want = j == i + n;
                if rows.anticommute(i, j) != want {
                    return value_err(format!(
                        "rows {i} and {j} should {}",
                        if want { "anticommute" } else { "commute" }
                    ));
                }
            }
        }
        Ok(())
    }

    /// The reduced row-echelon generating set of the stabilizer group, with
    /// signs. Pivots are taken in the column order x_0, z_0, x_1, z_1, ….
    /// Two tableaus describe the same state iff these lists are equal.
    pub fn canonical_stabilizers(&self) -> Vec<PauliString> {
        canonical_generators(self.n, self.stabilizers())
    }

    /// Same quantum state (the destabilizers may differ).
    pub fn same_state(&self, other: &Tableau) -> bool {
        self.n == other.n && self.canonical_stabilizers() == other.canonical_stabilizers()
    }

    /// The standard initial tableau, with all signs +.
    pub fn is_identity(&self) -> bool {
        *self == Tableau::new(self.n)
    }
}

/// Reduced row echelon form of commuting Hermitian generators, with signs
/// tracked by Pauli multiplication.
pub(crate) fn canonical_generators(n: usize, mut gens: Vec<PauliString>) -> Vec<PauliString> {
    let mut row = 0;
    for q in 0..n {
        for use_x in [true, false] {
            let bit = |p: &PauliString| {
                let (x, z) = p.get(q);
                if use_x {
                    x
                } else {
                    z
                }
            };
            let Some(piv) = (row..gens.len()).find(|&i| bit(&gens[i])) else {
                continue;
            };
            gens.swap(row, piv);
            for i in 0..gens.len() {
                if i != row && bit(&gens[i]) {
                    gens[i] = gens[row].mul(&gens[i]);
                }
            }
            row += 1;
        }
    }
    gens
}

fn broken_invariants() -> Error {
    Error::Value(
        "tableau invariants are broken (rowsum on anticommuting rows); \
         was a row overwritten by hand?"
            .into(),
    )
}

impl PartialEq for Tableau {
    /// Rows 0..2n equal, signs included (the scratch row is ignored). By
    /// Lemma 5, two unitary circuits are equivalent iff their tableaus are.
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.rows().rows_equal(other.rows(), 2 * self.n)
    }
}

impl Eq for Tableau {}

impl fmt::Display for Tableau {
    /// Destabilizers, a dashed line, then stabilizers.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, p) in self.destabilizers().iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{p}")?;
        }
        write!(f, "\n{}", "-".repeat(self.n + 1))?;
        for p in self.stabilizers() {
            write!(f, "\n{p}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_measurements_match_rows_without_materializing_a_view() {
        // Dense entanglement, Y phases, both outcomes, repeated deterministic
        // measurements, and non-aligned stabilizer halves/word boundaries.
        for n in [1, 2, 31, 63, 64, 65, 127, 128, 129, 257] {
            let mut packed = Tableau::new(n);
            let mut scalar = Tableau::new(n);
            scalar.rows_mut();
            let mut random = Rng::seed_from_u64(921 + n as u64);
            for _ in 0..20 * n {
                let a = random.below(n);
                match random.below(if n > 1 { 3 } else { 2 }) {
                    0 => {
                        packed.h(a).unwrap();
                        scalar.rows_mut().h(a);
                    }
                    1 => {
                        packed.p(a).unwrap();
                        scalar.rows_mut().p(a);
                    }
                    _ => {
                        let b = (a + 1 + random.below(n - 1)) % n;
                        packed.cnot(a, b).unwrap();
                        scalar.rows_mut().cnot(a, b);
                    }
                }
            }
            let mut actual = Rows::new(n, 2 * n + 1);
            let mut a_rng = Rng::seed_from_u64(71);
            let mut b_rng = a_rng.clone();
            for step in 0..3 * n {
                let q = step % n;
                assert_eq!(
                    packed.peek(q).unwrap(),
                    scalar.peek(q).unwrap(),
                    "n={n}, q={q}"
                );
                if step % 2 == 0 {
                    assert_eq!(
                        packed.measure(q, &mut a_rng).unwrap(),
                        scalar.measure(q, &mut b_rng).unwrap()
                    );
                } else {
                    let outcome = scalar.peek(q).unwrap().unwrap_or_else(|| random.bit());
                    assert_eq!(
                        packed.measure_forced(q, outcome).unwrap(),
                        scalar.measure_forced(q, outcome).unwrap()
                    );
                }
                let det = packed.peek(q).unwrap().unwrap();
                assert!(packed.measure_forced(q, !det).is_err());
                assert!(packed.columns_current);
                assert!(
                    packed.rows.get().is_none(),
                    "measurement allocated a row view"
                );
                packed.columns.write_rows(&mut actual);
                assert!(
                    actual.rows_equal(scalar.rows(), 2 * n),
                    "n={n}, step={step}"
                );
                // Re-entangle some measured qubits to exercise further random
                // collapses after mutations, rather than only the first pass.
                if step % 3 == 0 {
                    packed.h(q).unwrap();
                    scalar.rows_mut().h(q);
                    packed.p(q).unwrap();
                    scalar.rows_mut().p(q);
                    if n > 1 {
                        let b = (q + 1) % n;
                        packed.cnot(q, b).unwrap();
                        scalar.rows_mut().cnot(q, b);
                    }
                }
            }
            assert_eq!(a_rng.next_u64(), b_rng.next_u64());
        }
    }

    #[test]
    fn dense_deterministic_column_products_match_row_products() {
        for n in [63, 64, 65, 129, 257] {
            let mut packed = Tableau::new(n);
            let mut scalar = Tableau::new(n);
            let mut random = Rng::seed_from_u64(n as u64);
            // Signed computational basis state, represented in a dense basis
            // of Z stabilizers. Deterministic outcomes require products of many
            // selected generators, including selections spanning word boundaries.
            for q in 0..n {
                if random.bit() {
                    for gate in [Gate::H(q), Gate::P(q), Gate::P(q), Gate::H(q)] {
                        packed.apply_unitary(&gate).unwrap();
                        scalar.apply_unitary(&gate).unwrap();
                    }
                }
            }
            scalar.rows_mut();
            for _ in 0..40 * n {
                let a = random.below(n);
                let b = (a + 1 + random.below(n - 1)) % n;
                packed.cnot(a, b).unwrap();
                scalar.rows_mut().cnot(a, b);
            }
            for q in 0..n {
                assert_eq!(
                    packed.peek(q).unwrap(),
                    scalar.peek(q).unwrap(),
                    "n={n}, q={q}"
                );
                assert!(packed.is_deterministic(q).unwrap());
            }
            assert!(packed.rows.get().is_none());
        }
    }

    #[test]
    fn row_views_are_lazy_and_conversion_buffers_are_reused() {
        let mut t = Tableau::new(1025);
        assert!(t.rows.get().is_none());
        // A one-gate circuit on a large tableau must stay in column storage.
        t.apply_unitaries(&[Gate::H(1024)]);
        assert!(t.rows.get().is_none());
        assert!(t.get_z(1024, 1024).unwrap());
        assert!(t.get_x(1025 + 1024, 1024).unwrap());
        assert!(t.rows.get().is_none());

        let buffer = t.rows().x.as_ptr();
        for _ in 0..10 {
            t.h(0).unwrap();
            assert!(t.columns_current);
            assert!(t.rows.get().is_none());
            assert_eq!(t.rows().x.as_ptr(), buffer);
        }
        t.set_r(0, true).unwrap();
        assert!(!t.columns_current);
        assert!(t.get_r(0).unwrap());
        t.p(0).unwrap();
        assert!(t.columns_current);
        assert_eq!(t.rows().x.as_ptr(), buffer);
        assert!(t.get_r(0).unwrap());
    }

    #[test]
    fn persistent_storage_matches_row_algorithm_through_mutations() {
        for n in [1, 2, 63, 64, 65, 129] {
            let mut t = Tableau::new(n);
            let mut scalar = Tableau::new(n);
            scalar.rows_mut();
            let mut random = Rng::seed_from_u64(n as u64);
            let mut a_rng = Rng::seed_from_u64(17);
            let mut b_rng = a_rng.clone();
            for _ in 0..300 {
                let a = random.below(n);
                let b = (a + 1) % n;
                match random.below(8) {
                    0 => {
                        t.h(a).unwrap();
                        scalar.rows_mut().h(a);
                    }
                    1 => {
                        t.p(a).unwrap();
                        scalar.rows_mut().p(a);
                    }
                    2 if n > 1 => {
                        t.cnot(a, b).unwrap();
                        scalar.rows_mut().cnot(a, b);
                    }
                    3 => assert_eq!(
                        t.measure(a, &mut a_rng).unwrap(),
                        scalar.measure(a, &mut b_rng).unwrap()
                    ),
                    4 => assert_eq!(t.peek(a).unwrap(), scalar.peek(a).unwrap()),
                    5 => {
                        let row = t.row(a).unwrap();
                        t.set_row(a, &row).unwrap();
                    }
                    6 if n > 1 => {
                        // Change stabilizer basis and the paired destabilizers.
                        t.rowsum(n + a, n + b).unwrap();
                        t.rowsum(b, a).unwrap();
                        scalar.rows_mut().rowsum(n + a, n + b).unwrap();
                        scalar.rows_mut().rowsum(b, a).unwrap();
                    }
                    _ => {
                        let mut copy = t.clone();
                        assert_eq!(copy, scalar);
                        copy.h(a).unwrap();
                        copy.h(a).unwrap();
                        assert_eq!(copy, scalar);
                    }
                }
                assert_eq!(t, scalar, "n={n}");
                assert_eq!(t.get_x(a, b).unwrap(), scalar.get_x(a, b).unwrap());
                assert_eq!(t.get_z(a, b).unwrap(), scalar.get_z(a, b).unwrap());
                assert_eq!(t.get_r(a).unwrap(), scalar.get_r(a).unwrap());
                assert_eq!(
                    t.is_deterministic(a).unwrap(),
                    scalar.is_deterministic(a).unwrap()
                );
            }
            t.check_invariants().unwrap();
            assert_eq!(a_rng.next_u64(), b_rng.next_u64());
        }
    }

    #[test]
    fn concurrent_readers_share_a_synchronized_view() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Tableau>();
        let mut t = Tableau::new(65);
        t.h(64).unwrap();
        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..4).map(|_| scope.spawn(|| t.stabilizers())).collect();
            for thread in threads {
                let stabs = thread.join().unwrap();
                assert_eq!(stabs[64], PauliString::single(65, 64, 'X').unwrap());
            }
        });
        assert!(t.columns_current);
    }

    #[test]
    fn identity_and_printer() {
        let t = Tableau::new(3);
        assert_eq!(t.to_string(), "+XII\n+IXI\n+IIX\n----\n+ZII\n+IZI\n+IIZ");
        assert!(t.check_invariants().is_ok());
    }

    #[test]
    fn bell_pair_measurements_agree() {
        for seed in 0..20 {
            let mut rng = Rng::seed_from_u64(seed);
            let mut t = Tableau::new(2);
            t.h(0).unwrap();
            t.cnot(0, 1).unwrap();
            let a = t.measure(0, &mut rng).unwrap();
            let b = t.measure(1, &mut rng).unwrap();
            assert!(!a.deterministic);
            assert!(b.deterministic);
            assert_eq!(a.outcome, b.outcome);
            t.check_invariants().unwrap();
        }
    }

    #[test]
    fn x_gate_from_hpph() {
        // X = H Z H = H P P H, so |0⟩ → |1⟩ determinately.
        let mut t = Tableau::new(1);
        for g in [Gate::H(0), Gate::P(0), Gate::P(0), Gate::H(0)] {
            t.apply_unitary(&g).unwrap();
        }
        let mut rng = Rng::seed_from_u64(0);
        assert_eq!(
            t.measure(0, &mut rng).unwrap(),
            MeasureResult {
                outcome: true,
                deterministic: true
            }
        );
    }

    #[test]
    fn invariants_survive_random_circuits() {
        let mut rng = Rng::seed_from_u64(42);
        for n in [1, 2, 5, 64, 65] {
            let mut t = Tableau::new(n);
            for _ in 0..20 * n {
                match rng.below(4) {
                    0 if n > 1 => {
                        let a = rng.below(n);
                        let mut b = rng.below(n - 1);
                        if b >= a {
                            b += 1;
                        }
                        t.cnot(a, b).unwrap();
                    }
                    1 => t.h(rng.below(n)).unwrap(),
                    2 => t.p(rng.below(n)).unwrap(),
                    _ => {
                        t.measure(rng.below(n), &mut rng).unwrap();
                    }
                }
            }
            t.check_invariants().unwrap();
        }
    }

    #[test]
    fn forced_measurement() {
        let mut t = Tableau::new(1);
        assert!(t.measure_forced(0, true).is_err());
        t.h(0).unwrap();
        let r = t.measure_forced(0, true).unwrap();
        assert!(r.outcome && !r.deterministic);
        assert_eq!(t.peek(0).unwrap(), Some(true));
    }

    #[test]
    fn from_stabilizers_roundtrip() {
        let stabs: Vec<PauliString> = ["XXX", "ZZI", "-IZZ"]
            .iter()
            .map(|s| PauliString::parse(s).unwrap())
            .collect();
        let t = Tableau::from_stabilizers(&stabs).unwrap();
        t.check_invariants().unwrap();
        assert_eq!(t.stabilizers(), stabs);
    }
}
