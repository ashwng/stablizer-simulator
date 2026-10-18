//! Stabilizer circuits: sequences of CNOT, Hadamard, phase and measurement
//! gates (Fig. 1 of the paper).

use crate::error::{index_err, value_err, Result};
use crate::rng::Rng;
use crate::tableau::{MeasureResult, Tableau};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gate {
    /// CNOT from control to target.
    Cnot(usize, usize),
    H(usize),
    /// The phase gate P = diag(1, i), also called S.
    P(usize),
    /// Measurement in the standard basis.
    Measure(usize),
}

impl Gate {
    /// The CHP instruction letter: `c`, `h`, `p` or `m`.
    pub fn letter(&self) -> char {
        match self {
            Gate::Cnot(..) => 'c',
            Gate::H(_) => 'h',
            Gate::P(_) => 'p',
            Gate::Measure(_) => 'm',
        }
    }

    pub fn qubits(&self) -> Vec<usize> {
        match *self {
            Gate::Cnot(a, b) => vec![a, b],
            Gate::H(a) | Gate::P(a) | Gate::Measure(a) => vec![a],
        }
    }

    pub fn is_unitary(&self) -> bool {
        !matches!(self, Gate::Measure(_))
    }
}

impl fmt::Display for Gate {
    /// CHP syntax, e.g. `c 0 1`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Gate::Cnot(a, b) => write!(f, "c {a} {b}"),
            Gate::H(a) => write!(f, "h {a}"),
            Gate::P(a) => write!(f, "p {a}"),
            Gate::Measure(a) => write!(f, "m {a}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Circuit {
    n: usize,
    gates: Vec<Gate>,
}

impl Circuit {
    pub fn new(n: usize) -> Self {
        Circuit {
            n,
            gates: Vec::new(),
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    pub fn len(&self) -> usize {
        self.gates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    pub fn push(&mut self, gate: Gate) -> Result<()> {
        for q in gate.qubits() {
            if q >= self.n {
                return index_err(format!("qubit {q} out of range [0, {})", self.n));
            }
        }
        if let Gate::Cnot(a, b) = gate {
            if a == b {
                return value_err("CNOT control and target must differ");
            }
        }
        self.gates.push(gate);
        Ok(())
    }

    pub fn cnot(&mut self, a: usize, b: usize) -> Result<()> {
        self.push(Gate::Cnot(a, b))
    }

    pub fn h(&mut self, a: usize) -> Result<()> {
        self.push(Gate::H(a))
    }

    pub fn p(&mut self, a: usize) -> Result<()> {
        self.push(Gate::P(a))
    }

    pub fn measure(&mut self, a: usize) -> Result<()> {
        self.push(Gate::Measure(a))
    }

    pub fn extend(&mut self, other: &Circuit) -> Result<()> {
        for g in &other.gates {
            self.push(*g)?;
        }
        Ok(())
    }

    pub fn is_unitary(&self) -> bool {
        self.gates.iter().all(Gate::is_unitary)
    }

    /// The inverse of a unitary circuit: gates reversed, with P⁻¹ = P³.
    pub fn inverse(&self) -> Result<Circuit> {
        if !self.is_unitary() {
            return value_err("only unitary circuits have an inverse");
        }
        let mut out = Circuit::new(self.n);
        for g in self.gates.iter().rev() {
            let copies = if matches!(g, Gate::P(_)) { 3 } else { 1 };
            for _ in 0..copies {
                out.gates.push(*g);
            }
        }
        Ok(out)
    }

    /// Runs of consecutive gates of the same kind, as (letter, count).
    pub fn rounds(&self) -> Vec<(char, usize)> {
        let mut out: Vec<(char, usize)> = Vec::new();
        for g in &self.gates {
            match out.last_mut() {
                Some((c, k)) if *c == g.letter() => *k += 1,
                _ => out.push((g.letter(), 1)),
            }
        }
        out
    }

    /// The tableau this unitary circuit produces from the standard initial
    /// tableau. By Lemma 5 it determines the circuit up to equivalence.
    pub fn tableau(&self) -> Result<Tableau> {
        if !self.is_unitary() {
            return value_err("tableau() needs a unitary circuit");
        }
        let mut t = Tableau::new(self.n.max(1));
        self.run(&mut t, &mut Rng::seed_from_u64(0))?;
        Ok(t)
    }

    /// Runs the circuit on `tab`, returning (qubit, result) per measurement.
    pub fn run(&self, tab: &mut Tableau, rng: &mut Rng) -> Result<Vec<(usize, MeasureResult)>> {
        if tab.num_qubits() < self.n {
            return value_err(format!(
                "circuit uses {} qubits but the tableau has {}",
                self.n,
                tab.num_qubits()
            ));
        }
        let mut out = Vec::new();
        let mut remaining = self.gates.as_slice();
        while !remaining.is_empty() {
            let unitary_len = remaining.iter().take_while(|g| g.is_unitary()).count();
            let (unitaries, rest) = remaining.split_at(unitary_len);
            tab.apply_unitaries(unitaries);
            if let Some((&Gate::Measure(q), rest)) = rest.split_first() {
                out.push((q, tab.measure(q, rng)?));
                remaining = rest;
            } else {
                break;
            }
        }
        Ok(out)
    }

    /// The circuit as a CHP program.
    pub fn to_chp(&self) -> String {
        let mut s = String::from("#\n");
        for g in &self.gates {
            s.push_str(&g.to_string());
            s.push('\n');
        }
        s
    }
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_chp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_gate(n: usize, rng: &mut Rng) -> Gate {
        let a = rng.below(n);
        match rng.below(if n > 1 { 3 } else { 2 }) {
            0 => Gate::H(a),
            1 => Gate::P(a),
            _ => Gate::Cnot(a, (a + 1 + rng.below(n - 1)) % n),
        }
    }

    #[test]
    fn batches_preserve_measurement_results_and_full_tableau() {
        for n in [1, 2, 31, 32, 63, 64, 65, 127, 128, 129] {
            for idle in [0, 3] {
                let mut random = Rng::seed_from_u64(n as u64);
                let mut circuit = Circuit::new(n);
                // Leading, adjacent, and trailing measurements; runs on both
                // sides of the batch threshold, with larger destination states.
                for len in [0, 63, 64, 65, 257, 1, 0] {
                    for _ in 0..len {
                        circuit.push(random_gate(n, &mut random)).unwrap();
                    }
                    circuit.measure(random.below(n)).unwrap();
                }
                let mut scalar = Tableau::new(n + idle);
                for _ in 0..4 * (n + idle) {
                    scalar
                        .apply_unitary(&random_gate(n + idle, &mut random))
                        .unwrap();
                }
                let mut batched = scalar.clone();
                let mut scalar_rng = Rng::seed_from_u64(123);
                let mut batch_rng = scalar_rng.clone();
                let mut expected = Vec::new();
                for gate in circuit.gates() {
                    if let Some(result) = scalar.apply(gate, &mut scalar_rng).unwrap() {
                        expected.push((gate.qubits()[0], result));
                    }
                }
                let actual = circuit.run(&mut batched, &mut batch_rng).unwrap();
                assert_eq!(actual, expected, "n={n}, idle={idle}");
                assert_eq!(batched, scalar, "n={n}, idle={idle}");
                assert_eq!(batch_rng.next_u64(), scalar_rng.next_u64());
                batched.check_invariants().unwrap();
            }
        }
    }

    #[test]
    fn long_unitary_tableau_matches_scalar_and_rejects_small_destination() {
        let mut random = Rng::seed_from_u64(7);
        let mut circuit = Circuit::new(65);
        let mut expected = Tableau::new(65);
        for _ in 0..400 {
            let gate = random_gate(65, &mut random);
            circuit.push(gate).unwrap();
            expected.apply_unitary(&gate).unwrap();
        }
        assert_eq!(circuit.tableau().unwrap(), expected);
        let mut small = Tableau::new(64);
        assert!(circuit.run(&mut small, &mut random).is_err());
        assert_eq!(small, Tableau::new(64));
    }
}
