"""Check column-native collapse and deterministic phases against Stim."""

import random

import pytest

import stabilizer

stim = pytest.importorskip("stim")


@pytest.mark.parametrize("n", [1, 2, 63, 64, 65, 129])
def test_interleaved_measurements_match_stim(n):
    rng = random.Random(721 + n)
    rust = stabilizer.Tableau(n)
    reference = stim.TableauSimulator()
    reference.set_num_qubits(n)
    for step in range(30 * n):
        q = rng.randrange(n)
        kind = rng.randrange(4 if n > 1 else 3)
        if kind == 0:
            rust.apply_h(q)
            reference.h(q)
        elif kind == 1:
            rust.apply_s(q)
            reference.s(q)
        elif kind == 3:
            target = (q + 1 + rng.randrange(n - 1)) % n
            rust.apply_cnot(q, target)
            reference.cnot(q, target)
        else:
            expected = reference.peek_z(q)
            outcome = rng.randrange(2) if expected == 0 else int(expected == -1)
            assert rust.is_deterministic(q) == (expected != 0)
            assert rust.peek(q) == (None if expected == 0 else outcome)
            assert rust.measure(q, force=outcome) == outcome
            reference.postselect_z(q, desired_value=bool(outcome))
            assert rust.peek(q) == outcome
            # A rejected forced outcome must preserve the state.
            with pytest.raises(ValueError):
                rust.measure(q, force=1 - outcome)
            assert rust.peek(q) == outcome
        # Export only at the end: intermediate inspection would create a row
        # cache and conceal errors in alternating gates and column measurement.
    forward = reference.current_inverse_tableau().inverse()
    expected = stabilizer.Tableau.from_stabilizers(
        [str(forward.z_output(q)).replace("_", "I") for q in range(n)]
    )
    assert rust.same_state(expected)
    rust.check_invariants()
