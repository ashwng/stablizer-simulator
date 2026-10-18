"""Section IV: the CHP language and the example programs the paper mentions."""

import pathlib
import os
import random
import subprocess
import sys

import pytest

from stabilizer import Circuit, Tableau, parse_chp, run_chp

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples" / "chp"


def program(name):
    return (EXAMPLES / f"{name}.chp").read_text()


def body(name):
    """The instructions of an example, without its description header."""
    text = program(name)
    return text[text.index("#\n") + 2:]


def test_parse_paper_teleport():
    c = parse_chp(program("teleport"))
    assert c.n == 5
    assert len(c) == 12
    assert c.gates()[:3] == [("h", 1), ("c", 1, 2), ("c", 0, 1)]
    assert parse_chp(c.to_chp()) == c


def test_parse_errors():
    with pytest.raises(ValueError, match="line 3"):
        parse_chp("#\nh 0\nx 1\n")
    with pytest.raises(ValueError):
        parse_chp("c 0 0")
    with pytest.raises(ValueError):
        parse_chp("m")


# Single-qubit preparations: gate strings applied to |0> on qubit 0.
PREPS = ["", "h", "hp", "hpp", "hppp", "hpph", "hph"]


def invert(prep):
    """Inverse of a word in h and p (p^-1 = ppp), as gate letters."""
    out = []
    for letter in reversed(prep):
        out.extend(["p"] * 3 if letter == "p" else ["h"])
    return out


@pytest.mark.parametrize("prep", PREPS)
def test_teleportation_moves_the_state(prep):
    # Prepare qubit 0, teleport, undo the preparation on qubit 2: qubit 2
    # must then be determinately |0>.
    lines = [f"{g} 0" for g in prep] + [body("teleport").strip()]
    lines += [f"{g} 2" for g in invert(prep)] + ["m 2"]
    text = "#\n" + "\n".join(lines) + "\n"
    for seed in range(8):
        outcomes, _ = run_chp(text, seed=seed)
        assert outcomes[-1] == (2, 0, True)


def test_ghz():
    for seed in range(10):
        (a, b, c), _ = run_chp(program("ghz"), seed=seed)
        assert not a[2] and b[2] and c[2]
        assert a[1] == b[1] == c[1]


def test_dense_coding_sends_two_bits():
    for seed in range(5):
        outcomes, _ = run_chp(program("dense_coding"), seed=seed)
        assert outcomes == [(0, 1, True), (1, 1, True)]


def test_simon_outcomes_orthogonal_to_secret():
    secret = (1, 1, 0)
    seen = set()
    for seed in range(40):
        outcomes, _ = run_chp(program("simon"), seed=seed)
        y = tuple(o for _, o, _ in outcomes)
        assert sum(a * b for a, b in zip(y, secret)) % 2 == 0
        seen.add(y)
    assert len(seen) == 4  # the whole orthogonal complement of s


def test_shor9_corrects_bit_flip():
    for seed in range(5):
        outcomes, _ = run_chp(program("shor9"), seed=seed)
        assert outcomes == [(9, 1, True), (10, 1, True), (0, 0, True)]


def test_run_returns_final_tableau():
    _, t = run_chp("#\nh 0\nc 0 1\n")
    assert isinstance(t, Tableau)
    assert t.stabilizers() == ["+XX", "+ZZ"]


def test_cli(tmp_path):
    default_target = pathlib.Path(__file__).resolve().parent.parent / "target"
    target = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", default_target))
    exe = target / "release" / ("chp.exe" if sys.platform == "win32" else "chp")
    if not exe.exists():
        pytest.skip("chp binary not built (cargo build --release)")
    out = subprocess.run(
        [str(exe), "-s", "1", str(EXAMPLES / "dense_coding.chp")],
        capture_output=True, text=True, check=True,
    ).stdout
    assert out.splitlines() == [
        "Outcome of measuring qubit 0: 1 (determinate)",
        "Outcome of measuring qubit 1: 1 (determinate)",
    ]


def test_circuit_inverse_undoes_circuit():
    rng = random.Random(0)
    c = Circuit(4)
    for _ in range(50):
        kind = rng.choice("chp")
        a = rng.randrange(4)
        if kind == "c":
            c.cnot(a, (a + 1 + rng.randrange(3)) % 4)
        elif kind == "h":
            c.h(a)
        else:
            c.p(a)
    both = Circuit(4)
    both.extend(c)
    both.extend(c.inverse())
    assert both.tableau() == Tableau(4)
