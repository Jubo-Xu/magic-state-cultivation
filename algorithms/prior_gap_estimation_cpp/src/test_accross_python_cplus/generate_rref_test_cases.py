#!/usr/bin/env python3
"""
Generate cross-language test cases for IncrementalRREF.

Runs Python IncrementalRREF through a variety of operations and writes
the inputs + expected outputs to a text file.  The companion C++ test
(test_rref_vs_python.cpp) reads this file and verifies that its own
IncrementalRREF produces byte-identical results.

Usage (from this directory):
    python generate_rref_test_cases.py [output_file]
    # Default output: rref_test_cases.txt

File format
-----------
Each test is delimited by TEST <name> ... ENDTEST.

Inside a plain (non-merge) test:
    OP ADD_EXTRA <h_len> <h0 h1...> <s_len> <s0 s1...>
    OP ADD       <h_len> <h0 h1...>
    OP SET_S     <n> <v0 v1...>   # set syndrome, recompute s_prime = T @ s

Inside a merge test:
    C1 ADD_EXTRA <h_len> <h...> <s_len> <s...>
    C1 ADD       <h_len> <h...>
    C2 ADD_EXTRA <h_len> <h...> <s_len> <s...>
    C2 ADD       <h_len> <h...>
    CONNECT      <h_len> <h...>

Expected state (appears once before ENDTEST):
    NCHECKS  <n>
    NBITS    <n>
    PIVOT_MAP <p0 p1...>   # -1 means no pivot
    SPRIME   <v0 v1...>
    ZCOUNT   <k>
    Z        <z0 z1...>    # one line per null vector  (ZCOUNT lines)
    ISVALID  <0|1>
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Locate the Python IncrementalRREF
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_algo = os.path.normpath(os.path.join(_here, "..", "..", "..", "..", "algorithms"))
sys.path.insert(0, _algo)
from incremental_rref import IncrementalRREF  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _v(arr):
    """Format a 1-D integer array as a space-separated string."""
    return " ".join(str(int(x)) for x in arr)


def write_state(f, rref: IncrementalRREF) -> None:
    """Emit the expected state of *rref* to file *f*."""
    f.write(f"NCHECKS {rref.n_checks}\n")
    f.write(f"NBITS {rref.n_bits}\n")
    pm = " ".join(str(-1 if p is None else int(p)) for p in rref.pivot_map)
    f.write(f"PIVOT_MAP {pm}\n")
    f.write(f"SPRIME {_v(rref.s_prime)}\n")
    f.write(f"ZCOUNT {len(rref.Z)}\n")
    for z in rref.Z:
        f.write(f"Z {_v(z)}\n")
    f.write(f"ISVALID {1 if rref.is_valid() else 0}\n")


def emit_add_extra(f, rref: IncrementalRREF, h, s_extra) -> None:
    h = np.asarray(h, dtype=np.uint8)
    s_extra = np.asarray(s_extra, dtype=np.uint8)
    f.write(f"OP ADD_EXTRA {len(h)} {_v(h)} {len(s_extra)} {_v(s_extra)}\n")
    rref.add_column(h, s_extra)


def emit_add(f, rref: IncrementalRREF, h) -> None:
    h = np.asarray(h, dtype=np.uint8)
    f.write(f"OP ADD {len(h)} {_v(h)}\n")
    rref.add_column(h)


def emit_set_s(f, rref: IncrementalRREF, s) -> None:
    """Set syndrome and recompute s_prime = T @ s."""
    s = np.asarray(s, dtype=np.uint8)
    f.write(f"OP SET_S {len(s)} {_v(s)}\n")
    rref.s = s.copy()
    rref.s_prime = (rref.T @ s) % 2


def emit_check_valid(f, rref: IncrementalRREF) -> None:
    f.write(f"ISVALID {1 if rref.is_valid() else 0}\n")


# ---------------------------------------------------------------------------
# Chain-graph repetition code: n faults, n-1 checks
# ---------------------------------------------------------------------------

def gen_rep_code(f, n: int) -> None:
    assert n >= 2
    f.write(f"TEST rep_code_n{n}\n")
    rref = IncrementalRREF()

    # Fault 0 → introduces check 0
    emit_add_extra(f, rref, [1], [0])

    # Faults 1 .. n-2: h[i-1]=1, h[i]=1; introduces check i
    for i in range(1, n - 1):
        h = [0] * (i - 1) + [1, 1]   # length i+1; h[i-1]=1, h[i]=1
        emit_add_extra(f, rref, h, [0])

    # Fault n-1: connects to existing check n-2 only
    h = [0] * (n - 2) + [1]
    emit_add(f, rref, h)

    # Default (zero) syndrome → always valid
    write_state(f, rref)
    f.write("ENDTEST\n\n")


# ---------------------------------------------------------------------------
# Cycle-graph ring code: n faults, n checks  (null_dim = 1)
# ---------------------------------------------------------------------------

def gen_ring_code(f, n: int) -> None:
    assert n >= 3
    f.write(f"TEST ring_code_n{n}\n")
    rref = IncrementalRREF()

    # Fault 0 → checks 0 and 1, both new
    emit_add_extra(f, rref, [1, 1], [0, 0])

    # Faults 1 .. n-2: h[i]=1 (existing), h[i+1]=1 (new)
    for i in range(1, n - 1):
        h = [0] * i + [1, 1]          # length i+2; existing check i, new check i+1
        emit_add_extra(f, rref, h, [0])

    # Fault n-1: connects existing checks 0 and n-1
    h = [1] + [0] * (n - 2) + [1]    # length n
    emit_add(f, rref, h)

    # Zero syndrome → valid
    write_state(f, rref)
    f.write("ENDTEST\n\n")


def gen_ring_code_invalid_syn(f, n: int) -> None:
    """Ring code with the all-ones syndrome (invalid when n is odd)."""
    assert n >= 3
    f.write(f"TEST ring_code_n{n}_allones_syn\n")
    rref = IncrementalRREF()

    emit_add_extra(f, rref, [1, 1], [0, 0])
    for i in range(1, n - 1):
        h = [0] * i + [1, 1]
        emit_add_extra(f, rref, h, [0])
    h = [1] + [0] * (n - 2) + [1]
    emit_add(f, rref, h)

    # Set syndrome to all-ones then record state
    emit_set_s(f, rref, [1] * n)
    write_state(f, rref)
    f.write("ENDTEST\n\n")


# ---------------------------------------------------------------------------
# Dependent-column test: H = [[1,0,1],[0,1,1]], fault2 = fault0 XOR fault1
# ---------------------------------------------------------------------------

def gen_dependent_column(f) -> None:
    f.write("TEST dependent_column\n")
    rref = IncrementalRREF()

    emit_add_extra(f, rref, [1], [0])   # fault 0: check 0 new
    emit_add_extra(f, rref, [0, 1], [0])  # fault 1: check 0 existing (h[0]=0), check 1 new
    emit_add(f, rref, [1, 1])            # fault 2: checks 0 and 1 existing → dependent

    write_state(f, rref)
    f.write("ENDTEST\n\n")


# ---------------------------------------------------------------------------
# Random GF(2) matrices
# ---------------------------------------------------------------------------

def gen_random(f, rng: np.random.Generator, idx: int) -> None:
    """
    Build a random GF(2) parity-check matrix incrementally.

    Strategy:
    - Choose random m (n_checks) and n (n_bits) in [2, 8].
    - First column always introduces all m checks at once.
    - Remaining n-1 columns each connect to existing checks only (length = m).
    - Then set syndrome to H@e (valid) and to a random vector (possibly invalid).
    """
    m = int(rng.integers(2, 9))  # n_checks
    n = int(rng.integers(2, 9))  # n_bits

    f.write(f"TEST random_{idx:02d}_m{m}_n{n}\n")
    rref = IncrementalRREF()

    # Column 0: length m+1 (first entry + m new checks)
    # Introduce all m checks at once via a single column whose first entry is 1.
    first_col = rng.integers(0, 2, size=m, dtype=np.uint8)
    first_col[0] = 1  # ensure at least one connection
    s_init = rng.integers(0, 2, size=m, dtype=np.uint8)
    emit_add_extra(f, rref, first_col, s_init)

    # Columns 1..n-1: connect to existing checks only
    for _ in range(n - 1):
        col = rng.integers(0, 2, size=m, dtype=np.uint8)
        emit_add(f, rref, col)

    # Test: set syndrome to H@e (must be valid)
    e = rng.integers(0, 2, size=rref.n_bits, dtype=np.uint8)
    s_valid = (rref.H @ e) % 2
    emit_set_s(f, rref, s_valid)
    f.write(f"CHECK_ISVALID 1\n")

    # Test: random syndrome (valid iff in col(H))
    s_rand = rng.integers(0, 2, size=rref.n_checks, dtype=np.uint8)
    sp_rand = (rref.T @ s_rand) % 2
    expected_valid = all(
        sp_rand[i] == 0
        for i in range(rref.n_checks)
        if rref.pivot_map[i] is None
    )
    emit_set_s(f, rref, s_rand)
    f.write(f"CHECK_ISVALID {1 if expected_valid else 0}\n")

    # Final state (after all SET_S ops)
    write_state(f, rref)
    f.write("ENDTEST\n\n")


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------

def gen_merge_no_connect(f) -> None:
    """Merge two independent clusters with zero connecting edges."""
    f.write("TEST merge_no_connect\n")
    c1 = IncrementalRREF()
    c2 = IncrementalRREF()

    # c1: rep code n=3 (2 checks, 3 faults, null_dim=1)
    f.write("C1 ADD_EXTRA 1 1 1 0\n"); c1.add_column(np.array([1], dtype=np.uint8), np.array([0], dtype=np.uint8))
    f.write("C1 ADD_EXTRA 2 1 1 1 0\n"); c1.add_column(np.array([1, 1], dtype=np.uint8), np.array([0], dtype=np.uint8))
    f.write("C1 ADD 2 0 1\n"); c1.add_column(np.array([0, 1], dtype=np.uint8))

    # c2: single-fault cluster (1 check, 1 fault)
    f.write("C2 ADD_EXTRA 1 1 1 0\n"); c2.add_column(np.array([1], dtype=np.uint8), np.array([0], dtype=np.uint8))

    merged = IncrementalRREF.merge(c1, c2, [])
    # Write expected state
    f.write(f"NCHECKS {merged.n_checks}\n")
    f.write(f"NBITS {merged.n_bits}\n")
    pm = " ".join(str(-1 if p is None else int(p)) for p in merged.pivot_map)
    f.write(f"PIVOT_MAP {pm}\n")
    f.write(f"SPRIME {_v(merged.s_prime)}\n")
    f.write(f"ZCOUNT {len(merged.Z)}\n")
    for z in merged.Z:
        f.write(f"Z {_v(z)}\n")
    f.write(f"ISVALID {1 if merged.is_valid() else 0}\n")
    f.write("ENDTEST\n\n")


def gen_merge_one_connect(f) -> None:
    """Two rep-2 clusters joined by a single connecting fault."""
    f.write("TEST merge_one_connect\n")
    c1 = IncrementalRREF()
    c2 = IncrementalRREF()

    # c1: 2 faults, 2 checks
    f.write("C1 ADD_EXTRA 2 1 1 2 0 0\n"); c1.add_column(np.array([1, 1], dtype=np.uint8), np.array([0, 0], dtype=np.uint8))
    # c2: 2 faults, 2 checks
    f.write("C2 ADD_EXTRA 2 1 1 2 0 0\n"); c2.add_column(np.array([1, 1], dtype=np.uint8), np.array([0, 0], dtype=np.uint8))

    # Connecting fault: check 1 of c1 and check 0 of c2
    # In merged coords: h = [0, 1, 1, 0]
    h_conn = np.array([0, 1, 1, 0], dtype=np.uint8)
    f.write(f"CONNECT 4 0 1 1 0\n")

    merged = IncrementalRREF.merge(c1, c2, [h_conn])
    f.write(f"NCHECKS {merged.n_checks}\n")
    f.write(f"NBITS {merged.n_bits}\n")
    pm = " ".join(str(-1 if p is None else int(p)) for p in merged.pivot_map)
    f.write(f"PIVOT_MAP {pm}\n")
    f.write(f"SPRIME {_v(merged.s_prime)}\n")
    f.write(f"ZCOUNT {len(merged.Z)}\n")
    for z in merged.Z:
        f.write(f"Z {_v(z)}\n")
    f.write(f"ISVALID {1 if merged.is_valid() else 0}\n")
    f.write("ENDTEST\n\n")


def gen_merge_two_connect(f) -> None:
    """Two single-check clusters joined by two connecting faults → ring of 4."""
    f.write("TEST merge_two_connect\n")
    c1 = IncrementalRREF()
    c2 = IncrementalRREF()

    # c1: 1 fault, 1 check
    f.write("C1 ADD_EXTRA 1 1 1 0\n"); c1.add_column(np.array([1], dtype=np.uint8), np.array([0], dtype=np.uint8))
    # c2: 1 fault, 1 check
    f.write("C2 ADD_EXTRA 1 1 1 0\n"); c2.add_column(np.array([1], dtype=np.uint8), np.array([0], dtype=np.uint8))

    # Two connecting faults, each touching check 0 of c1 and check 0 of c2
    h1 = np.array([1, 1], dtype=np.uint8)
    h2 = np.array([1, 1], dtype=np.uint8)
    f.write("CONNECT 2 1 1\n")
    f.write("CONNECT 2 1 1\n")

    merged = IncrementalRREF.merge(c1, c2, [h1, h2])
    f.write(f"NCHECKS {merged.n_checks}\n")
    f.write(f"NBITS {merged.n_bits}\n")
    pm = " ".join(str(-1 if p is None else int(p)) for p in merged.pivot_map)
    f.write(f"PIVOT_MAP {pm}\n")
    f.write(f"SPRIME {_v(merged.s_prime)}\n")
    f.write(f"ZCOUNT {len(merged.Z)}\n")
    for z in merged.Z:
        f.write(f"Z {_v(z)}\n")
    f.write(f"ISVALID {1 if merged.is_valid() else 0}\n")
    f.write("ENDTEST\n\n")


def gen_merge_with_syndrome(f) -> None:
    """Merge two clusters, each carrying a non-trivial syndrome."""
    f.write("TEST merge_with_syndrome\n")
    c1 = IncrementalRREF()
    c2 = IncrementalRREF()

    # c1: 1 fault, 1 check, syndrome = 1
    f.write("C1 ADD_EXTRA 1 1 1 1\n"); c1.add_column(np.array([1], dtype=np.uint8), np.array([1], dtype=np.uint8))
    # c2: 1 fault, 1 check, syndrome = 1
    f.write("C2 ADD_EXTRA 1 1 1 1\n"); c2.add_column(np.array([1], dtype=np.uint8), np.array([1], dtype=np.uint8))

    # Connecting fault between their checks
    h_conn = np.array([1, 1], dtype=np.uint8)
    f.write("CONNECT 2 1 1\n")

    merged = IncrementalRREF.merge(c1, c2, [h_conn])
    f.write(f"NCHECKS {merged.n_checks}\n")
    f.write(f"NBITS {merged.n_bits}\n")
    pm = " ".join(str(-1 if p is None else int(p)) for p in merged.pivot_map)
    f.write(f"PIVOT_MAP {pm}\n")
    f.write(f"SPRIME {_v(merged.s_prime)}\n")
    f.write(f"ZCOUNT {len(merged.Z)}\n")
    for z in merged.Z:
        f.write(f"Z {_v(z)}\n")
    f.write(f"ISVALID {1 if merged.is_valid() else 0}\n")
    f.write("ENDTEST\n\n")


def gen_merge_large(f) -> None:
    """Merge two rep-3 clusters with a bridging fault."""
    f.write("TEST merge_large\n")
    c1 = IncrementalRREF()
    c2 = IncrementalRREF()

    # c1: rep code n=4 (3 checks, 4 faults)
    f.write("C1 ADD_EXTRA 1 1 1 0\n"); c1.add_column(np.array([1], dtype=np.uint8), np.array([0], dtype=np.uint8))
    f.write("C1 ADD_EXTRA 2 1 1 1 0\n"); c1.add_column(np.array([1, 1], dtype=np.uint8), np.array([0], dtype=np.uint8))
    f.write("C1 ADD_EXTRA 3 0 1 1 1 0\n"); c1.add_column(np.array([0, 1, 1], dtype=np.uint8), np.array([0], dtype=np.uint8))
    f.write("C1 ADD 3 0 0 1\n"); c1.add_column(np.array([0, 0, 1], dtype=np.uint8))

    # c2: rep code n=3 (2 checks, 3 faults)
    f.write("C2 ADD_EXTRA 1 1 1 0\n"); c2.add_column(np.array([1], dtype=np.uint8), np.array([0], dtype=np.uint8))
    f.write("C2 ADD_EXTRA 2 1 1 1 0\n"); c2.add_column(np.array([1, 1], dtype=np.uint8), np.array([0], dtype=np.uint8))
    f.write("C2 ADD 2 0 1\n"); c2.add_column(np.array([0, 1], dtype=np.uint8))

    # Connecting fault: check 2 (last of c1) and check 3 (first of c2 = offset 3)
    h_conn = np.array([0, 0, 1, 1, 0], dtype=np.uint8)
    f.write(f"CONNECT 5 0 0 1 1 0\n")

    merged = IncrementalRREF.merge(c1, c2, [h_conn])
    f.write(f"NCHECKS {merged.n_checks}\n")
    f.write(f"NBITS {merged.n_bits}\n")
    pm = " ".join(str(-1 if p is None else int(p)) for p in merged.pivot_map)
    f.write(f"PIVOT_MAP {pm}\n")
    f.write(f"SPRIME {_v(merged.s_prime)}\n")
    f.write(f"ZCOUNT {len(merged.Z)}\n")
    for z in merged.Z:
        f.write(f"Z {_v(z)}\n")
    f.write(f"ISVALID {1 if merged.is_valid() else 0}\n")
    f.write("ENDTEST\n\n")


# ---------------------------------------------------------------------------
# Syndrome tests: set syndrome after building and test is_valid separately
# ---------------------------------------------------------------------------

def gen_syndrome_tests(f) -> None:
    """Rep code n=5 with various syndromes tested after construction."""
    f.write("TEST rep_code_n5_syndrome_tests\n")
    rref = IncrementalRREF()
    n = 5

    emit_add_extra(f, rref, [1], [0])
    for i in range(1, n - 1):
        h = [0] * (i - 1) + [1, 1]
        emit_add_extra(f, rref, h, [0])
    h = [0] * (n - 2) + [1]
    emit_add(f, rref, h)

    # Syndrome in col(H): H @ [1,0,0,0,0] = first column of H
    e = np.zeros(n, dtype=np.uint8); e[0] = 1
    s_valid = (rref.H @ e) % 2
    emit_set_s(f, rref, s_valid)
    f.write(f"CHECK_ISVALID 1\n")

    # Another syndrome in col(H): H @ [1,1,0,0,0]
    e2 = np.zeros(n, dtype=np.uint8); e2[0] = 1; e2[1] = 1
    s_valid2 = (rref.H @ e2) % 2
    emit_set_s(f, rref, s_valid2)
    f.write(f"CHECK_ISVALID 1\n")

    # Zero syndrome: always valid (final state recorded here)
    emit_set_s(f, rref, np.zeros(n - 1, dtype=np.uint8))
    f.write(f"CHECK_ISVALID 1\n")

    write_state(f, rref)
    f.write("ENDTEST\n\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_here, "rref_test_cases.txt")

    rng = np.random.default_rng(42)

    with open(out_path, "w") as f:
        f.write("# IncrementalRREF cross-language test cases\n")
        f.write("# Generated by generate_rref_test_cases.py\n")
        f.write("# Do NOT edit by hand — regenerate instead.\n\n")

        # Chain-graph repetition codes
        for n in range(2, 9):
            gen_rep_code(f, n)

        # Cycle-graph ring codes (zero syndrome)
        for n in range(3, 9):
            gen_ring_code(f, n)

        # Ring codes with all-ones syndrome
        for n in range(3, 9):
            gen_ring_code_invalid_syn(f, n)

        # Dependent-column test
        gen_dependent_column(f)

        # Syndrome tests
        gen_syndrome_tests(f)

        # Random GF(2) matrices
        for idx in range(25):
            gen_random(f, rng, idx)

        # Merge tests
        gen_merge_no_connect(f)
        gen_merge_one_connect(f)
        gen_merge_two_connect(f)
        gen_merge_with_syndrome(f)
        gen_merge_large(f)

    print(f"Written {out_path}")


if __name__ == "__main__":
    main()
