#!/usr/bin/env python3
"""
generate_rref_batch_test_cases.py

Generates rref_batch_test_cases.txt for test_rref_batch_vs_python.cpp.

Each test runs IncrementalRREFBatch.add_columns (Python reference), records
the resulting state, and writes it so the C++ test can replay the same
ADD_BATCH operations and verify bit-for-bit identity.

Sections
--------
S1  : Single-column add_columns — equivalent to sequential add_column.
S2a : All-at-once multi-column batch.
S2b : Multi-column split into contiguous groups.
S2c : Null-space capture (dependent columns inside a batch).
S2d : Block-diagonal optimisation (batch touches only new check rows).
S2f : Varying-length columns within one batch call.
"""

import sys
import os
import numpy as np

# Reach algorithms/ from src/test_accross_python_cplus/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from incremental_rref import IncrementalRREF
from incremental_rref_batch import IncrementalRREFBatch

OUTPUT = os.path.join(os.path.dirname(__file__), 'rref_batch_test_cases.txt')


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _bytes(arr):
    return ' '.join(str(int(x)) for x in arr)


def write_single_op(f, col, s_extra=None):
    """OP ADD_BATCH 1 <len> <bytes> SEXTRA/NO_SEXTRA"""
    if s_extra is not None:
        f.write(f"OP ADD_BATCH 1  {len(col)} {_bytes(col)}"
                f"  SEXTRA {len(s_extra)} {_bytes(s_extra)}\n")
    else:
        f.write(f"OP ADD_BATCH 1  {len(col)} {_bytes(col)}  NO_SEXTRA\n")


def write_batch_op(f, columns, s_extra=None):
    """OP ADD_BATCH k  <len0> <bytes0>  ...  SEXTRA/NO_SEXTRA"""
    k = len(columns)
    col_parts = '  '.join(f"{len(c)} {_bytes(c)}" for c in columns)
    if s_extra is not None:
        f.write(f"OP ADD_BATCH {k}  {col_parts}"
                f"  SEXTRA {len(s_extra)} {_bytes(s_extra)}\n")
    else:
        f.write(f"OP ADD_BATCH {k}  {col_parts}  NO_SEXTRA\n")


def write_state(f, rref):
    pm = [-1 if p is None else p for p in rref.pivot_map]
    sp = [int(x) for x in rref.s_prime]
    f.write(f"NCHECKS {rref.n_checks}\n")
    f.write(f"NBITS {rref.n_bits}\n")
    f.write(f"PIVOT_MAP {' '.join(map(str, pm))}\n" if pm else "PIVOT_MAP\n")
    f.write(f"SPRIME {' '.join(map(str, sp))}\n" if sp else "SPRIME\n")
    f.write(f"ZCOUNT {len(rref.Z)}\n")
    for z in rref.Z:
        f.write(f"Z {_bytes(z)}\n")
    f.write(f"ISVALID {1 if rref.is_valid() else 0}\n")


def begin_test(f, name, comment=''):
    f.write(f"\nTEST {name}\n")
    if comment:
        f.write(f"# {comment}\n")


def end_test(f, rref, ref=None):
    """Write state and ENDTEST.  If ref is given, assert field-identity first."""
    if ref is not None:
        assert rref.n_checks == ref.n_checks,   f"n_checks mismatch"
        assert rref.n_bits   == ref.n_bits,     f"n_bits mismatch"
        assert list(rref.s_prime) == list(ref.s_prime), f"s_prime mismatch"
        assert rref.pivot_map == ref.pivot_map, f"pivot_map mismatch"
        assert len(rref.Z) == len(ref.Z),       f"|Z| mismatch"
        for k, (zb, zr) in enumerate(zip(rref.Z, ref.Z)):
            assert np.array_equal(zb, zr),      f"Z[{k}] mismatch"
    write_state(f, rref)
    f.write("ENDTEST\n")


# ---------------------------------------------------------------------------
# Section 1 — Single-column add_columns
# (equivalent to sequential add_column; field-identity verified internally)
# ---------------------------------------------------------------------------

def gen_s1_identity_3x3(f):
    H = np.eye(3, dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    begin_test(f, "s1_identity_3x3",
               "3x3 identity H; single-col batches; 3 independent cols")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    for j in range(3):
        se = s if j == 0 else None
        write_single_op(f, H[:, j], se)
        rref.add_columns([H[:, j]], s_extra=se)
        if se is not None:
            ref.add_column(H[:, j], s)
        else:
            ref.add_column(H[:, j])
    end_test(f, rref, ref)


def gen_s1_with_dependent(f):
    H = np.array([[1, 0, 1],
                  [0, 1, 1],
                  [1, 1, 0]], dtype=np.uint8)
    s = np.array([1, 1, 0], dtype=np.uint8)
    begin_test(f, "s1_with_dependent",
               "rank-2 matrix; col2 = col0^col1; one null vector expected")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    for j in range(3):
        se = s if j == 0 else None
        write_single_op(f, H[:, j], se)
        rref.add_columns([H[:, j]], s_extra=se)
        if se is not None:
            ref.add_column(H[:, j], s)
        else:
            ref.add_column(H[:, j])
    end_test(f, rref, ref)


def gen_s1_block_structure(f):
    """Each column introduces exactly one new check row."""
    begin_test(f, "s1_block_structure",
               "block structure: 4 cols each with one new check row")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    # Col j: length j+1, with h[j]=1 (block diagonal structure → all independent)
    rng = np.random.default_rng(42)
    for j in range(4):
        h = np.zeros(j + 1, dtype=np.uint8)
        h[:j] = rng.integers(0, 2, size=j, dtype=np.uint8)
        h[j] = 1
        se = np.array([int(rng.integers(0, 2))], dtype=np.uint8)
        write_single_op(f, h, se)
        rref.add_columns([h], s_extra=se)
        ref.add_column(h, se)
    end_test(f, rref, ref)


def gen_s1_rep_code(f):
    H = np.array([[1, 1, 0],
                  [0, 1, 1]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    begin_test(f, "s1_rep_code",
               "repetition code: rank=2, 1 null vector, is_valid=True")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    for j in range(3):
        se = s if j == 0 else None
        write_single_op(f, H[:, j], se)
        rref.add_columns([H[:, j]], s_extra=se)
        if se is not None:
            ref.add_column(H[:, j], s)
        else:
            ref.add_column(H[:, j])
    end_test(f, rref, ref)


def gen_s1_mixed_inserts(f):
    """Columns with progressively growing lengths (mixed same-check and new-check)."""
    begin_test(f, "s1_mixed_inserts",
               "3 cols of lengths 2,3,3 with growing check rows")
    rng = np.random.default_rng(7)
    cols = [
        (np.array([1, 0], dtype=np.uint8), np.array([1, 0], dtype=np.uint8)),
        (np.array([0, 1, 1], dtype=np.uint8), np.array([1], dtype=np.uint8)),
        (np.array([1, 1, 0], dtype=np.uint8), None),
    ]
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    for h, se in cols:
        write_single_op(f, h, se)
        rref.add_columns([h], s_extra=se)
        if se is not None:
            ref.add_column(h, se)
        else:
            ref.add_column(h)
    end_test(f, rref, ref)


# ---------------------------------------------------------------------------
# Section 2a — All columns at once (full batch)
# ---------------------------------------------------------------------------

def gen_s2a_full_batch_3_independent(f):
    H = np.array([[1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 1]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    begin_test(f, "s2a_full_batch_3_independent",
               "3 independent cols added all at once; rank=3 no null vectors")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    cols = [H[:, j] for j in range(3)]
    write_batch_op(f, cols, s)
    rref.add_columns(cols, s_extra=s)
    ref.add_column(H[:, 0], s)
    ref.add_column(H[:, 1])
    ref.add_column(H[:, 2])
    end_test(f, rref, ref)


def gen_s2a_full_batch_4cols_2null(f):
    H = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    begin_test(f, "s2a_full_batch_4cols_2null",
               "4 cols: cols 0,1 independent; cols 2,3 dependent -> 2 null vectors")
    ref  = IncrementalRREFBatch()
    ref.add_columns([H[:, j] for j in range(4)], s_extra=s)
    rref = IncrementalRREFBatch()
    cols = [H[:, j] for j in range(4)]
    write_batch_op(f, cols, s)
    rref.add_columns(cols, s_extra=s)
    # Both are batch; compare final state
    end_test(f, rref, ref)


def gen_s2a_existing_then_full_batch(f):
    """Establish 2 cols first, then add 3 more as a full batch."""
    begin_test(f, "s2a_existing_then_full_batch",
               "2 single-col ops, then batch of 3; field-identical to 5 sequential")
    H = np.array([[1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)

    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()

    # First two cols individually
    write_single_op(f, H[:, 0], s)
    write_single_op(f, H[:, 1])
    rref.add_columns([H[:, 0]], s_extra=s)
    rref.add_columns([H[:, 1]])
    ref.add_column(H[:, 0], s)
    ref.add_column(H[:, 1])

    # Remaining three as a batch
    cols = [H[:, j] for j in range(2, 5)]
    write_batch_op(f, cols)
    rref.add_columns(cols)
    for j in range(2, 5):
        ref.add_column(H[:, j])

    end_test(f, rref, ref)


def gen_s2a_k8_batch(f):
    """8 columns at once — exercises the k=8 path."""
    begin_test(f, "s2a_k8_batch",
               "8 independent columns added in one batch; tests k=8 code path")
    rng = np.random.default_rng(99)
    n = 8
    # Build a lower-triangular matrix so all columns are independent
    H = np.zeros((n, n), dtype=np.uint8)
    for j in range(n):
        H[j, j] = 1
        H[:j, j] = rng.integers(0, 2, size=j, dtype=np.uint8)
    s = rng.integers(0, 2, size=n, dtype=np.uint8)

    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    cols = [H[:, j] for j in range(n)]
    write_batch_op(f, cols, s)
    rref.add_columns(cols, s_extra=s)
    ref.add_column(H[:, 0], s)
    for j in range(1, n):
        ref.add_column(H[:, j])
    end_test(f, rref, ref)


# ---------------------------------------------------------------------------
# Section 2b — Random batch grouping
# ---------------------------------------------------------------------------

def gen_s2b_two_groups(f):
    H = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 1],
                  [1, 0, 1, 0]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    begin_test(f, "s2b_two_groups",
               "4 cols split as batch[2] + batch[2]; field-identical to sequential")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()

    g1 = [H[:, 0], H[:, 1]]
    g2 = [H[:, 2], H[:, 3]]

    write_batch_op(f, g1, s)
    rref.add_columns(g1, s_extra=s)
    ref.add_column(H[:, 0], s)
    ref.add_column(H[:, 1])

    write_batch_op(f, g2)
    rref.add_columns(g2)
    ref.add_column(H[:, 2])
    ref.add_column(H[:, 3])

    end_test(f, rref, ref)


def gen_s2b_three_groups(f):
    begin_test(f, "s2b_three_groups",
               "6 cols split as batch[2]+batch[3]+batch[1]; field-identical")
    rng = np.random.default_rng(55)
    H = rng.integers(0, 2, size=(4, 6), dtype=np.uint8)
    s = rng.integers(0, 2, size=4, dtype=np.uint8)

    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    groups = [list(range(0, 2)), list(range(2, 5)), list(range(5, 6))]

    first = True
    for g in groups:
        cols = [H[:, j] for j in g]
        se   = s if first else None
        write_batch_op(f, cols, se)
        rref.add_columns(cols, s_extra=se)
        for j in g:
            if first:
                ref.add_column(H[:, j], s)
                first = False
            else:
                ref.add_column(H[:, j])
        first = False

    end_test(f, rref, ref)


# ---------------------------------------------------------------------------
# Section 2c — Null-space capture
# ---------------------------------------------------------------------------

def gen_s2c_explicit_dependent_in_batch(f):
    H = np.array([[1, 0, 1],
                  [0, 1, 0]], dtype=np.uint8)
    s = np.array([1, 1], dtype=np.uint8)
    begin_test(f, "s2c_explicit_dependent_in_batch",
               "col2=col0 (dependent); 1 null vector; batch and sequential match")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    cols = [H[:, j] for j in range(3)]
    write_batch_op(f, cols, s)
    rref.add_columns(cols, s_extra=s)
    ref.add_column(H[:, 0], s)
    ref.add_column(H[:, 1])
    ref.add_column(H[:, 2])
    end_test(f, rref, ref)


def gen_s2c_all_same_columns(f):
    """All 4 columns identical — rank 1, 3 null vectors."""
    h = np.array([1, 1, 1], dtype=np.uint8)
    s = np.array([1, 1, 1], dtype=np.uint8)
    begin_test(f, "s2c_all_same_columns",
               "4 identical cols; rank=1; 3 null vectors; is_valid depends on s")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    cols = [h.copy() for _ in range(4)]
    write_batch_op(f, cols, s)
    rref.add_columns(cols, s_extra=s)
    ref.add_column(h, s)
    for _ in range(3):
        ref.add_column(h)
    end_test(f, rref, ref)


def gen_s2c_alternating_dependent(f):
    H = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    begin_test(f, "s2c_alternating_dependent",
               "[1,0],[0,1],[1,0],[0,1]: rank=2, 2 null vectors")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    cols = [H[:, j] for j in range(4)]
    write_batch_op(f, cols, s)
    rref.add_columns(cols, s_extra=s)
    ref.add_column(H[:, 0], s)
    for j in range(1, 4):
        ref.add_column(H[:, j])
    end_test(f, rref, ref)


def gen_s2c_mixed_dep_in_larger_batch(f):
    """5-col batch: cols 0,1,3 independent; cols 2,4 dependent."""
    H = np.array([[1, 0, 1, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1]], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    begin_test(f, "s2c_mixed_dep_in_larger_batch",
               "5 cols; rank=3; 2 null vectors (cols 2 and 4 dependent)")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    cols = [H[:, j] for j in range(5)]
    write_batch_op(f, cols, s)
    rref.add_columns(cols, s_extra=s)
    ref.add_column(H[:, 0], s)
    for j in range(1, 5):
        ref.add_column(H[:, j])
    end_test(f, rref, ref)


# ---------------------------------------------------------------------------
# Section 2d — Block-diagonal optimisation
# ---------------------------------------------------------------------------

def gen_s2d_explicit_block_diagonal(f):
    """Deterministic example from the Python tests."""
    H_init  = np.eye(2, dtype=np.uint8)
    s_init  = np.array([1, 0], dtype=np.uint8)
    H_batch = np.array([[0, 0, 0],
                         [0, 0, 0],
                         [1, 0, 1],
                         [0, 1, 1]], dtype=np.uint8)
    s_extra = np.array([1, 1], dtype=np.uint8)

    begin_test(f, "s2d_explicit_block_diagonal",
               "2 existing rows then batch of 3 new-row-only cols; opt fires")

    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()

    write_single_op(f, H_init[:, 0], s_init)
    write_single_op(f, H_init[:, 1])
    rref.add_columns([H_init[:, 0]], s_extra=s_init)
    rref.add_columns([H_init[:, 1]])
    ref.add_column(H_init[:, 0], s_init)
    ref.add_column(H_init[:, 1])

    cols = [H_batch[:, j] for j in range(3)]
    write_batch_op(f, cols, s_extra)
    rref.add_columns(cols, s_extra=s_extra)
    ref.add_column(H_batch[:, 0], s_extra)
    ref.add_column(H_batch[:, 1])
    ref.add_column(H_batch[:, 2])

    end_test(f, rref, ref)


def gen_s2d_block_diagonal_random(f):
    """Random example: n_old=3 existing rows, then batch of 4 new-only cols."""
    rng    = np.random.default_rng(77)
    n_old  = 3
    n_new  = 4
    n_cols = 4

    H_init  = np.eye(n_old, dtype=np.uint8)
    s_init  = rng.integers(0, 2, size=n_old, dtype=np.uint8)

    # Batch: zero in top n_old rows, random in new rows
    H_batch = np.vstack([
        np.zeros((n_old, n_cols), dtype=np.uint8),
        rng.integers(0, 2, size=(n_new, n_cols), dtype=np.uint8),
    ])
    s_extra = rng.integers(0, 2, size=n_new, dtype=np.uint8)

    begin_test(f, "s2d_block_diagonal_random",
               f"n_old={n_old} existing rows; batch of {n_cols} new-only cols; opt fires")

    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()

    for j in range(n_old):
        se = s_init if j == 0 else None
        write_single_op(f, H_init[:, j], se)
        rref.add_columns([H_init[:, j]], s_extra=se)
        if se is not None:
            ref.add_column(H_init[:, j], se)
        else:
            ref.add_column(H_init[:, j])

    cols = [H_batch[:, j] for j in range(n_cols)]
    write_batch_op(f, cols, s_extra)
    rref.add_columns(cols, s_extra=s_extra)
    ref.add_column(H_batch[:, 0], s_extra)
    for j in range(1, n_cols):
        ref.add_column(H_batch[:, j])

    end_test(f, rref, ref)


# ---------------------------------------------------------------------------
# Section 2f — Varying-length columns
# ---------------------------------------------------------------------------

def gen_s2f_explicit_varying_lengths(f):
    """Deterministic: 2 existing rows; batch cols of lengths 4, 3, 2."""
    H_init  = np.eye(2, dtype=np.uint8)
    s_init  = np.array([1, 0], dtype=np.uint8)
    col0    = np.array([1, 0, 1, 1], dtype=np.uint8)
    col1    = np.array([0, 1, 0],    dtype=np.uint8)
    col2    = np.array([1, 1],       dtype=np.uint8)
    s_extra = np.array([1, 0],       dtype=np.uint8)

    begin_test(f, "s2f_explicit_varying_lengths",
               "cols of len 4,3,2 — shorter zero-padded internally")

    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()

    write_single_op(f, H_init[:, 0], s_init)
    write_single_op(f, H_init[:, 1])
    rref.add_columns([H_init[:, 0]], s_extra=s_init)
    rref.add_columns([H_init[:, 1]])
    ref.add_column(H_init[:, 0], s_init)
    ref.add_column(H_init[:, 1])

    write_batch_op(f, [col0, col1, col2], s_extra)
    rref.add_columns([col0, col1, col2], s_extra=s_extra)
    n_total = 4
    ref.add_column(col0, s_extra)
    ref.add_column(np.pad(col1, (0, n_total - len(col1))))
    ref.add_column(np.pad(col2, (0, n_total - len(col2))))

    end_test(f, rref, ref)


def gen_s2f_varying_lengths_random(f):
    """Random: 2 existing rows; batch of 5 cols with random lengths."""
    rng       = np.random.default_rng(33)
    n_exist   = 2
    n_new     = 3
    n_total   = n_exist + n_new
    n_cols    = 5

    H_init    = np.eye(n_exist, dtype=np.uint8)
    s_init    = rng.integers(0, 2, size=n_exist, dtype=np.uint8)

    lengths   = rng.integers(n_exist, n_total + 1, size=n_cols).tolist()
    lengths[0] = n_total   # guarantee longest
    cols_raw  = [rng.integers(0, 2, size=int(l), dtype=np.uint8) for l in lengths]
    s_extra   = rng.integers(0, 2, size=n_new, dtype=np.uint8)

    begin_test(f, "s2f_varying_lengths_random",
               "random col lengths [2..5], longest=5; zero-padding verified")

    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()

    for j in range(n_exist):
        se = s_init if j == 0 else None
        write_single_op(f, H_init[:, j], se)
        rref.add_columns([H_init[:, j]], s_extra=se)
        if se is not None:
            ref.add_column(H_init[:, j], se)
        else:
            ref.add_column(H_init[:, j])

    write_batch_op(f, cols_raw, s_extra)
    rref.add_columns(cols_raw, s_extra=s_extra)
    ref.add_column(cols_raw[0], s_extra)
    for c in cols_raw[1:]:
        ref.add_column(np.pad(c, (0, n_total - len(c))))

    end_test(f, rref, ref)


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------

def gen_extra_batch_then_batch(f):
    """Two sequential batch calls, each with k=2 columns."""
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [1, 1, 1, 0],
                  [0, 0, 1, 1]], dtype=np.uint8)
    s = np.array([1, 0, 1, 1], dtype=np.uint8)
    begin_test(f, "extra_batch_then_batch",
               "two consecutive batch[2] calls; field-identical to 4 sequential")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()
    g1   = [H[:, 0], H[:, 1]]
    g2   = [H[:, 2], H[:, 3]]

    write_batch_op(f, g1, s)
    rref.add_columns(g1, s_extra=s)
    ref.add_column(H[:, 0], s)
    ref.add_column(H[:, 1])

    write_batch_op(f, g2)
    rref.add_columns(g2)
    ref.add_column(H[:, 2])
    ref.add_column(H[:, 3])

    end_test(f, rref, ref)


def gen_extra_all_dependent_batch(f):
    """Batch where every column is dependent — all go to null space."""
    h_base = np.array([1, 0, 1], dtype=np.uint8)
    s      = np.array([1, 0, 1], dtype=np.uint8)
    begin_test(f, "extra_all_dependent_batch",
               "1 independent col + batch of 3 copies (all dependent)")
    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()

    # Establish the single independent column
    write_single_op(f, h_base, s)
    rref.add_columns([h_base], s_extra=s)
    ref.add_column(h_base, s)

    # Now add 3 identical (dependent) columns as a batch
    cols = [h_base.copy() for _ in range(3)]
    write_batch_op(f, cols)
    rref.add_columns(cols)
    for _ in range(3):
        ref.add_column(h_base)

    end_test(f, rref, ref)


def gen_extra_is_valid_false(f):
    """Syndrome not in image — is_valid must be False."""
    H = np.array([[1, 0],
                  [0, 1]], dtype=np.uint8)
    s = np.array([1, 1, 1], dtype=np.uint8)   # 3rd row will have no pivot
    H3 = np.zeros((3, 2), dtype=np.uint8)
    H3[:2, :] = H
    begin_test(f, "extra_is_valid_false",
               "syndrome with a zero-row RREF entry: is_valid=False")
    rref = IncrementalRREFBatch()
    cols = [H3[:, 0], H3[:, 1]]
    write_batch_op(f, cols, s)
    rref.add_columns(cols, s_extra=s)
    end_test(f, rref)


def gen_extra_large_random(f):
    """Larger random test: 10 checks, 12 faults, batch of 6."""
    rng = np.random.default_rng(888)
    n   = 10
    m   = 12
    H   = rng.integers(0, 2, size=(n, m), dtype=np.uint8)
    s   = rng.integers(0, 2, size=n, dtype=np.uint8)
    begin_test(f, "extra_large_random",
               f"n={n} checks, m={m} faults; first 6 sequential, last 6 as batch")

    ref  = IncrementalRREF()
    rref = IncrementalRREFBatch()

    for j in range(6):
        se = s if j == 0 else None
        write_single_op(f, H[:, j], se)
        rref.add_columns([H[:, j]], s_extra=se)
        if se is not None:
            ref.add_column(H[:, j], se)
        else:
            ref.add_column(H[:, j])

    cols = [H[:, j] for j in range(6, 12)]
    write_batch_op(f, cols)
    rref.add_columns(cols)
    for j in range(6, 12):
        ref.add_column(H[:, j])

    end_test(f, rref, ref)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = [
    # Section 1
    gen_s1_identity_3x3,
    gen_s1_with_dependent,
    gen_s1_block_structure,
    gen_s1_rep_code,
    gen_s1_mixed_inserts,
    # Section 2a
    gen_s2a_full_batch_3_independent,
    gen_s2a_full_batch_4cols_2null,
    gen_s2a_existing_then_full_batch,
    gen_s2a_k8_batch,
    # Section 2b
    gen_s2b_two_groups,
    gen_s2b_three_groups,
    # Section 2c
    gen_s2c_explicit_dependent_in_batch,
    gen_s2c_all_same_columns,
    gen_s2c_alternating_dependent,
    gen_s2c_mixed_dep_in_larger_batch,
    # Section 2d
    gen_s2d_explicit_block_diagonal,
    gen_s2d_block_diagonal_random,
    # Section 2f
    gen_s2f_explicit_varying_lengths,
    gen_s2f_varying_lengths_random,
    # Additional
    gen_extra_batch_then_batch,
    gen_extra_all_dependent_batch,
    gen_extra_is_valid_false,
    gen_extra_large_random,
]


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else OUTPUT
    with open(out, 'w') as f:
        f.write("# rref_batch_test_cases.txt\n")
        f.write("# Auto-generated by generate_rref_batch_test_cases.py\n")
        f.write("# Each ADD_BATCH op is replayed by test_rref_batch_vs_python.cpp\n")
        f.write("# and the resulting C++ state is compared bit-for-bit against\n")
        f.write("# the Python IncrementalRREFBatch reference recorded here.\n")
        for gen in GENERATORS:
            gen(f)
    print(f"Wrote {len(GENERATORS)} tests to {out}")
