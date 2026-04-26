#!/usr/bin/env python3
"""
generate_prior_gap_estimator_test_cases.py

Generates prior_gap_estimator_test_cases.txt for
test_prior_gap_estimator_vs_python.cpp.

For each test case: runs ClusteringOvergrowBatch (Python) and the standalone
Python gap functions (which mirror PriorGapEstimatorUse exactly), writes:
  - PCM adjacency data, L matrix, syndrome, run parameters
  - Expected: gap (float), nonzero_count (int), overall_logical_flip (uint8[])

The C++ test replays each case with PriorGapEstimator::execute and asserts
bit-for-bit identity on nonzero_count and flip, and abs tolerance 1e-10 on gap.

Run:
    python generate_prior_gap_estimator_test_cases.py
"""

import sys
import os
import math
import numpy as np

# Make algorithms/ importable.
_alg_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, os.path.abspath(_alg_dir))

from clustering_overgrow_batch import ClusteringOvergrowBatch

OUTPUT = os.path.join(os.path.dirname(__file__),
                      'prior_gap_estimator_test_cases.txt')


# ---------------------------------------------------------------------------
# Minimal PCM stub
# ---------------------------------------------------------------------------

class _PCM:
    """Minimal PCM compatible with ClusteringOvergrowBatch."""

    def __init__(self, H, probs, L=None, n_logical=0):
        self.H = np.asarray(H, dtype=np.uint8)
        self.error_data = [
            {'detectors': frozenset(int(i) for i in np.where(self.H[:, j])[0]),
             'prob': float(probs[j])}
            for j in range(self.H.shape[1])
        ]
        self.n_logical_check_nodes = int(n_logical)
        self.L = np.asarray(L, dtype=np.uint8) if L is not None else None

    @classmethod
    def rep(cls, n, p=0.1):
        H = np.zeros((n - 1, n), dtype=np.uint8)
        for i in range(n - 1):
            H[i, i] = H[i, i + 1] = 1
        return cls(H, [p] * n)

    @classmethod
    def ring(cls, n, p=0.1, L=None, n_logical=0):
        H = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            H[i, i] = H[i, (i + 1) % n] = 1
        return cls(H, [p] * n, L=L, n_logical=n_logical)


def _faults_to_syndrome(H, fired_faults):
    s = np.zeros(H.shape[0], dtype=np.uint8)
    for j in fired_faults:
        s ^= H[:, j]
    return s


# ---------------------------------------------------------------------------
# Reference Python gap functions
# (mirror PriorGapEstimatorUse._gap_* exactly)
# ---------------------------------------------------------------------------

def _get_correction(cl):
    """RREF pivot correction in local fault index space."""
    rref = cl.rref
    corr = np.zeros(rref.n_bits, dtype=np.uint8)
    for i, pm in enumerate(rref.pivot_map):
        if pm is not None and rref.s_prime[i] == 1:
            corr[pm] = 1
    return corr


def _gap_binary(regions):
    for groups in regions.values():
        if groups['logical_error']:
            return float('-inf')
    return float('inf')


def _gap_hamming(regions, aggregate):
    cluster_gaps = []
    for groups in regions.values():
        vecs = groups['logical_error']
        if not vecs:
            continue
        min_hw = min(int(np.sum(z)) for z, *_ in vecs)
        cluster_gaps.append(float(min_hw))
    if not cluster_gaps:
        return float('inf')
    return min(cluster_gaps) if aggregate == 'min' else sum(cluster_gaps)


def _gap_prior(regions, avc, weights, aggregate):
    cluster_gaps = []
    for cl_id, groups in regions.items():
        vecs = groups['logical_error']
        if not vecs:
            continue
        cl = avc[cl_id]
        fm = cl.cluster_fault_idx_to_pcm_fault_idx
        min_pw = min(
            float(sum(weights[fm[j]] for j in range(len(z)) if z[j]))
            for z, *_ in vecs
        )
        cluster_gaps.append(min_pw)
    if not cluster_gaps:
        return float('inf')
    return min(cluster_gaps) if aggregate == 'min' else sum(cluster_gaps)


def _gap_weight_diff(regions, avc, weights, aggregate, asb):
    cluster_gaps = []
    for cl_id, groups in regions.items():
        vecs = groups['logical_error']
        if not vecs:
            continue
        cl = avc[cl_id]
        fm = cl.cluster_fault_idx_to_pcm_fault_idx
        e1 = _get_correction(cl)
        min_diff = min(
            float(abs(
                sum(weights[fm[j]] for j in range(len(z)) if z[j] and not e1[j])
              - sum(weights[fm[j]] for j in range(len(z)) if z[j] and     e1[j])
            ) if asb else (
                sum(weights[fm[j]] for j in range(len(z)) if z[j] and not e1[j])
              - sum(weights[fm[j]] for j in range(len(z)) if z[j] and     e1[j])
            ))
            for z, *_ in vecs
        )
        cluster_gaps.append(min_diff)
    if not cluster_gaps:
        return float('inf')
    return min(cluster_gaps) if aggregate == 'min' else sum(cluster_gaps)


def _compute_gap(regions, avc, weights, gap_type, aggregate, asb):
    if gap_type == 'binary':
        return _gap_binary(regions)
    if gap_type == 'hamming':
        return _gap_hamming(regions, aggregate)
    if gap_type == 'prior_weight':
        return _gap_prior(regions, avc, weights, aggregate)
    if gap_type == 'weight_diff':
        return _gap_weight_diff(regions, avc, weights, aggregate, asb)
    raise ValueError(f'unknown gap_type {gap_type!r}')


def _overall_flip(avc, n_logical):
    """overall_logical_flip: XOR of per-cluster L @ e1 (mod 2)."""
    flip = np.zeros(n_logical, dtype=np.uint8)
    for cl in avc.values():
        e1 = _get_correction(cl)
        if cl.L is not None:
            cl_flip = (cl.L @ e1) % 2
        else:
            cl_flip = np.zeros(n_logical, dtype=np.uint8)
        flip ^= cl_flip
    return flip


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _ints(seq):
    return ' '.join(str(int(x)) for x in seq)


def _floats(seq):
    return ' '.join(repr(float(x)) for x in seq)


def _floatstr(v):
    if math.isinf(v):
        return 'inf' if v > 0 else '-inf'
    return repr(v)   # exact round-trip for doubles


# ---------------------------------------------------------------------------
# Write one test case
# ---------------------------------------------------------------------------

def write_test(f, name, pcm, syndrome,
               gap_type, aggregate,
               asb=False, decode=False,
               over_grow_step=0, bits_per_step=1):

    c = ClusteringOvergrowBatch(pcm)
    regions = c.run_and_create_degenerate_cycle_regions(
        syndrome, over_grow_step=over_grow_step, bits_per_step=bits_per_step)
    avc     = c.active_valid_clusters
    weights = c.weights

    nonzero_count = sum(len(g['logical_error']) for g in regions.values())
    gap  = _compute_gap(regions, avc, weights, gap_type, aggregate, asb)

    n_logical = pcm.n_logical_check_nodes
    flip = _overall_flip(avc, n_logical) if (decode and n_logical > 0) \
           else np.array([], dtype=np.uint8)

    ctf = c.check_to_faults
    ftc = c.fault_to_checks
    n_det   = pcm.H.shape[0]
    n_fault = pcm.H.shape[1]

    f.write(f'\nTEST {name}\n')
    f.write(f'GAP_TYPE {gap_type}\n')
    f.write(f'AGGREGATE {aggregate}\n')
    f.write(f'ASB {1 if asb else 0}\n')
    f.write(f'DECODE {1 if decode else 0}\n')
    f.write(f'OVER_GROW_STEP {over_grow_step}\n')
    f.write(f'BITS_PER_STEP {bits_per_step}\n')
    f.write(f'NDET {n_det}\n')
    f.write(f'NFAULT {n_fault}\n')
    f.write(f'NLOGICAL {n_logical}\n')

    for i in range(n_det):
        row = ctf[i]
        f.write(f'CTF {len(row)} {_ints(row)}\n' if row else 'CTF 0\n')
    for j in range(n_fault):
        row = ftc[j]
        f.write(f'FTC {len(row)} {_ints(row)}\n' if row else 'FTC 0\n')
    f.write(f'WEIGHTS {_floats(weights)}\n')

    if pcm.L is not None:
        for row in pcm.L:
            f.write(f'L_ROW {_ints(row)}\n')

    f.write(f'SYNDROME {_ints(syndrome)}\n')
    f.write(f'EXPECTED_GAP {_floatstr(gap)}\n')
    f.write(f'EXPECTED_NONZERO {nonzero_count}\n')
    f.write(f'EXPECTED_FLIP {_ints(flip)}\n')
    f.write('ENDTEST\n')


# ---------------------------------------------------------------------------
# Test-case generators
# ---------------------------------------------------------------------------

def gen_tests(f):

    # -----------------------------------------------------------------------
    # Section 1: no logical observables (n_logical=0, L=None)
    # All null-space vectors are stabilisers → gap_binary = +inf, nonzero = 0.
    # -----------------------------------------------------------------------

    rep5 = _PCM.rep(5)
    ring4 = _PCM.ring(4)

    # 1.1 Empty syndrome — no clusters
    for gt in ('binary', 'hamming', 'prior_weight'):
        write_test(f, f's1_empty_{gt}', rep5,
                   np.zeros(4, dtype=np.uint8), gt, 'min')
    write_test(f, 's1_empty_wdiff', rep5,
               np.zeros(4, dtype=np.uint8), 'weight_diff', 'min',
               asb=False, decode=True)

    # 1.2 Rep5 syndrome [1,0,0,1] — two seeds, merge
    syn_rep5 = np.array([1, 0, 0, 1], dtype=np.uint8)
    for gt in ('binary', 'hamming', 'prior_weight'):
        for agg in ('min', 'sum'):
            write_test(f, f's1_rep5_{gt}_{agg}', rep5, syn_rep5, gt, agg)
    write_test(f, 's1_rep5_wdiff_min', rep5, syn_rep5,
               'weight_diff', 'min', asb=False, decode=True)
    write_test(f, 's1_rep5_wdiff_asb', rep5, syn_rep5,
               'weight_diff', 'min', asb=True, decode=True)

    # 1.3 Ring4 — single fault, one cluster with null-space (but no logical since L=None)
    syn_ring4_f0 = _faults_to_syndrome(ring4.H, [0])
    for gt in ('binary', 'hamming', 'prior_weight'):
        write_test(f, f's1_ring4_f0_{gt}', ring4, syn_ring4_f0, gt, 'min')
    write_test(f, 's1_ring4_f0_wdiff', ring4, syn_ring4_f0,
               'weight_diff', 'min', asb=False, decode=True)

    # 1.4 Rep5 with ogs and bps — verify same logic under overgrow
    write_test(f, 's1_rep5_binary_ogs1', rep5, syn_rep5,
               'binary', 'min', over_grow_step=1)
    write_test(f, 's1_rep5_prior_bps3', rep5, syn_rep5,
               'prior_weight', 'min', bits_per_step=3)
    write_test(f, 's1_rep5_wdiff_ogs1_bps2', rep5, syn_rep5,
               'weight_diff', 'min', asb=False, decode=True,
               over_grow_step=1, bits_per_step=2)

    # -----------------------------------------------------------------------
    # Section 2: with logical observables — null-space vectors are logical errors.
    # Ring code: the all-ones vector spans the null-space of H; with L s.t.
    # L @ [1..1] = 1 (mod 2), this vector is a logical error.
    # -----------------------------------------------------------------------

    # Ring4: null-space basis = {[1,1,1,1]}
    # L = [[1,0,0,0]] → L @ [1,1,1,1] = 1 → logical error.
    L4 = [[1, 0, 0, 0]]
    ring4_L = _PCM.ring(4, p=0.1, L=L4, n_logical=1)

    # 2.1 Syndrome from fault 0 — single cluster, logical error in null-space
    syn_r4L_f0 = _faults_to_syndrome(ring4_L.H, [0])
    for gt in ('binary', 'hamming', 'prior_weight'):
        for agg in ('min', 'sum'):
            write_test(f, f's2_ring4L_f0_{gt}_{agg}',
                       ring4_L, syn_r4L_f0, gt, agg)
    write_test(f, 's2_ring4L_f0_wdiff_min',
               ring4_L, syn_r4L_f0, 'weight_diff', 'min',
               asb=False, decode=True)
    write_test(f, 's2_ring4L_f0_wdiff_asb',
               ring4_L, syn_r4L_f0, 'weight_diff', 'min',
               asb=True, decode=True)
    write_test(f, 's2_ring4L_f0_wdiff_sum',
               ring4_L, syn_r4L_f0, 'weight_diff', 'sum',
               asb=False, decode=True)

    # 2.2 decode=True vs decode=False on same syndrome
    write_test(f, 's2_ring4L_f0_binary_decode_false',
               ring4_L, syn_r4L_f0, 'binary', 'min', decode=False)
    write_test(f, 's2_ring4L_f0_binary_decode_true',
               ring4_L, syn_r4L_f0, 'binary', 'min', decode=True)
    write_test(f, 's2_ring4L_f0_hamming_decode_true',
               ring4_L, syn_r4L_f0, 'hamming', 'min', decode=True)

    # 2.3 Faults 0+2 — two seeds that merge into one cluster
    syn_r4L_f02 = _faults_to_syndrome(ring4_L.H, [0, 2])
    for gt in ('binary', 'hamming', 'prior_weight'):
        write_test(f, f's2_ring4L_f02_{gt}', ring4_L, syn_r4L_f02, gt, 'min')
    write_test(f, 's2_ring4L_f02_wdiff',
               ring4_L, syn_r4L_f02, 'weight_diff', 'min',
               asb=False, decode=True)

    # 2.4 Empty syndrome — no clusters, flip should be zeros
    write_test(f, 's2_ring4L_empty_binary_decode',
               ring4_L, np.zeros(4, dtype=np.uint8),
               'binary', 'min', decode=True)

    # 2.5 Ring5 with one logical: null-space basis = {[1,1,1,1,1]}
    # L = [[1,0,0,0,0]] → L @ [1..1] = 1 → logical error.
    L5 = [[1, 0, 0, 0, 0]]
    ring5_L = _PCM.ring(5, p=0.1, L=L5, n_logical=1)
    syn_r5L_f1 = _faults_to_syndrome(ring5_L.H, [1])
    for gt in ('binary', 'hamming', 'prior_weight'):
        write_test(f, f's2_ring5L_f1_{gt}', ring5_L, syn_r5L_f1, gt, 'min')
    write_test(f, 's2_ring5L_f1_wdiff',
               ring5_L, syn_r5L_f1, 'weight_diff', 'min',
               asb=False, decode=True)

    # 2.6 Overgrow cases with logicals
    write_test(f, 's2_ring4L_f0_binary_ogs1',
               ring4_L, syn_r4L_f0, 'binary', 'min', over_grow_step=1)
    write_test(f, 's2_ring4L_f0_prior_bps2',
               ring4_L, syn_r4L_f0, 'prior_weight', 'min', bits_per_step=2)
    write_test(f, 's2_ring4L_f0_wdiff_ogs1_bps2',
               ring4_L, syn_r4L_f0, 'weight_diff', 'min',
               asb=False, decode=True, over_grow_step=1, bits_per_step=2)

    # -----------------------------------------------------------------------
    # Section 3: multiple clusters — each contributes its own gap
    # Two disconnected ring-like sub-codes each with a logical.
    # -----------------------------------------------------------------------

    # Build a PCM with two independent ring-4 blocks sharing an L matrix.
    # Block 0: dets 0-3, faults 0-3 (ring)
    # Block 1: dets 4-7, faults 4-7 (ring)
    # L = [[1,0,0,0, 0,0,0,0],   # observable 0 → fault 0 of block 0
    #       [0,0,0,0, 1,0,0,0]]  # observable 1 → fault 4 of block 1
    H2 = np.zeros((8, 8), dtype=np.uint8)
    for i in range(4):
        H2[i, i]         = 1
        H2[i, (i+1) % 4] = 1
    for i in range(4):
        H2[i+4, i+4]           = 1
        H2[i+4, ((i+1) % 4)+4] = 1
    L2 = np.zeros((2, 8), dtype=np.uint8)
    L2[0, 0] = 1
    L2[1, 4] = 1
    probs2 = [0.1] * 8
    pcm2 = _PCM(H2, probs2, L=L2, n_logical=2)

    # Fire fault 0 (block 0) and fault 4 (block 1) — two independent clusters
    syn2 = _faults_to_syndrome(H2, [0, 4])
    for gt in ('binary', 'hamming', 'prior_weight'):
        for agg in ('min', 'sum'):
            write_test(f, f's3_two_blocks_{gt}_{agg}', pcm2, syn2, gt, agg)
    write_test(f, 's3_two_blocks_wdiff_min',
               pcm2, syn2, 'weight_diff', 'min', asb=False, decode=True)
    write_test(f, 's3_two_blocks_wdiff_sum',
               pcm2, syn2, 'weight_diff', 'sum', asb=False, decode=True)
    write_test(f, 's3_two_blocks_hamming_decode',
               pcm2, syn2, 'hamming', 'min', decode=True)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else OUTPUT
    with open(out, 'w') as f:
        f.write('# prior_gap_estimator_test_cases.txt\n')
        f.write('# Auto-generated by generate_prior_gap_estimator_test_cases.py\n')
        f.write('# Replayed by test_prior_gap_estimator_vs_python.cpp\n')
        gen_tests(f)
    print(f'Wrote test cases to {out}')
