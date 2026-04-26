#!/usr/bin/env python3
"""
generate_clustering_overgrow_batch_test_cases.py

Generates clustering_overgrow_batch_test_cases.txt for
test_clustering_overgrow_batch_vs_python.cpp.

Each test runs ClusteringOvergrowBatch (Python reference) with given
over_grow_step and bits_per_step, then records the complete final state
of every active valid cluster:
  - adjacency data (check_to_faults, fault_to_checks, weights, syndrome)
  - per cluster: fault_nodes, check_nodes, fault_order, check_order,
                 n_checks, n_bits, pivot_map, s_prime, Z, is_valid

The C++ test replays the same run and asserts bit-for-bit identity.

Sections
--------
S1  : over_grow_step=0, bits_per_step=1  — drop-in for Clustering.
S2a : over_grow_step > 0, bits_per_step=1.
S2b : over_grow_step=0,   bits_per_step > 1.
S2c : both over_grow_step > 0 and bits_per_step > 1.
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from clustering_overgrow_batch import ClusteringOvergrowBatch

OUTPUT = os.path.join(os.path.dirname(__file__),
                      'clustering_overgrow_batch_test_cases.txt')


# ---------------------------------------------------------------------------
# Minimal PCM stub (same pattern as test files)
# ---------------------------------------------------------------------------

class _PCM:
    def __init__(self, H: np.ndarray, probs: list):
        self.H = H.astype(np.uint8)
        self.error_data = [
            {'detectors': frozenset(int(i) for i in np.where(H[:, j])[0]),
             'prob': float(probs[j])}
            for j in range(H.shape[1])
        ]
        self.n_logical_check_nodes = 0
        self.L = None

    @classmethod
    def rep(cls, n: int, p: float = 0.1):
        H = np.zeros((n - 1, n), dtype=np.uint8)
        for i in range(n - 1):
            H[i, i] = H[i, i + 1] = 1
        return cls(H, [p] * n)

    @classmethod
    def ring(cls, n: int, p: float = 0.1):
        H = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            H[i, i] = H[i, (i + 1) % n] = 1
        return cls(H, [p] * n)


def _syndrome_from_faults(H, fired):
    s = np.zeros(H.shape[0], dtype=np.uint8)
    for j in fired:
        s ^= H[:, j]
    return s


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _ints(seq):
    return ' '.join(str(int(x)) for x in seq)


def write_pcm(f, clustering):
    """Write adjacency data extracted from the clustering engine."""
    n_det   = clustering.n_det
    n_fault = clustering.n_fault

    f.write(f"NDET {n_det}\n")
    f.write(f"NFAULT {n_fault}\n")

    # check_to_faults — use the exact lists that Clustering built
    for i in range(n_det):
        adj = clustering.check_to_faults[i]
        f.write(f"CTF {len(adj)} {_ints(adj)}\n" if adj else f"CTF 0\n")

    # fault_to_checks
    for j in range(n_fault):
        adj = clustering.fault_to_checks[j]
        f.write(f"FTC {len(adj)} {_ints(adj)}\n" if adj else f"FTC 0\n")

    # weights (log-likelihood ratios)
    f.write(f"WEIGHTS {_ints(clustering.weights)}\n")


def write_syndrome(f, syndrome):
    f.write(f"SYNDROME {_ints(syndrome)}\n")


def write_cluster(f, cl):
    """Write one active valid cluster's expected state."""
    cid = cl.cluster_id
    f.write(f"CLUSTER {cid}\n")

    fault_nodes_sorted = sorted(cl.fault_nodes)
    check_nodes_sorted = sorted(cl.check_nodes)
    fault_order        = list(cl.cluster_fault_idx_to_pcm_fault_idx)
    check_order        = list(cl.cluster_check_idx_to_pcm_check_idx)

    f.write(f"FAULT_NODES {len(fault_nodes_sorted)} {_ints(fault_nodes_sorted)}\n")
    f.write(f"CHECK_NODES {len(check_nodes_sorted)} {_ints(check_nodes_sorted)}\n")
    f.write(f"FAULT_ORDER {len(fault_order)} {_ints(fault_order)}\n")
    f.write(f"CHECK_ORDER {len(check_order)} {_ints(check_order)}\n")

    rref = cl.rref
    pm   = [-1 if p is None else p for p in rref.pivot_map]
    sp   = [int(x) for x in rref.s_prime]

    f.write(f"NCHECKS {rref.n_checks}\n")
    f.write(f"NBITS {rref.n_bits}\n")
    f.write(f"PIVOT_MAP {_ints(pm)}\n" if pm else "PIVOT_MAP\n")
    f.write(f"SPRIME {_ints(sp)}\n" if sp else "SPRIME\n")
    f.write(f"ZCOUNT {len(rref.Z)}\n")
    for z in rref.Z:
        f.write(f"Z {_ints(z)}\n")
    f.write(f"ISVALID {1 if rref.is_valid() else 0}\n")
    f.write("ENDCLUSTER\n")


def write_test(f, name, pcm, syndrome, over_grow_step=0, bits_per_step=1,
               comment=''):
    """Run ClusteringOvergrowBatch and write the full test case."""
    c = ClusteringOvergrowBatch(pcm)
    c.run(syndrome, over_grow_step=over_grow_step, bits_per_step=bits_per_step)

    f.write(f"\nTEST {name}\n")
    if comment:
        f.write(f"# {comment}\n")

    f.write(f"OVER_GROW_STEP {over_grow_step}\n")
    f.write(f"BITS_PER_STEP {bits_per_step}\n")
    write_pcm(f, c)
    write_syndrome(f, syndrome)

    avc = c.active_valid_clusters
    f.write(f"NACTIVE_VALID {len(avc)}\n")
    for cl_id in sorted(avc):
        write_cluster(f, avc[cl_id])

    f.write("ENDTEST\n")


# ---------------------------------------------------------------------------
# Test-case generators
# ---------------------------------------------------------------------------

def gen_tests(f):

    # ------------------------------------------------------------------
    # Section 1: ogs=0, bps=1 — drop-in for Clustering
    # ------------------------------------------------------------------

    # 1. Empty syndrome — no clusters
    pcm = _PCM.rep(5)
    write_test(f, "s1_empty_syndrome", pcm,
               np.zeros(4, dtype=np.uint8), 0, 1,
               "no non-trivial detectors; no clusters")

    # 2. Rep code n=5, two seeds, one merge
    pcm = _PCM.rep(5)
    write_test(f, "s1_rep5_chain", pcm,
               np.array([1,0,0,1], dtype=np.uint8), 0, 1,
               "rep5; seeds at det0 and det3; clusters merge")

    # 3. Rep code n=5, single boundary fault
    pcm = _PCM.rep(5)
    H = pcm.H.copy()
    s = _syndrome_from_faults(H, [0])   # fault 0: det0 only (boundary)
    write_test(f, "s1_rep5_single_boundary", pcm, s, 0, 1,
               "rep5; single boundary fault; one cluster")

    # 4. Ring code n=4, fire faults 0 and 2
    pcm = _PCM.ring(4)
    s = _syndrome_from_faults(pcm.H, [0, 2])
    write_test(f, "s1_ring4_faults_0_2", pcm, s, 0, 1,
               "ring4; fire f0+f2; clusters merge via cycle")

    # 5. Ring code n=5, all faults — achievable syndromes
    pcm = _PCM.ring(5)
    s = _syndrome_from_faults(pcm.H, [0, 1])
    write_test(f, "s1_ring5_faults_0_1", pcm, s, 0, 1,
               "ring5; fire f0+f1")

    # 6. Three-way merge via 3-body hyperedge
    H = np.zeros((3, 1), dtype=np.uint8)
    H[0,0] = H[1,0] = H[2,0] = 1
    pcm = _PCM(H, [0.1])
    write_test(f, "s1_three_way_merge", pcm,
               np.array([1,1,1], dtype=np.uint8), 0, 1,
               "3 seeds; one 3-body fault; all merge to one cluster")

    # 7. Independent disconnected components
    H = np.zeros((4, 2), dtype=np.uint8)
    H[0,0] = H[1,0] = 1
    H[2,1] = H[3,1] = 1
    pcm = _PCM(H, [0.3, 0.3])
    write_test(f, "s1_independent_components", pcm,
               np.array([1,1,1,1], dtype=np.uint8), 0, 1,
               "two disconnected components; two independent clusters")

    # 8. Long chain n=10
    n = 10
    H = np.zeros((n, n-1), dtype=np.uint8)
    for j in range(n-1):
        H[j,j] = H[j+1,j] = 1
    pcm = _PCM(H, [0.1]*(n-1))
    s = np.zeros(n, dtype=np.uint8); s[0] = s[n-1] = 1
    write_test(f, "s1_chain_n10", pcm, s, 0, 1,
               "chain length 10; seeds at both ends; one merged cluster")

    # 9. Star graph: hub-and-spoke
    n_sp = 4
    H = np.zeros((n_sp+1, n_sp), dtype=np.uint8)
    for j in range(n_sp):
        H[0,j] = H[j+1,j] = 1
    pcm = _PCM(H, [0.2]*n_sp)
    s = _syndrome_from_faults(H, list(range(n_sp)))
    write_test(f, "s1_star_graph", pcm, s, 0, 1,
               "star; spokes merge via central hub")

    # 10. Connecting fault with unclaimed check
    H = np.zeros((3, 2), dtype=np.uint8)
    H[0,0] = H[1,0] = 1
    H[0,1] = H[2,1] = H[1,1] = 1
    pcm = _PCM(H, [0.45, 0.1])
    s = _syndrome_from_faults(H, [1])
    write_test(f, "s1_connecting_unclaimed_check", pcm, s, 0, 1,
               "connecting fault introduces unclaimed check as new RREF row")

    # ------------------------------------------------------------------
    # Section 2a: ogs > 0, bps=1
    # ------------------------------------------------------------------

    # 11. Rep5 with ogs=1
    pcm = _PCM.rep(5)
    write_test(f, "s2a_rep5_ogs1", pcm,
               np.array([1,0,0,1], dtype=np.uint8), 1, 1,
               "rep5; over_grow_step=1; cluster keeps growing one extra step")

    # 12. Rep5 with ogs=2
    pcm = _PCM.rep(5)
    write_test(f, "s2a_rep5_ogs2", pcm,
               np.array([1,0,0,1], dtype=np.uint8), 2, 1,
               "rep5; over_grow_step=2")

    # 13. Ring4 with ogs=1
    pcm = _PCM.ring(4)
    s = _syndrome_from_faults(pcm.H, [0, 2])
    write_test(f, "s2a_ring4_ogs1", pcm, s, 1, 1,
               "ring4; over_grow_step=1")

    # 14. Ring4 with ogs=2
    pcm = _PCM.ring(4)
    s = _syndrome_from_faults(pcm.H, [0, 2])
    write_test(f, "s2a_ring4_ogs2", pcm, s, 2, 1,
               "ring4; over_grow_step=2; more faults absorbed")

    # 15. Star with ogs=1
    n_sp = 4
    H = np.zeros((n_sp+1, n_sp), dtype=np.uint8)
    for j in range(n_sp):
        H[0,j] = H[j+1,j] = 1
    pcm = _PCM(H, [0.2]*n_sp)
    s = _syndrome_from_faults(H, list(range(n_sp)))
    write_test(f, "s2a_star_ogs1", pcm, s, 1, 1,
               "star graph; over_grow_step=1")

    # ------------------------------------------------------------------
    # Section 2b: ogs=0, bps > 1
    # ------------------------------------------------------------------

    # 16. Rep5, bps=3
    pcm = _PCM.rep(5)
    write_test(f, "s2b_rep5_bps3", pcm,
               np.array([1,0,0,1], dtype=np.uint8), 0, 3,
               "rep5; bits_per_step=3; terminates with valid clusters")

    # 17. Ring4, bps=2
    pcm = _PCM.ring(4)
    s = _syndrome_from_faults(pcm.H, [0, 2])
    write_test(f, "s2b_ring4_bps2", pcm, s, 0, 2,
               "ring4; bits_per_step=2")

    # 18. Chain n=10, bps=4
    n = 10
    H = np.zeros((n, n-1), dtype=np.uint8)
    for j in range(n-1):
        H[j,j] = H[j+1,j] = 1
    pcm = _PCM(H, [0.1]*(n-1))
    s = np.zeros(n, dtype=np.uint8); s[0] = s[n-1] = 1
    write_test(f, "s2b_chain_n10_bps4", pcm, s, 0, 4,
               "chain n=10; bits_per_step=4; batch free-fault adds")

    # 19. Two colliding clusters at once (bps=2 + two collision faults)
    H = np.zeros((4, 5), dtype=np.uint8)
    H[0,0] = H[1,0] = 1   # f0: det0-det1 (cheap)
    H[2,1] = H[3,1] = 1   # f1: det2-det3 (cheap)
    H[0,2] = H[2,2] = 1   # f2: connect det0-det2
    H[1,3] = H[3,3] = 1   # f3: connect det1-det3
    H[0,4] = H[3,4] = 1   # f4: connect det0-det3
    pcm = _PCM(H, [0.4, 0.4, 0.2, 0.2, 0.15])
    s = _syndrome_from_faults(H, [0, 1, 2, 3])
    write_test(f, "s2b_multi_connect_bps3", pcm, s, 0, 3,
               "two clusters; multiple connecting faults popped in one step")

    # 20. Ring5 with bps=3
    pcm = _PCM.ring(5)
    s = _syndrome_from_faults(pcm.H, [0, 1])
    write_test(f, "s2b_ring5_bps3", pcm, s, 0, 3,
               "ring5; bits_per_step=3")

    # ------------------------------------------------------------------
    # Section 2c: ogs > 0 and bps > 1
    # ------------------------------------------------------------------

    # 21. Rep5, ogs=1, bps=2
    pcm = _PCM.rep(5)
    write_test(f, "s2c_rep5_ogs1_bps2", pcm,
               np.array([1,0,0,1], dtype=np.uint8), 1, 2,
               "rep5; over_grow_step=1; bits_per_step=2")

    # 22. Ring4, ogs=2, bps=3
    pcm = _PCM.ring(4)
    s = _syndrome_from_faults(pcm.H, [0, 2])
    write_test(f, "s2c_ring4_ogs2_bps3", pcm, s, 2, 3,
               "ring4; over_grow_step=2; bits_per_step=3")

    # 23. Chain n=10, ogs=1, bps=4
    n = 10
    H = np.zeros((n, n-1), dtype=np.uint8)
    for j in range(n-1):
        H[j,j] = H[j+1,j] = 1
    pcm = _PCM(H, [0.1]*(n-1))
    s = np.zeros(n, dtype=np.uint8); s[0] = s[n-1] = 1
    write_test(f, "s2c_chain_n10_ogs1_bps4", pcm, s, 1, 4,
               "chain n=10; both overgrow and batch")

    # 24. Star, ogs=1, bps=3
    n_sp = 4
    H = np.zeros((n_sp+1, n_sp), dtype=np.uint8)
    for j in range(n_sp):
        H[0,j] = H[j+1,j] = 1
    pcm = _PCM(H, [0.2]*n_sp)
    s = _syndrome_from_faults(H, list(range(n_sp)))
    write_test(f, "s2c_star_ogs1_bps3", pcm, s, 1, 3,
               "star; over_grow_step=1; bits_per_step=3")

    # 25. Rep code exhaustive: all 2^(n-1) syndromes, ogs=1, bps=2
    n = 4
    H = np.zeros((n-1, n), dtype=np.uint8)
    for i in range(n-1):
        H[i,i] = H[i,i+1] = 1
    pcm = _PCM(H, [0.1]*n)
    for bits in range(2**(n-1)):
        s = np.array([(bits >> i) & 1 for i in range(n-1)], dtype=np.uint8)
        write_test(f, f"s2c_rep4_exhaust_bits{bits}_ogs1_bps2", pcm, s, 1, 2,
                   f"rep4 exhaustive: syndrome={list(s)} ogs=1 bps=2")


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else OUTPUT
    with open(out, 'w') as f:
        f.write("# clustering_overgrow_batch_test_cases.txt\n")
        f.write("# Auto-generated by generate_clustering_overgrow_batch_test_cases.py\n")
        f.write("# Replayed by test_clustering_overgrow_batch_vs_python.cpp\n")
        gen_tests(f)
    print(f"Wrote test cases to {out}")
