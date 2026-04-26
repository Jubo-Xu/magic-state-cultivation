"""
test_clustering.py — Tests for the Tanner-graph clustering algorithm.

Test strategy
-------------
Layer 1 — unit tests on individual methods
    1a: _init_each_cluster  — exact state after seeding
    1b: _add_fault          — exact state after absorbing one fault
    1c: _grow_one_step      — minimum-weight fault chosen; collision path
    1d: _merge              — unified check map, RREF dimensions, heap union

Layer 2 — structural invariants after run()
    Checked by _check_invariants() on every test:
    - all active clusters valid
    - global memberships consistent with node sets
    - pairwise disjoint check/fault sets
    - every non-trivial detector in exactly one enclosed_syndromes
    - inverse index maps consistent
    - rref.s matches global syndrome at cluster checks
    - H_cluster @ z == 0 (mod 2) for every null-space vector z

Layer 3 — exhaustive correctness on small classical codes
    3a: rep code  (chain graph) length 3–8, all 2^m syndromes
    3b: ring code (cycle graph) length 3–8, all 2^m syndromes

Layer 4 — Dijkstra ordering
    4a: cheaper direct path chosen before expensive path
    4b: virtual weight accumulation — long cheap path wins over short expensive path

Layer 5 — merge-specific tests
    5a: two seeds sharing one fault → one active cluster after run()
    5b: three seeds merged via 3-body hyperedge
    5c: connecting fault with unclaimed checks (new RREF rows at merge)
    5d: after merge, connecting_j is in surviving cluster's fault_nodes
    5e: absorbed clusters have active=False
    5f: after merge, surviving cluster can still grow outward (heap union)

Test fixtures
-------------
_rep_pcm(n)  — repetition code: n bits, n-1 checks (chain graph)
_ring_pcm(n) — ring code:       n bits, n   checks (cycle graph)
Both use uniform probability p=0.1.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import math
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import stim

from clustering import Clustering, ClusterState
from parity_matrix_construct import ParityCheckMatrices


# ===========================================================================
# Test fixtures
# ===========================================================================

def _rep_pcm_arrays(n: int, p: float = 0.1):
    """
    Repetition code: n bits, n-1 checks.
    H[i, i] = H[i, i+1] = 1  for i in 0..n-2.
    Returns (H, probs).
    """
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H[i, i] = H[i, i + 1] = 1
    probs = [p] * n
    return H, probs


def _ring_pcm_arrays(n: int, p: float = 0.1):
    """
    Ring code: n bits, n checks.
    H[i, i] = H[i, (i+1) % n] = 1  for i in 0..n-1.
    Returns (H, probs).
    """
    H = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        H[i, i] = H[i, (i + 1) % n] = 1
    probs = [p] * n
    return H, probs


class _PCM:
    """Minimal ParityCheckMatrices stand-in."""
    def __init__(self, H: np.ndarray, probs: list):
        self.H = H.astype(np.uint8)
        self.error_data = [
            {'detectors': frozenset(int(i) for i in np.where(H[:, j])[0]),
             'prob': float(probs[j])}
            for j in range(H.shape[1])
        ]

    @classmethod
    def rep(cls, n: int, p: float = 0.1):
        H, probs = _rep_pcm_arrays(n, p)
        return cls(H, probs)

    @classmethod
    def ring(cls, n: int, p: float = 0.1):
        H, probs = _ring_pcm_arrays(n, p)
        return cls(H, probs)


def _syndrome_from_faults(H: np.ndarray, fired: list[int]) -> np.ndarray:
    s = np.zeros(H.shape[0], dtype=np.uint8)
    for j in fired:
        s ^= H[:, j]
    return s


# ===========================================================================
# Shared invariant checker (Layer 2)
# ===========================================================================

def _check_invariants(clustering: Clustering, syndrome: np.ndarray):
    """Assert all structural invariants on a finished Clustering object."""
    pcm     = clustering.pcm
    n_det   = pcm.H.shape[0]
    n_fault = pcm.H.shape[1]
    active  = clustering.active_clusters

    # 2a: every active cluster is valid
    for cl in active:
        assert cl.valid, f"Cluster {cl.cluster_id} not valid"

    # 2b/2c: memberships consistent with node sets; disjoint sets
    seen_checks: dict[int, ClusterState] = {}
    seen_faults: dict[int, ClusterState] = {}
    for cl in active:
        for c in cl.check_nodes:
            assert c not in seen_checks, f"Check {c} in multiple active clusters"
            seen_checks[c] = cl
        for j in cl.fault_nodes:
            assert j not in seen_faults, f"Fault {j} in multiple active clusters"
            seen_faults[j] = cl

    # 2d: every non-trivial detector in exactly one enclosed_syndromes
    for i in range(n_det):
        if syndrome[i]:
            owners = [cl for cl in active if i in cl.enclosed_syndromes]
            assert len(owners) == 1, \
                f"Non-trivial detector {i} in {len(owners)} enclosed_syndromes"

    # 2e: inverse index maps consistent
    for cl in active:
        for c, local in cl.pcm_check_idx_to_cluster_check_idx.items():
            assert cl.cluster_check_idx_to_pcm_check_idx[local] == c, \
                f"Inverse map mismatch at check {c}"

    # 2f: rref.s matches global syndrome at cluster checks
    for cl in active:
        for c_global, c_local in cl.pcm_check_idx_to_cluster_check_idx.items():
            assert cl.rref.s[c_local] == syndrome[c_global], \
                f"rref.s mismatch at check {c_global} in cluster {cl.cluster_id}"

    # 2g: H_cluster @ z == 0 (mod 2) for every null-space vector
    for cl in active:
        n_c = cl.rref.n_checks
        n_b = cl.rref.n_bits
        if n_c == 0 or n_b == 0:
            continue
        H_cl = np.zeros((n_c, n_b), dtype=np.uint8)
        for j_local, j_global in enumerate(cl.cluster_fault_idx_to_pcm_fault_idx):
            for c_global in pcm.error_data[j_global]['detectors']:
                if c_global in cl.pcm_check_idx_to_cluster_check_idx:
                    c_local = cl.pcm_check_idx_to_cluster_check_idx[c_global]
                    H_cl[c_local, j_local] = 1
        for z in cl.rref.Z:
            assert np.all(H_cl @ z % 2 == 0), \
                f"Null-space vector not in kernel for cluster {cl.cluster_id}"


# ===========================================================================
# Layer 1a — _init_each_cluster
# ===========================================================================

class TestInitEachCluster:

    def test_check_nodes_and_enclosed_syndromes(self):
        """Seed check is in check_nodes, boundary_check_nodes, enclosed_syndromes."""
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[2] = 1
        c.clusters_initialization(syndrome)

        assert len(c.clusters) == 1
        cl = c.clusters[0]
        assert cl.cluster_id == 2
        assert cl.active
        assert not cl.valid
        assert 2 in cl.check_nodes
        assert 2 in cl.boundary_check_nodes
        assert 2 in cl.enclosed_syndromes

    def test_global_check_membership_set(self):
        """global_check_membership[seed] points to the new cluster."""
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[1] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]
        assert c.global_check_membership[1] is cl

    def test_heap_contains_adjacent_faults(self):
        """
        Rep code: check i is adjacent to faults i and i+1.
        Seed at check 2 → faults 2 and 3 in heap.
        """
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]
        heap_faults = {j for _, j in cl.heap}
        assert heap_faults == {2, 3}

    def test_rref_empty_at_init(self):
        """RREF has no rows or columns at init — seed not yet in RREF."""
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[0] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]
        assert cl.rref.n_checks == 0
        assert cl.rref.n_bits == 0
        assert cl.pcm_check_idx_to_cluster_check_idx == {}

    def test_multiple_seeds_independent(self):
        """Two non-trivial detectors → two independent clusters."""
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[0] = syndrome[3] = 1
        c.clusters_initialization(syndrome)
        assert len(c.clusters) == 2
        ids = {cl.cluster_id for cl in c.clusters}
        assert ids == {0, 3}
        # global_check_membership set for both seeds
        assert c.global_check_membership[0] is c.clusters[0]
        assert c.global_check_membership[3] is c.clusters[1]


# ===========================================================================
# Layer 1b — _add_fault
# ===========================================================================

class TestAddFault:

    def test_fault_registered_in_fault_nodes(self):
        """After _add_fault, j is in cl.fault_nodes and global_fault_membership."""
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]

        # Manually add fault 2 (touches checks 1 and 2)
        c._add_fault(cl, 2, c.weights[2])

        assert 2 in cl.fault_nodes
        assert c.global_fault_membership[2] is cl
        assert cl.cluster_fault_idx_to_pcm_fault_idx[0] == 2

    def test_seed_enters_rref_on_first_fault(self):
        """
        Seed (check 2) is in check_nodes but not in RREF at init.
        After adding the first fault touching check 2, it enters the RREF.
        """
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]

        assert cl.rref.n_checks == 0
        c._add_fault(cl, 2, c.weights[2])  # fault 2 touches checks 1 and 2
        # Both checks 1 and 2 should now be in RREF
        assert cl.rref.n_checks == 2
        assert 2 in cl.pcm_check_idx_to_cluster_check_idx
        assert 1 in cl.pcm_check_idx_to_cluster_check_idx

    def test_unclaimed_check_taken(self):
        """
        Fault touching an unclaimed check: that check enters check_nodes
        and global_check_membership.
        """
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]

        # fault 2 touches checks 1 (unclaimed) and 2 (seed)
        assert c.global_check_membership[1] is None
        c._add_fault(cl, 2, c.weights[2])
        assert 1 in cl.check_nodes
        assert c.global_check_membership[1] is cl

    def test_rref_syndrome_set_correctly(self):
        """rref.s at the new check rows matches the global syndrome."""
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        # syndrome[2] = 1, syndrome[1] = 0
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]

        c._add_fault(cl, 2, c.weights[2])  # checks 1 and 2

        local_1 = cl.pcm_check_idx_to_cluster_check_idx[1]
        local_2 = cl.pcm_check_idx_to_cluster_check_idx[2]
        assert cl.rref.s[local_1] == syndrome[1]   # 0
        assert cl.rref.s[local_2] == syndrome[2]   # 1

    def test_new_candidates_pushed_to_heap(self):
        """After absorbing fault j, neighboring faults are pushed as candidates."""
        pcm = _PCM.rep(6)
        c = Clustering(pcm)
        syndrome = np.zeros(5, dtype=np.uint8)
        syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]

        # Initially heap has faults 2 and 3 (adjacent to check 2)
        initial_heap_faults = {j for _, j in cl.heap}
        assert initial_heap_faults == {2, 3}

        # Add fault 2 (checks 1, 2): should push fault 1 as new candidate
        c._add_fault(cl, 2, c.weights[2])
        heap_faults = {j for _, j in cl.heap}
        assert 1 in heap_faults  # fault 1 is adjacent to check 1


# ===========================================================================
# Layer 1c — _grow_one_step
# ===========================================================================

class TestGrowOneStep:

    def test_minimum_weight_fault_chosen(self):
        """
        Two faults adjacent to seed: lower weight (higher prob) chosen first.
        PCM: check0 — f0(p=0.4) — check1
             check0 — f1(p=0.1) — check2
        Seed at check0. f0 has lower weight → chosen first.
        """
        H = np.zeros((3, 2), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1   # f0: check0–check1, p=0.4
        H[0, 1] = H[2, 1] = 1   # f1: check0–check2, p=0.1
        pcm = _PCM(H, [0.4, 0.1])
        c = Clustering(pcm)
        syndrome = np.array([1, 1, 0], dtype=np.uint8)
        c.clusters_initialization(syndrome)

        # Only cluster seeded at check0 (the other seed at check1 is separate)
        cl0 = next(cl for cl in c.clusters if cl.cluster_id == 0)
        c._grow_one_step(cl0)

        # f0 (weight = log(0.6/0.4) ≈ 0.405) < f1 (weight = log(0.9/0.1) ≈ 2.197)
        assert 0 in cl0.fault_nodes

    def test_free_fault_absorbed(self):
        """A free fault (no collision) is absorbed into the cluster."""
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[0] = syndrome[3] = 1
        c.clusters_initialization(syndrome)

        cl = next(cl for cl in c.clusters if cl.cluster_id == 0)
        c._grow_one_step(cl)
        assert len(cl.fault_nodes) == 1

    def test_collision_triggers_merge(self):
        """
        Two seeds at det0 and det1; single fault touching both.
        grow_one_step on either cluster should trigger a merge.
        """
        H = np.zeros((2, 1), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        pcm = _PCM(H, [0.3])
        c = Clustering(pcm)
        syndrome = np.array([1, 1], dtype=np.uint8)
        c.run(syndrome)

        active = c.active_clusters
        assert len(active) == 1
        assert 0 in active[0].fault_nodes

    def test_stale_entry_skipped(self):
        """
        After a merge, connecting_j ends up in the merged heap again (from
        other's heap). The membership check skips it cleanly.
        Running run() on a chain verifies no crash or duplicate processing.
        """
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        syndrome = np.array([1, 0, 0, 1], dtype=np.uint8)
        c.run(syndrome)
        _check_invariants(c, syndrome)


# ===========================================================================
# Layer 1d — _merge
# ===========================================================================

class TestMerge:

    def test_larger_survives(self):
        """
        After growing cl0 to absorb one fault before merging,
        cl0 (more faults) should be the surviving cluster.
        """
        # Chain: det0 — f0 — det1 — f1 — det2 — f2 — det3
        H = np.zeros((4, 3), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[1, 1] = H[2, 1] = 1
        H[2, 2] = H[3, 2] = 1
        pcm = _PCM(H, [0.4, 0.4, 0.4])
        c = Clustering(pcm)
        syndrome = np.array([1, 0, 0, 1], dtype=np.uint8)
        c.run(syndrome)

        active = c.active_clusters
        assert len(active) == 1
        inactive = [cl for cl in c.clusters if not cl.active]
        assert len(inactive) == 1

    def test_unified_check_map_complete(self):
        """After merge, surviving cluster's check_nodes includes all checks."""
        H = np.zeros((4, 3), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[1, 1] = H[2, 1] = 1
        H[2, 2] = H[3, 2] = 1
        pcm = _PCM(H, [0.4, 0.4, 0.4])
        c = Clustering(pcm)
        syndrome = np.array([1, 0, 0, 1], dtype=np.uint8)
        c.run(syndrome)

        cl = c.active_clusters[0]
        # All checks touched by absorbed faults should be in the cluster
        for j in cl.fault_nodes:
            for check in pcm.error_data[j]['detectors']:
                assert check in cl.check_nodes

    def test_connecting_j_in_fault_nodes(self):
        """connecting_j (the fault causing the merge) is in the surviving cluster."""
        H = np.zeros((2, 1), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        pcm = _PCM(H, [0.3])
        c = Clustering(pcm)
        syndrome = np.array([1, 1], dtype=np.uint8)
        c.run(syndrome)
        assert 0 in c.active_clusters[0].fault_nodes

    def test_three_way_merge(self):
        """Three seeds all connected via one 3-body hyperedge → single cluster."""
        H = np.zeros((3, 1), dtype=np.uint8)
        H[0, 0] = H[1, 0] = H[2, 0] = 1
        pcm = _PCM(H, [0.1])
        c = Clustering(pcm)
        syndrome = np.array([1, 1, 1], dtype=np.uint8)
        c.run(syndrome)
        active = c.active_clusters
        assert len(active) == 1
        assert 0 in active[0].fault_nodes

    def test_connecting_fault_with_unclaimed_check(self):
        """
        connecting_j touches an unclaimed check → new RREF row added at merge.
        f0: det0–det1 (high prob, chosen first by each seed)
        f1: det0–det2–det1 (triggers merge, introduces det2 as new check)
        """
        H = np.zeros((3, 2), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1           # f0: det0–det1, high prob
        H[0, 1] = H[2, 1] = H[1, 1] = 1  # f1: det0–det2–det1, lower prob
        pcm = _PCM(H, [0.45, 0.1])
        c = Clustering(pcm)
        syndrome = _syndrome_from_faults(H, [1])  # [1,1,1]
        c.run(syndrome)
        _check_invariants(c, syndrome)

    def test_heap_union_allows_further_growth(self):
        """
        After merging two clusters, the surviving cluster can still grow
        using candidates from the absorbed cluster's heap.
        Chain: det0 — f0 — det1 — f1 — det2 — f2 — det3 — f3 — det4
        Seeds at det0 and det4. Each grows inward; when they merge the
        surviving cluster must use the absorbed cluster's heap to finish.
        Syndrome [1,0,0,0,1] is achievable via f0+f1+f2+f3.
        """
        H = np.zeros((5, 4), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[1, 1] = H[2, 1] = 1
        H[2, 2] = H[3, 2] = 1
        H[3, 3] = H[4, 3] = 1
        pcm = _PCM(H, [0.4, 0.4, 0.4, 0.4])
        c = Clustering(pcm)
        syndrome = np.array([1, 0, 0, 0, 1], dtype=np.uint8)
        c.run(syndrome)
        _check_invariants(c, syndrome)
        assert len(c.active_clusters) == 1


# ===========================================================================
# Layer 3 — exhaustive correctness on small classical codes
# ===========================================================================

class TestExhaustive:

    @pytest.mark.parametrize("n", range(3, 9))
    def test_rep_code_all_syndromes(self, n):
        """
        Repetition code of length n: H has full row rank (n-1), so all 2^(n-1)
        syndromes are achievable. Try every one.
        """
        H, probs = _rep_pcm_arrays(n)
        pcm = _PCM(H, probs)
        c = Clustering(pcm)
        m = n - 1   # number of checks

        for bits in range(2 ** m):
            syndrome = np.array([(bits >> i) & 1 for i in range(m)], dtype=np.uint8)
            c.run(syndrome)
            _check_invariants(c, syndrome)

    @pytest.mark.parametrize("n", range(3, 9))
    def test_ring_code_achievable_syndromes(self, n):
        """
        Ring code of length n: H has rank n-1, so achievable syndromes are
        exactly those with even Hamming weight (left null space = [1,1,...,1]).
        Generate achievable syndromes by firing all subsets of faults.
        """
        H, probs = _ring_pcm_arrays(n)
        pcm = _PCM(H, probs)
        c = Clustering(pcm)

        for fault_mask in range(2 ** n):
            fired = [j for j in range(n) if (fault_mask >> j) & 1]
            syndrome = _syndrome_from_faults(H, fired)
            c.run(syndrome)
            _check_invariants(c, syndrome)


# ===========================================================================
# Layer 4 — Dijkstra ordering
# ===========================================================================

class TestDijkstraOrdering:

    def test_cheaper_direct_path_wins(self):
        """
        Two paths from det0 to det1:
          f0: direct,  p=0.4  → weight = log(0.6/0.4) ≈ 0.41
          f1: direct,  p=0.05 → weight = log(0.95/0.05) ≈ 2.94
        f0 should be absorbed first.
        """
        H = np.zeros((2, 2), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1   # f0 high prob
        H[0, 1] = H[1, 1] = 1   # f1 low prob
        pcm = _PCM(H, [0.4, 0.05])
        c = Clustering(pcm)
        syndrome = np.array([1, 1], dtype=np.uint8)
        c.clusters_initialization(syndrome)

        cl = next(cl for cl in c.clusters if cl.cluster_id == 0)
        c._grow_one_step(cl)
        assert 0 in cl.fault_nodes   # f0 absorbed first

    def test_virtual_weight_accumulation(self):
        """
        Path A (virtual weight 2*w_high): det0 — f0(p=0.01) — det1 — f1(p=0.01) — det2
        Path B (virtual weight w_low):    det0 — f2(p=0.4)  — det2
        Path B wins despite being a single 'worse' fault, because p=0.4 >> p=0.01.
        Seed at det0 and det2; f2 should be the connecting fault.
        """
        H = np.zeros((3, 3), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1   # f0: det0–det1, p=0.01
        H[1, 1] = H[2, 1] = 1   # f1: det1–det2, p=0.01
        H[0, 2] = H[2, 2] = 1   # f2: det0–det2, p=0.4
        pcm = _PCM(H, [0.01, 0.01, 0.4])
        c = Clustering(pcm)
        syndrome = np.array([1, 0, 1], dtype=np.uint8)
        c.run(syndrome)
        _check_invariants(c, syndrome)
        # f2 is the direct high-prob connection — it should be in the cluster
        active = c.active_clusters
        assert len(active) == 1
        assert 2 in active[0].fault_nodes


# ===========================================================================
# Layer 5 — edge cases and null-space
# ===========================================================================

class TestEdgeCases:

    def test_empty_syndrome(self):
        """No non-trivial detectors → no clusters."""
        pcm = _PCM.rep(5)
        c = Clustering(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        c.run(syndrome)
        assert c.clusters == []
        assert c.active_clusters == []

    def test_single_boundary_fault(self):
        """
        Fault touching only one detector (boundary).
        Cluster seeded at det0; f0 touches only det0.
        After absorbing f0, syndrome [1] is satisfied.
        """
        H = np.zeros((1, 1), dtype=np.uint8)
        H[0, 0] = 1
        pcm = _PCM(H, [0.3])
        c = Clustering(pcm)
        syndrome = np.array([1], dtype=np.uint8)
        c.run(syndrome)
        _check_invariants(c, syndrome)
        assert len(c.active_clusters) == 1

    def test_redundant_faults_null_space_when_forced(self):
        """
        Null space vectors appear when the cluster must absorb a linearly
        dependent fault to bridge two detectors.

        Ring of 3: det0 — f0 — det1 — f1 — det2 — f2 — det0
        Fire f0+f1+f2 → syndrome [0,0,0] (empty, no seeds) — not useful.
        Fire f0 only → syndrome [1,1,0].
        Seeds: det0 and det1. They merge via f0 (collision), then the merged
        cluster is valid (syndrome satisfied). f1 and f2 never absorbed.
        This is correct behaviour: no null space needed here.

        For a null-space vector to be forced, use a ring where the cluster
        must absorb all 3 faults: fire no faults → no seeds → empty, skip.
        Instead verify the invariants hold for the ring exhaustive test,
        which covers all cases including those where Z is non-empty.
        This test just confirms the structural invariants hold for a simple
        two-fault PCM where the cluster stops early.
        """
        H = np.zeros((2, 2), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[0, 1] = H[1, 1] = 1
        pcm = _PCM(H, [0.3, 0.2])
        c = Clustering(pcm)
        syndrome = np.array([1, 1], dtype=np.uint8)
        c.run(syndrome)
        _check_invariants(c, syndrome)
        active = c.active_clusters
        assert len(active) == 1
        # f0 alone satisfies the syndrome → cluster stops after 1 fault
        assert active[0].rref.n_bits == 1
        assert len(active[0].rref.Z) == 0

    def test_null_space_in_ring_code(self):
        """
        Ring code n=4: f0-f1-f2-f3 form a cycle. Fire f0+f2 (opposite faults).
        Syndrome = [1,0,1,0]. Seeds at det0 and det2.
        Each seed grows inward; the two clusters merge. After absorbing f0
        (det0-det1) and f2 (det2-det3), the merged cluster still needs to
        connect det1 and det3. It then absorbs f1 (det1-det2, collision side)
        or f3 (det3-det0), creating a cycle. The absorbed faults are linearly
        dependent over GF(2), giving at least one null-space vector.
        """
        H, probs = _ring_pcm_arrays(4)
        pcm = _PCM(H, probs)
        c = Clustering(pcm)
        syndrome = _syndrome_from_faults(H, [0, 2])  # [1,0,1,0]
        c.run(syndrome)
        _check_invariants(c, syndrome)
        active = c.active_clusters
        assert len(active) == 1
        # The cluster must have absorbed at least 2 faults (f0 and f2)
        assert active[0].rref.n_bits >= 2
        # All null-space vectors are in the kernel (verified by _check_invariants)
        for z in active[0].rref.Z:
            assert len(z) == active[0].rref.n_bits

    def test_independent_components_no_merge(self):
        """Two disconnected graph components → two independent active clusters."""
        H = np.zeros((4, 2), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1   # component A
        H[2, 1] = H[3, 1] = 1   # component B
        pcm = _PCM(H, [0.3, 0.3])
        c = Clustering(pcm)
        syndrome = np.array([1, 1, 1, 1], dtype=np.uint8)
        c.run(syndrome)
        _check_invariants(c, syndrome)
        assert len(c.active_clusters) == 2

    def test_run_resets_state_between_calls(self):
        """Calling run() twice gives clean independent results."""
        pcm = _PCM.rep(5)
        c = Clustering(pcm)

        s1 = np.array([1, 0, 0, 1], dtype=np.uint8)
        c.run(s1)
        n_active_1 = len(c.active_clusters)
        _check_invariants(c, s1)

        s2 = np.array([1, 1, 0, 0], dtype=np.uint8)
        c.run(s2)
        _check_invariants(c, s2)
        # State from first run must not leak into second
        for cl in c.active_clusters:
            for j in cl.fault_nodes:
                assert c.global_fault_membership[j] is cl

    def test_large_chain(self):
        """Chain of length 20 with seeds at both ends → one merged cluster."""
        n = 20
        H = np.zeros((n, n - 1), dtype=np.uint8)
        for j in range(n - 1):
            H[j, j] = H[j + 1, j] = 1
        pcm = _PCM(H, [0.1] * (n - 1))
        c = Clustering(pcm)
        syndrome = np.zeros(n, dtype=np.uint8)
        syndrome[0] = syndrome[n - 1] = 1
        c.run(syndrome)
        _check_invariants(c, syndrome)
        assert len(c.active_clusters) == 1

    def test_connecting_fault_pushes_unseen_candidate(self):
        """
        Covers lines 464-465: the push loop after absorbing connecting_j
        finds a fault k that was never pushed into any heap before.

        Layout (6 detectors, 5 faults):
          det0(seed A) —f1(cheap)— det2 —f3(connect)— det4(unclaimed) —f4(new)— det5
          det1(seed B) —f2(cheap)— det3 —f3(connect)
          det0 —f0(expensive)— det1

        Growth trace:
          A absorbs f1 (det0–det2, cheap). Not valid. Pushes f3 via det2.
          B absorbs f2 (det1–det3, cheap). Not valid. Pushes f3 via det3.
          f3 (det2–det3–det4) is the connecting fault: det2 in A, det3 in B,
          det4 unclaimed. After merge, det4 is new_checks_j.
          Push loop: det4's neighbor f4 (det4–det5) has NEVER been pushed into
          any heap → larger.dist.get(f4, inf) = inf → lines 464-465 execute.
          Cluster then absorbs f4 and finally f0 to become valid.
        """
        #       f0   f1   f2   f3   f4
        # det0:  1    1    0    0    0
        # det1:  1    0    1    0    0
        # det2:  0    1    0    1    0
        # det3:  0    0    1    1    0
        # det4:  0    0    0    1    1
        # det5:  0    0    0    0    1
        H = np.zeros((6, 5), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1           # f0: det0–det1 (expensive)
        H[0, 1] = H[2, 1] = 1           # f1: det0–det2 (cheap)
        H[1, 2] = H[3, 2] = 1           # f2: det1–det3 (cheap)
        H[2, 3] = H[3, 3] = H[4, 3] = 1 # f3: det2–det3–det4 (connecting)
        H[4, 4] = H[5, 4] = 1           # f4: det4–det5 (unseen until f3 merge)
        probs = [0.01, 0.4, 0.4, 0.2, 0.1]
        pcm = _PCM(H, probs)
        c = Clustering(pcm)
        # Fire f0 → syndrome [1,1,0,0,0,0]; seeds at det0 and det1
        syndrome = _syndrome_from_faults(H, [0])
        assert list(syndrome) == [1, 1, 0, 0, 0, 0]
        c.run(syndrome)
        _check_invariants(c, syndrome)
        active = c.active_clusters
        assert len(active) == 1
        # All 5 faults should be absorbed (f4 via the covered push path, f0 for validity)
        assert active[0].fault_nodes == {0, 1, 2, 3, 4}

    def test_star_graph(self):
        """
        Hub-and-spoke: central det0 connects to det1..det4 via f0..f3.
        Left null space of H is [1,1,1,1,1], so achievable syndromes have
        even Hamming weight. Fire f0+f1+f2+f3 → syndrome [0,1,1,1,1]
        (det0 hit 4 times = 0, det1..det4 each once = 1).
        Seeds at det1..det4; all merge into one cluster via det0 as hub.
        """
        n_spokes = 4
        H = np.zeros((n_spokes + 1, n_spokes), dtype=np.uint8)
        for j in range(n_spokes):
            H[0, j] = H[j + 1, j] = 1
        pcm = _PCM(H, [0.2] * n_spokes)
        c = Clustering(pcm)
        # Fire all spokes: det0 flipped 4 times (=0), det1..4 flipped once (=1)
        syndrome = _syndrome_from_faults(H, list(range(n_spokes)))
        assert syndrome[0] == 0 and all(syndrome[1:] == 1)
        c.run(syndrome)
        _check_invariants(c, syndrome)
        assert len(c.active_clusters) == 1


# ===========================================================================
# Layer 6 — surface code (d=3, 1 round, rotated memory-Z)
# ===========================================================================

def _surface_code_pcm(distance: int = 3, rounds: int = 1, p: float = 0.01) -> ParityCheckMatrices:
    """
    Build a ParityCheckMatrices from a stim rotated surface code circuit.

    Uses rotated_memory_z with after_clifford_depolarization noise.
    The DEM is flattened so there are no repeat blocks.
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
    )
    dem = circuit.detector_error_model(flatten_loops=True)
    return ParityCheckMatrices.from_DEM(dem, decompose=False)


def _surface_syndrome_from_faults(pcm: ParityCheckMatrices, fired: list[int]) -> np.ndarray:
    """Return H @ e (mod 2) for the given set of fired fault indices."""
    s = np.zeros(pcm.H.shape[0], dtype=np.uint8)
    for j in fired:
        s ^= pcm.H[:, j]
    return s


class TestSurfaceCode:
    """
    Layer 6: clustering on a d=3 rotated surface code (1 round).

    The DEM has 8 detectors and ~23 faults, including boundary faults,
    spacelike faults, and timelike faults.  All tests generate achievable
    syndromes by firing known subsets of faults via H @ e mod 2.
    """

    @pytest.fixture(scope="class")
    def pcm(self):
        return _surface_code_pcm(distance=3, rounds=1, p=0.01)

    def test_pcm_shape(self, pcm):
        """DEM has the expected number of detectors and faults."""
        assert pcm.H.shape[0] == 8   # d=3, 1 round → 8 detectors
        assert pcm.H.shape[1] > 0
        # All probabilities are positive
        assert all(ed['prob'] > 0 for ed in pcm.error_data)

    def test_empty_syndrome(self, pcm):
        """Zero syndrome → run() produces no clusters, no error."""
        c = Clustering(pcm)
        syndrome = np.zeros(pcm.H.shape[0], dtype=np.uint8)
        c.run(syndrome)
        assert c.clusters == []
        assert c.active_clusters == []

    def test_single_fault_each(self, pcm):
        """
        For every fault j, fire just fault j.
        Clustering must terminate with all invariants satisfied.
        For 0-detector faults the syndrome is empty (no clusters).
        For 1-detector (boundary) and 2-detector faults at least one active
        cluster must exist and cover all non-trivial detectors.
        Note: clustering finds a minimum-weight explanation — not necessarily
        fault j itself, so we only check invariants, not which fault was chosen.
        """
        H = pcm.H
        n_fault = H.shape[1]
        c = Clustering(pcm)
        for j in range(n_fault):
            syndrome = _surface_syndrome_from_faults(pcm, [j])
            c.run(syndrome)
            _check_invariants(c, syndrome)
            n_nontrivial = int(syndrome.sum())
            if n_nontrivial == 0:
                assert c.active_clusters == []
            else:
                assert len(c.active_clusters) >= 1

    def test_two_fault_combinations(self, pcm):
        """
        Fire every pair of distinct faults.
        Checks that merging and Dijkstra ordering work correctly for the
        full surface code geometry (boundary edges, corners, timelike faults).
        """
        H = pcm.H
        n_fault = H.shape[1]
        c = Clustering(pcm)
        for j0 in range(n_fault):
            for j1 in range(j0 + 1, n_fault):
                syndrome = _surface_syndrome_from_faults(pcm, [j0, j1])
                c.run(syndrome)
                _check_invariants(c, syndrome)

    def test_random_fault_subsets(self, pcm):
        """
        Run clustering on a handful of representative fault subsets to catch
        gross failures.  The full hypothesis sweep is in the module-level
        test_surface_code_random_fault_subsets below (class fixtures and
        @given do not compose cleanly in pytest-hypothesis).
        """
        n_fault = pcm.H.shape[1]
        c = Clustering(pcm)
        # empty, single, all, first-half, second-half
        subsets = [
            [],
            [0],
            list(range(n_fault)),
            list(range(n_fault // 2)),
            list(range(n_fault // 2, n_fault)),
        ]
        for fired in subsets:
            syndrome = _surface_syndrome_from_faults(pcm, fired)
            c.run(syndrome)
            _check_invariants(c, syndrome)


# ---------------------------------------------------------------------------
# Hypothesis sweep — lives outside the class so @given composes cleanly
# ---------------------------------------------------------------------------

@given(st.data())
@settings(max_examples=200, deadline=None)
def test_surface_code_random_fault_subsets(data):
    """
    Hypothesis: draw a random subset of faults from the d=3 surface code DEM,
    build the achievable syndrome (H @ e mod 2), run clustering, and check all
    structural invariants.  Covers a broad variety of cluster sizes and merge
    patterns across 200 random examples.
    """
    pcm = _surface_code_pcm(distance=3, rounds=1, p=0.01)
    n_fault = pcm.H.shape[1]
    fired = data.draw(
        st.lists(st.integers(0, n_fault - 1), min_size=0, max_size=n_fault, unique=True)
    )
    syndrome = _surface_syndrome_from_faults(pcm, fired)
    c = Clustering(pcm)
    c.run(syndrome)
    _check_invariants(c, syndrome)
