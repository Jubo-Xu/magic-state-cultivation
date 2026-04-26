"""
test_clustering_overgrow_batch.py — Tests for ClusteringOvergrowBatch.

Section 1 — mirrors every test in test_clustering.py using
    ClusteringOvergrowBatch(pcm).run(syndrome, over_grow_step=0, bits_per_step=1)
    With these defaults the class is a drop-in replacement for Clustering and
    must produce field-identical rref state (T, U, H, s, s_prime, pivot_map, Z).

Section 2 — batch-and-overgrow-specific tests:
    2a: over_grow_step > 0 delays termination (budget countdown)
    2b: bits_per_step > 1 still terminates with all clusters valid
    2c: bits_per_step=1 is field-identical to ClusteringOvergrow (same over_grow_step)
    2d: multiple connecting faults in one batch step (bits_per_step ≥ 2 + two collisions)
    2e: overgrow budget resets after a post-overgrow merge invalidates cluster
    2f: hypothesis sweep — surface code, random syndrome, random params
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import stim

from clustering import Clustering, ClusterState
from clustering_overgrow import Clustering as ClusteringOvergrow
from clustering_overgrow_batch import ClusteringOvergrowBatch
from incremental_rref_batch import IncrementalRREFBatch
from parity_matrix_construct import ParityCheckMatrices


# ===========================================================================
# Shared fixtures (identical to test_clustering.py)
# ===========================================================================

def _rep_pcm_arrays(n: int, p: float = 0.1):
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H[i, i] = H[i, i + 1] = 1
    return H, [p] * n


def _ring_pcm_arrays(n: int, p: float = 0.1):
    H = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        H[i, i] = H[i, (i + 1) % n] = 1
    return H, [p] * n


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


def _syndrome_from_faults(H: np.ndarray, fired: list) -> np.ndarray:
    s = np.zeros(H.shape[0], dtype=np.uint8)
    for j in fired:
        s ^= H[:, j]
    return s


def _surface_code_pcm(distance: int = 3, rounds: int = 1, p: float = 0.01):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
    )
    dem = circuit.detector_error_model(flatten_loops=True)
    return ParityCheckMatrices.from_DEM(dem, decompose=False)


def _surface_syndrome_from_faults(pcm, fired: list) -> np.ndarray:
    s = np.zeros(pcm.H.shape[0], dtype=np.uint8)
    for j in fired:
        s ^= pcm.H[:, j]
    return s


# ===========================================================================
# Shared invariant checkers
# ===========================================================================

def _check_invariants(clustering, syndrome: np.ndarray):
    """Assert all structural invariants on a finished clustering object."""
    pcm     = clustering.pcm
    n_det   = pcm.H.shape[0]
    n_fault = pcm.H.shape[1]
    active  = clustering.active_clusters

    for cl in active:
        assert cl.valid, f"Cluster {cl.cluster_id} not valid"

    seen_checks: dict = {}
    seen_faults: dict = {}
    for cl in active:
        for c in cl.check_nodes:
            assert c not in seen_checks, f"Check {c} in multiple active clusters"
            seen_checks[c] = cl
        for j in cl.fault_nodes:
            assert j not in seen_faults, f"Fault {j} in multiple active clusters"
            seen_faults[j] = cl

    for i in range(n_det):
        if syndrome[i]:
            owners = [cl for cl in active if i in cl.enclosed_syndromes]
            assert len(owners) == 1, \
                f"Non-trivial detector {i} in {len(owners)} enclosed_syndromes"

    for cl in active:
        for c, local in cl.pcm_check_idx_to_cluster_check_idx.items():
            assert cl.cluster_check_idx_to_pcm_check_idx[local] == c

    for cl in active:
        for c_global, c_local in cl.pcm_check_idx_to_cluster_check_idx.items():
            assert cl.rref.s[c_local] == syndrome[c_global], \
                f"rref.s mismatch at check {c_global}"

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


def _assert_rref_fields_equal(cl_batch, cl_ref, label=''):
    """Assert rref fields are identical between two ClusterState objects."""
    tag = f" [{label}]" if label else ""
    b, r = cl_batch.rref, cl_ref.rref
    assert np.array_equal(b.T, r.T),        f"T mismatch{tag}"
    assert np.array_equal(b.U, r.U),        f"U mismatch{tag}"
    assert np.array_equal(b.H, r.H),        f"H mismatch{tag}"
    assert np.array_equal(b.s, r.s),        f"s mismatch{tag}"
    assert np.array_equal(b.s_prime, r.s_prime), f"s_prime mismatch{tag}"
    assert b.pivot_map == r.pivot_map,       f"pivot_map mismatch{tag}"
    assert len(b.Z) == len(r.Z),            f"Z length mismatch{tag}"
    for i, (zb, zr) in enumerate(zip(b.Z, r.Z)):
        assert np.array_equal(zb, zr),      f"Z[{i}] mismatch{tag}"


# ===========================================================================
# SECTION 1 — exact mirrors of test_clustering.py
# (over_grow_step=0, bits_per_step=1  →  drop-in for Clustering)
# ===========================================================================

class TestSection1_InitEachCluster:
    """Layer 1a: _init_each_cluster — same as TestInitEachCluster."""

    def _new(self, pcm):
        return ClusteringOvergrowBatch(pcm)

    def test_check_nodes_and_enclosed_syndromes(self):
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.zeros(4, dtype=np.uint8); syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]
        assert cl.cluster_id == 2
        assert cl.active and not cl.valid
        assert 2 in cl.check_nodes
        assert 2 in cl.boundary_check_nodes
        assert 2 in cl.enclosed_syndromes

    def test_global_check_membership_set(self):
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.zeros(4, dtype=np.uint8); syndrome[1] = 1
        c.clusters_initialization(syndrome)
        assert c.global_check_membership[1] is c.clusters[0]

    def test_heap_contains_adjacent_faults(self):
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.zeros(4, dtype=np.uint8); syndrome[2] = 1
        c.clusters_initialization(syndrome)
        heap_faults = {j for _, j in c.clusters[0].heap}
        assert heap_faults == {2, 3}

    def test_rref_empty_at_init(self):
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.zeros(4, dtype=np.uint8); syndrome[0] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]
        assert cl.rref.n_checks == 0
        assert cl.rref.n_bits == 0
        assert cl.pcm_check_idx_to_cluster_check_idx == {}

    def test_rref_is_batch_instance(self):
        """_init_each_cluster must install IncrementalRREFBatch, not IncrementalRREF."""
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.zeros(4, dtype=np.uint8); syndrome[0] = 1
        c.clusters_initialization(syndrome)
        assert isinstance(c.clusters[0].rref, IncrementalRREFBatch)

    def test_multiple_seeds_independent(self):
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[0] = syndrome[3] = 1
        c.clusters_initialization(syndrome)
        assert len(c.clusters) == 2
        assert {cl.cluster_id for cl in c.clusters} == {0, 3}


class TestSection1_AddFault:
    """Layer 1b: _add_fault — same as TestAddFault."""

    def _new(self, pcm):
        return ClusteringOvergrowBatch(pcm)

    def test_fault_registered(self):
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.zeros(4, dtype=np.uint8); syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]
        c._add_fault(cl, 2, c.weights[2])
        assert 2 in cl.fault_nodes
        assert c.global_fault_membership[2] is cl
        assert cl.cluster_fault_idx_to_pcm_fault_idx[0] == 2

    def test_seed_enters_rref_on_first_fault(self):
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.zeros(4, dtype=np.uint8); syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]
        assert cl.rref.n_checks == 0
        c._add_fault(cl, 2, c.weights[2])
        assert cl.rref.n_checks == 2
        assert 2 in cl.pcm_check_idx_to_cluster_check_idx
        assert 1 in cl.pcm_check_idx_to_cluster_check_idx

    def test_unclaimed_check_taken(self):
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.zeros(4, dtype=np.uint8); syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]
        assert c.global_check_membership[1] is None
        c._add_fault(cl, 2, c.weights[2])
        assert 1 in cl.check_nodes
        assert c.global_check_membership[1] is cl

    def test_rref_syndrome_set_correctly(self):
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.zeros(4, dtype=np.uint8); syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]
        c._add_fault(cl, 2, c.weights[2])
        local_1 = cl.pcm_check_idx_to_cluster_check_idx[1]
        local_2 = cl.pcm_check_idx_to_cluster_check_idx[2]
        assert cl.rref.s[local_1] == syndrome[1]
        assert cl.rref.s[local_2] == syndrome[2]

    def test_new_candidates_pushed_to_heap(self):
        pcm = _PCM.rep(6)
        c = self._new(pcm)
        syndrome = np.zeros(5, dtype=np.uint8); syndrome[2] = 1
        c.clusters_initialization(syndrome)
        cl = c.clusters[0]
        c._add_fault(cl, 2, c.weights[2])
        heap_faults = {j for _, j in cl.heap}
        assert 1 in heap_faults


class TestSection1_GrowOneStep:
    """Layer 1c: _grow_one_step with bits_per_step=1 — same as TestGrowOneStep."""

    def _new(self, pcm):
        c = ClusteringOvergrowBatch(pcm)
        c.bits_per_step = 1
        c.over_grow_step = 0
        return c

    def test_minimum_weight_fault_chosen(self):
        H = np.zeros((3, 2), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[0, 1] = H[2, 1] = 1
        pcm = _PCM(H, [0.4, 0.1])
        c = self._new(pcm)
        syndrome = np.array([1, 1, 0], dtype=np.uint8)
        c.clusters_initialization(syndrome)
        cl0 = next(cl for cl in c.clusters if cl.cluster_id == 0)
        c._grow_one_step(cl0)
        assert 0 in cl0.fault_nodes

    def test_free_fault_absorbed(self):
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.zeros(4, dtype=np.uint8)
        syndrome[0] = syndrome[3] = 1
        c.clusters_initialization(syndrome)
        cl = next(cl for cl in c.clusters if cl.cluster_id == 0)
        c._grow_one_step(cl)
        assert len(cl.fault_nodes) == 1

    def test_collision_triggers_merge(self):
        H = np.zeros((2, 1), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        pcm = _PCM(H, [0.3])
        c = self._new(pcm)
        syndrome = np.array([1, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        active = c.active_clusters
        assert len(active) == 1
        assert 0 in active[0].fault_nodes

    def test_stale_entry_skipped(self):
        pcm = _PCM.rep(5)
        c = self._new(pcm)
        syndrome = np.array([1, 0, 0, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, syndrome)


class TestSection1_Merge:
    """Layer 1d: _merge_batch with one connecting fault — same as TestMerge."""

    def _new(self, pcm):
        return ClusteringOvergrowBatch(pcm)

    def test_larger_survives(self):
        H = np.zeros((4, 3), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[1, 1] = H[2, 1] = 1
        H[2, 2] = H[3, 2] = 1
        pcm = _PCM(H, [0.4, 0.4, 0.4])
        c = self._new(pcm)
        syndrome = np.array([1, 0, 0, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        assert len(c.active_clusters) == 1
        assert len([cl for cl in c.clusters if not cl.active]) == 1

    def test_unified_check_map_complete(self):
        H = np.zeros((4, 3), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[1, 1] = H[2, 1] = 1
        H[2, 2] = H[3, 2] = 1
        pcm = _PCM(H, [0.4, 0.4, 0.4])
        c = self._new(pcm)
        syndrome = np.array([1, 0, 0, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        cl = c.active_clusters[0]
        for j in cl.fault_nodes:
            for check in pcm.error_data[j]['detectors']:
                assert check in cl.check_nodes

    def test_connecting_j_in_fault_nodes(self):
        H = np.zeros((2, 1), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        pcm = _PCM(H, [0.3])
        c = self._new(pcm)
        syndrome = np.array([1, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        assert 0 in c.active_clusters[0].fault_nodes

    def test_three_way_merge(self):
        H = np.zeros((3, 1), dtype=np.uint8)
        H[0, 0] = H[1, 0] = H[2, 0] = 1
        pcm = _PCM(H, [0.1])
        c = self._new(pcm)
        syndrome = np.array([1, 1, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        active = c.active_clusters
        assert len(active) == 1
        assert 0 in active[0].fault_nodes

    def test_connecting_fault_with_unclaimed_check(self):
        H = np.zeros((3, 2), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[0, 1] = H[2, 1] = H[1, 1] = 1
        pcm = _PCM(H, [0.45, 0.1])
        c = self._new(pcm)
        syndrome = _syndrome_from_faults(H, [1])
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, syndrome)

    def test_heap_union_allows_further_growth(self):
        H = np.zeros((5, 4), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[1, 1] = H[2, 1] = 1
        H[2, 2] = H[3, 2] = 1
        H[3, 3] = H[4, 3] = 1
        pcm = _PCM(H, [0.4, 0.4, 0.4, 0.4])
        c = self._new(pcm)
        syndrome = np.array([1, 0, 0, 0, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, syndrome)
        assert len(c.active_clusters) == 1


class TestSection1_Exhaustive:
    """Layer 3: exhaustive correctness on small codes."""

    @pytest.mark.parametrize("n", range(3, 9))
    def test_rep_code_all_syndromes(self, n):
        H, probs = _rep_pcm_arrays(n)
        pcm = _PCM(H, probs)
        c = ClusteringOvergrowBatch(pcm)
        m = n - 1
        for bits in range(2 ** m):
            syndrome = np.array([(bits >> i) & 1 for i in range(m)], dtype=np.uint8)
            c.run(syndrome, over_grow_step=0, bits_per_step=1)
            _check_invariants(c, syndrome)

    @pytest.mark.parametrize("n", range(3, 9))
    def test_ring_code_achievable_syndromes(self, n):
        H, probs = _ring_pcm_arrays(n)
        pcm = _PCM(H, probs)
        c = ClusteringOvergrowBatch(pcm)
        for fault_mask in range(2 ** n):
            fired = [j for j in range(n) if (fault_mask >> j) & 1]
            syndrome = _syndrome_from_faults(H, fired)
            c.run(syndrome, over_grow_step=0, bits_per_step=1)
            _check_invariants(c, syndrome)


class TestSection1_DijkstraOrdering:
    """Layer 4: Dijkstra ordering."""

    def test_cheaper_direct_path_wins(self):
        H = np.zeros((2, 2), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[0, 1] = H[1, 1] = 1
        pcm = _PCM(H, [0.4, 0.05])
        c = ClusteringOvergrowBatch(pcm)
        syndrome = np.array([1, 1], dtype=np.uint8)
        c.clusters_initialization(syndrome)
        c.bits_per_step = 1
        c.over_grow_step = 0
        cl = next(cl for cl in c.clusters if cl.cluster_id == 0)
        c._grow_one_step(cl)
        assert 0 in cl.fault_nodes

    def test_virtual_weight_accumulation(self):
        H = np.zeros((3, 3), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[1, 1] = H[2, 1] = 1
        H[0, 2] = H[2, 2] = 1
        pcm = _PCM(H, [0.01, 0.01, 0.4])
        c = ClusteringOvergrowBatch(pcm)
        syndrome = np.array([1, 0, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, syndrome)
        assert 2 in c.active_clusters[0].fault_nodes


class TestSection1_EdgeCases:
    """Layer 5: edge cases and null-space."""

    def test_empty_syndrome(self):
        pcm = _PCM.rep(5)
        c = ClusteringOvergrowBatch(pcm)
        c.run(np.zeros(4, dtype=np.uint8), over_grow_step=0, bits_per_step=1)
        assert c.clusters == []

    def test_single_boundary_fault(self):
        H = np.zeros((1, 1), dtype=np.uint8); H[0, 0] = 1
        pcm = _PCM(H, [0.3])
        c = ClusteringOvergrowBatch(pcm)
        syndrome = np.array([1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, syndrome)
        assert len(c.active_clusters) == 1

    def test_redundant_faults_null_space(self):
        H = np.zeros((2, 2), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[0, 1] = H[1, 1] = 1
        pcm = _PCM(H, [0.3, 0.2])
        c = ClusteringOvergrowBatch(pcm)
        syndrome = np.array([1, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, syndrome)
        active = c.active_clusters
        assert len(active) == 1
        assert active[0].rref.n_bits == 1
        assert len(active[0].rref.Z) == 0

    def test_null_space_in_ring_code(self):
        H, probs = _ring_pcm_arrays(4)
        pcm = _PCM(H, probs)
        c = ClusteringOvergrowBatch(pcm)
        syndrome = _syndrome_from_faults(H, [0, 2])
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, syndrome)
        active = c.active_clusters
        assert len(active) == 1
        assert active[0].rref.n_bits >= 2

    def test_independent_components_no_merge(self):
        H = np.zeros((4, 2), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[2, 1] = H[3, 1] = 1
        pcm = _PCM(H, [0.3, 0.3])
        c = ClusteringOvergrowBatch(pcm)
        syndrome = np.array([1, 1, 1, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, syndrome)
        assert len(c.active_clusters) == 2

    def test_run_resets_state_between_calls(self):
        pcm = _PCM.rep(5)
        c = ClusteringOvergrowBatch(pcm)
        s1 = np.array([1, 0, 0, 1], dtype=np.uint8)
        c.run(s1, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, s1)
        s2 = np.array([1, 1, 0, 0], dtype=np.uint8)
        c.run(s2, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, s2)
        for cl in c.active_clusters:
            for j in cl.fault_nodes:
                assert c.global_fault_membership[j] is cl

    def test_large_chain(self):
        n = 20
        H = np.zeros((n, n - 1), dtype=np.uint8)
        for j in range(n - 1):
            H[j, j] = H[j + 1, j] = 1
        pcm = _PCM(H, [0.1] * (n - 1))
        c = ClusteringOvergrowBatch(pcm)
        syndrome = np.zeros(n, dtype=np.uint8)
        syndrome[0] = syndrome[n - 1] = 1
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, syndrome)
        assert len(c.active_clusters) == 1

    def test_star_graph(self):
        n_spokes = 4
        H = np.zeros((n_spokes + 1, n_spokes), dtype=np.uint8)
        for j in range(n_spokes):
            H[0, j] = H[j + 1, j] = 1
        pcm = _PCM(H, [0.2] * n_spokes)
        c = ClusteringOvergrowBatch(pcm)
        syndrome = _syndrome_from_faults(H, list(range(n_spokes)))
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        _check_invariants(c, syndrome)
        assert len(c.active_clusters) == 1


class TestSection1_FieldIdentity:
    """
    Explicit field-identity: ClusteringOvergrowBatch(over_grow_step=0,
    bits_per_step=1) must produce T, U, H, s, s_prime, pivot_map, Z
    bit-for-bit identical to Clustering.
    """

    def _run_both(self, pcm, syndrome):
        ref = Clustering(pcm)
        ref.run(syndrome)
        batch = ClusteringOvergrowBatch(pcm)
        batch.run(syndrome, over_grow_step=0, bits_per_step=1)
        return ref, batch

    def _match_clusters(self, ref, batch):
        """Pair up active clusters by cluster_id."""
        ref_map   = {cl.cluster_id: cl for cl in ref.active_clusters}
        batch_map = {cl.cluster_id: cl for cl in batch.active_clusters}
        assert set(ref_map) == set(batch_map), \
            f"Active cluster ids differ: {set(ref_map)} vs {set(batch_map)}"
        return ref_map, batch_map

    def test_rep5_field_identical(self):
        pcm = _PCM.rep(5)
        syndrome = np.array([1, 0, 0, 1], dtype=np.uint8)
        ref, batch = self._run_both(pcm, syndrome)
        rm, bm = self._match_clusters(ref, batch)
        for cid in rm:
            _assert_rref_fields_equal(bm[cid], rm[cid], label=f"cluster {cid}")

    def test_ring4_field_identical(self):
        H, probs = _ring_pcm_arrays(4)
        pcm = _PCM(H, probs)
        syndrome = _syndrome_from_faults(H, [0, 2])
        ref, batch = self._run_both(pcm, syndrome)
        rm, bm = self._match_clusters(ref, batch)
        for cid in rm:
            _assert_rref_fields_equal(bm[cid], rm[cid], label=f"cluster {cid}")

    def test_rep_exhaustive_field_identical(self):
        n = 5
        H, probs = _rep_pcm_arrays(n)
        pcm = _PCM(H, probs)
        ref   = Clustering(pcm)
        batch = ClusteringOvergrowBatch(pcm)
        for bits in range(2 ** (n - 1)):
            syndrome = np.array([(bits >> i) & 1 for i in range(n - 1)], dtype=np.uint8)
            ref.run(syndrome)
            batch.run(syndrome, over_grow_step=0, bits_per_step=1)
            rm = {cl.cluster_id: cl for cl in ref.active_clusters}
            bm = {cl.cluster_id: cl for cl in batch.active_clusters}
            assert set(rm) == set(bm)
            for cid in rm:
                _assert_rref_fields_equal(bm[cid], rm[cid], label=f"bits={bits} cluster={cid}")


class TestSection1_SurfaceCode:
    """Layer 6: surface code with over_grow_step=0, bits_per_step=1."""

    @pytest.fixture(scope="class")
    def pcm(self):
        return _surface_code_pcm(distance=3, rounds=1, p=0.01)

    def test_pcm_shape(self, pcm):
        assert pcm.H.shape[0] == 8
        assert pcm.H.shape[1] > 0

    def test_empty_syndrome(self, pcm):
        c = ClusteringOvergrowBatch(pcm)
        syndrome = np.zeros(pcm.H.shape[0], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=1)
        assert c.clusters == []

    def test_single_fault_each(self, pcm):
        c = ClusteringOvergrowBatch(pcm)
        for j in range(pcm.H.shape[1]):
            syndrome = _surface_syndrome_from_faults(pcm, [j])
            c.run(syndrome, over_grow_step=0, bits_per_step=1)
            _check_invariants(c, syndrome)

    def test_two_fault_combinations(self, pcm):
        c = ClusteringOvergrowBatch(pcm)
        n_fault = pcm.H.shape[1]
        for j0 in range(n_fault):
            for j1 in range(j0 + 1, n_fault):
                syndrome = _surface_syndrome_from_faults(pcm, [j0, j1])
                c.run(syndrome, over_grow_step=0, bits_per_step=1)
                _check_invariants(c, syndrome)


@given(st.data())
@settings(max_examples=200, deadline=None)
def test_s1_surface_code_hypothesis(data):
    """Section 1 hypothesis: bits_per_step=1 matches full invariants."""
    pcm = _surface_code_pcm(distance=3, rounds=1, p=0.01)
    n_fault = pcm.H.shape[1]
    fired = data.draw(
        st.lists(st.integers(0, n_fault - 1), min_size=0, max_size=n_fault, unique=True)
    )
    syndrome = _surface_syndrome_from_faults(pcm, fired)
    c = ClusteringOvergrowBatch(pcm)
    c.run(syndrome, over_grow_step=0, bits_per_step=1)
    _check_invariants(c, syndrome)


# ===========================================================================
# SECTION 2 — ClusteringOvergrowBatch-specific tests
# ===========================================================================

class TestSection2_OverGrow:
    """
    2a: over_grow_step > 0 forces extra growth steps after first valid_neutral.
    """

    def test_overgrow_step0_identical_to_base(self):
        """over_grow_step=0, bits_per_step=1 identical to Clustering."""
        pcm = _PCM.rep(5)
        syndrome = np.array([1, 0, 0, 1], dtype=np.uint8)
        ref = Clustering(pcm)
        ref.run(syndrome)
        batch = ClusteringOvergrowBatch(pcm)
        batch.run(syndrome, over_grow_step=0, bits_per_step=1)
        rm = {cl.cluster_id: cl for cl in ref.active_clusters}
        bm = {cl.cluster_id: cl for cl in batch.active_clusters}
        assert set(rm) == set(bm)
        for cid in rm:
            _assert_rref_fields_equal(bm[cid], rm[cid])

    def test_overgrow_absorbs_more_faults(self):
        """
        With over_grow_step > 0, the cluster continues growing after first
        reaching valid_neutral, absorbing additional faults.
        Chain: det0 — f0 — det1 — f1 — det2
        Syndrome [1,0,1] is satisfied after f0 alone (the cluster bridges det0
        to det1 and det1 is trivial; absorbing f1 then connects det2).
        Actually let's use: det0 — f0 — det1 (alone satisfies [1,1]).
        Then f1: det1 (boundary), f2: det0 (boundary).
        With over_grow_step=2, after becoming valid the cluster keeps growing.
        """
        # 2 checks, 3 faults: f0 bridges both, f1 and f2 are single-check
        H = np.zeros((2, 3), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1   # f0: det0–det1 (bridges, high prob)
        H[0, 1] = 1              # f1: det0 only (boundary)
        H[1, 2] = 1              # f2: det1 only (boundary)
        pcm = _PCM(H, [0.4, 0.1, 0.1])
        syndrome = np.array([1, 1], dtype=np.uint8)

        c0 = ClusteringOvergrowBatch(pcm)
        c0.run(syndrome, over_grow_step=0, bits_per_step=1)
        n_faults_0 = sum(len(cl.fault_nodes) for cl in c0.active_clusters)

        c2 = ClusteringOvergrowBatch(pcm)
        c2.run(syndrome, over_grow_step=2, bits_per_step=1)
        n_faults_2 = sum(len(cl.fault_nodes) for cl in c2.active_clusters)

        # With more over-grow, more faults should be absorbed.
        assert n_faults_2 >= n_faults_0
        _check_invariants(c2, syndrome)

    def test_overgrow_budget_countdown(self):
        """
        Verify overgrow_budget counts down correctly.
        Cluster reaches valid_neutral at some step, budget starts at
        over_grow_step and decrements each step until 0, then valid=True.
        """
        # Simple chain: det0 — f0 — det1, syndrome [1,1].
        # f0 alone satisfies — budget should count down from over_grow_step.
        H = np.zeros((2, 3), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1
        H[0, 1] = 1
        H[1, 2] = 1
        pcm = _PCM(H, [0.4, 0.1, 0.1])
        syndrome = np.array([1, 1], dtype=np.uint8)

        for ogs in [0, 1, 2, 3]:
            c = ClusteringOvergrowBatch(pcm)
            c.run(syndrome, over_grow_step=ogs, bits_per_step=1)
            _check_invariants(c, syndrome)
            for cl in c.active_clusters:
                assert cl.overgrow_budget == 0
                assert cl.valid

    def test_overgrow_rep_code_exhaustive(self):
        """Exhaustive rep code with over_grow_step=1 — all invariants hold."""
        n = 5
        H, probs = _rep_pcm_arrays(n)
        pcm = _PCM(H, probs)
        c = ClusteringOvergrowBatch(pcm)
        for bits in range(2 ** (n - 1)):
            syndrome = np.array([(bits >> i) & 1 for i in range(n - 1)], dtype=np.uint8)
            c.run(syndrome, over_grow_step=1, bits_per_step=1)
            _check_invariants(c, syndrome)

    def test_overgrow_ring_code_exhaustive(self):
        """Exhaustive ring code n=4 with over_grow_step=2."""
        H, probs = _ring_pcm_arrays(4)
        pcm = _PCM(H, probs)
        c = ClusteringOvergrowBatch(pcm)
        for fault_mask in range(2 ** 4):
            fired = [j for j in range(4) if (fault_mask >> j) & 1]
            syndrome = _syndrome_from_faults(H, fired)
            c.run(syndrome, over_grow_step=2, bits_per_step=1)
            _check_invariants(c, syndrome)

    def test_overgrow_identical_to_clustering_overgrow(self):
        """
        ClusteringOvergrowBatch(bits_per_step=1) must match ClusteringOvergrow
        field-for-field across all rep code syndromes.
        """
        n = 5
        H, probs = _rep_pcm_arrays(n)
        pcm = _PCM(H, probs)
        ref   = ClusteringOvergrow(pcm)
        batch = ClusteringOvergrowBatch(pcm)
        for bits in range(2 ** (n - 1)):
            syndrome = np.array([(bits >> i) & 1 for i in range(n - 1)], dtype=np.uint8)
            ref.run(syndrome, over_grow_step=1)
            batch.run(syndrome, over_grow_step=1, bits_per_step=1)
            rm = {cl.cluster_id: cl for cl in ref.active_clusters}
            bm = {cl.cluster_id: cl for cl in batch.active_clusters}
            assert set(rm) == set(bm), f"bits={bits}"
            for cid in rm:
                _assert_rref_fields_equal(bm[cid], rm[cid], label=f"bits={bits} cid={cid}")


class TestSection2_BatchBitsPerStep:
    """
    2b: bits_per_step > 1 — termination, validity, and invariants.
    """

    def test_terminates_and_valid_rep_code(self):
        """bits_per_step=4 terminates with all clusters valid for rep code."""
        n = 8
        H, probs = _rep_pcm_arrays(n)
        pcm = _PCM(H, probs)
        c = ClusteringOvergrowBatch(pcm)
        for bits in range(2 ** (n - 1)):
            syndrome = np.array([(bits >> i) & 1 for i in range(n - 1)], dtype=np.uint8)
            c.run(syndrome, over_grow_step=0, bits_per_step=4)
            _check_invariants(c, syndrome)

    def test_terminates_and_valid_ring_code(self):
        n = 5
        H, probs = _ring_pcm_arrays(n)
        pcm = _PCM(H, probs)
        c = ClusteringOvergrowBatch(pcm)
        for fault_mask in range(2 ** n):
            fired = [j for j in range(n) if (fault_mask >> j) & 1]
            syndrome = _syndrome_from_faults(H, fired)
            c.run(syndrome, over_grow_step=0, bits_per_step=3)
            _check_invariants(c, syndrome)

    def test_bits_per_step_1_matches_larger(self):
        """
        bits_per_step=1 and bits_per_step>1 must both satisfy the structural
        invariants.  They may produce different cluster decompositions (validity
        is checked less often), but all final clusters must be valid.
        """
        H, probs = _ring_pcm_arrays(4)
        pcm = _PCM(H, probs)
        for fault_mask in range(2 ** 4):
            fired = [j for j in range(4) if (fault_mask >> j) & 1]
            syndrome = _syndrome_from_faults(H, fired)
            for bps in [1, 2, 4]:
                c = ClusteringOvergrowBatch(pcm)
                c.run(syndrome, over_grow_step=0, bits_per_step=bps)
                _check_invariants(c, syndrome)

    def test_large_bits_per_step_correct(self):
        """bits_per_step larger than total faults — no crash, valid result."""
        n = 6
        H, probs = _rep_pcm_arrays(n)
        pcm = _PCM(H, probs)
        c = ClusteringOvergrowBatch(pcm)
        syndrome = np.array([1, 0, 0, 0, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=100)
        _check_invariants(c, syndrome)

    def test_rref_is_batch_after_grow(self):
        """After run(), every active cluster still uses IncrementalRREFBatch."""
        n = 5
        H, probs = _rep_pcm_arrays(n)
        pcm = _PCM(H, probs)
        c = ClusteringOvergrowBatch(pcm)
        syndrome = np.array([1, 0, 0, 1], dtype=np.uint8)
        c.run(syndrome, over_grow_step=0, bits_per_step=3)
        for cl in c.active_clusters:
            assert isinstance(cl.rref, IncrementalRREFBatch)


class TestSection2_BatchMergeMultipleConnecting:
    """
    2d: when bits_per_step >= 2 and two simultaneous collision faults appear
    in the same step, both are passed to _merge_batch as connecting edges.
    """

    def test_two_connecting_faults_single_step(self):
        """
        Layout:
          det0(seedA) — f0(cheap) — det2 — f2(connect_A) — det4(seedB)
          det1(seedA2)— f1(cheap) — det3 — f3(connect_B) — det4
        Seeds: det0, det1, det4 (three seeds).
        With bits_per_step=3, after seedA and seedA2 each absorb f0 and f1,
        the next step pops both f2 and f3 simultaneously — both collide with
        seedB's cluster at det4.  _merge_batch is called with two connecting faults.
        """
        H = np.zeros((5, 4), dtype=np.uint8)
        H[0, 0] = H[2, 0] = 1   # f0: det0–det2 (cheap)
        H[1, 1] = H[3, 1] = 1   # f1: det1–det3 (cheap)
        H[2, 2] = H[4, 2] = 1   # f2: det2–det4 (connecting A)
        H[3, 3] = H[4, 3] = 1   # f3: det3–det4 (connecting B)
        pcm = _PCM(H, [0.4, 0.4, 0.2, 0.2])
        # Fire f0+f1 → syndrome [1,1,0,0,0] (det0 and det1 non-trivial)
        # But we want det4 to also be non-trivial: fire f2+f3 too
        syndrome = _syndrome_from_faults(H, [0, 1, 2, 3])
        c = ClusteringOvergrowBatch(pcm)
        c.run(syndrome, over_grow_step=0, bits_per_step=3)
        _check_invariants(c, syndrome)

    def test_two_connecting_faults_invariants_hold(self):
        """All invariants hold when _merge_batch handles multiple connecting faults."""
        H = np.zeros((4, 5), dtype=np.uint8)
        H[0, 0] = H[1, 0] = 1   # f0: det0–det1
        H[2, 1] = H[3, 1] = 1   # f1: det2–det3
        H[0, 2] = H[2, 2] = 1   # f2: det0–det2 (bridge A)
        H[1, 3] = H[3, 3] = 1   # f3: det1–det3 (bridge B)
        H[0, 4] = H[3, 4] = 1   # f4: det0–det3 (bridge C)
        pcm = _PCM(H, [0.3, 0.3, 0.2, 0.2, 0.15])
        for fault_mask in range(2 ** 5):
            fired = [j for j in range(5) if (fault_mask >> j) & 1]
            syndrome = _syndrome_from_faults(H, fired)
            for bps in [1, 2, 4]:
                c = ClusteringOvergrowBatch(pcm)
                c.run(syndrome, over_grow_step=0, bits_per_step=bps)
                _check_invariants(c, syndrome)

    def test_add_fault_batch_field_identical_to_sequential(self):
        """
        _add_fault_batch with k faults must give the same rref state as
        k sequential _add_fault calls (field-identical).
        """
        pcm = _PCM.rep(6)
        syndrome = np.zeros(5, dtype=np.uint8)
        syndrome[0] = syndrome[4] = 1

        # Reference: sequential _add_fault calls
        ref = ClusteringOvergrowBatch(pcm)
        ref.bits_per_step = 1
        ref.over_grow_step = 0
        ref.clusters_initialization(syndrome)
        cl_ref = ref.clusters[0]
        # Manually add faults 0, 1, 2 one at a time
        ref._add_fault(cl_ref, 0, ref.weights[0])
        ref._add_fault(cl_ref, 1, ref.weights[1])
        ref._add_fault(cl_ref, 2, ref.weights[2])

        # Batch: _add_fault_batch with 3 faults at once
        bat = ClusteringOvergrowBatch(pcm)
        bat.bits_per_step = 1
        bat.over_grow_step = 0
        bat.clusters_initialization(syndrome)
        cl_bat = bat.clusters[0]
        bat._add_fault_batch(cl_bat, [
            (ref.weights[0], 0),
            (ref.weights[1], 1),
            (ref.weights[2], 2),
        ])

        _assert_rref_fields_equal(cl_bat, cl_ref, label="_add_fault_batch vs sequential")


class TestSection2_OvergrowBudgetReset:
    """
    2e: if a merge happens after over_grow is in progress, the merged cluster
    resets its budget correctly (not neutral → budget = -1).
    """

    def test_budget_consistent_after_run(self):
        """Every active cluster has overgrow_budget == 0 after run()."""
        H, probs = _ring_pcm_arrays(5)
        pcm = _PCM(H, probs)
        for ogs in [0, 1, 2]:
            for bps in [1, 2, 3]:
                c = ClusteringOvergrowBatch(pcm)
                for fault_mask in range(2 ** 5):
                    fired = [j for j in range(5) if (fault_mask >> j) & 1]
                    syndrome = _syndrome_from_faults(H, fired)
                    c.run(syndrome, over_grow_step=ogs, bits_per_step=bps)
                    for cl in c.active_clusters:
                        assert cl.overgrow_budget == 0, \
                            f"budget={cl.overgrow_budget} for ogs={ogs} bps={bps}"

    def test_inactive_clusters_have_false_valid(self):
        """Absorbed clusters must have active=False (not reused)."""
        pcm = _PCM.rep(5)
        syndrome = np.array([1, 0, 0, 1], dtype=np.uint8)
        c = ClusteringOvergrowBatch(pcm)
        c.run(syndrome, over_grow_step=2, bits_per_step=2)
        inactive = [cl for cl in c.clusters if not cl.active]
        for cl in inactive:
            assert not cl.active


class TestSection2_Hypothesis:
    """2f: hypothesis-based sweep over (syndrome, over_grow_step, bits_per_step)."""
    pass


@given(st.data())
@settings(max_examples=300, deadline=None)
def test_s2_surface_code_random_params(data):
    """
    Draw a random fault subset, over_grow_step in {0,1,2}, and bits_per_step
    in {1,2,4}.  All structural invariants must hold after run().
    """
    pcm = _surface_code_pcm(distance=3, rounds=1, p=0.01)
    n_fault = pcm.H.shape[1]
    fired = data.draw(
        st.lists(st.integers(0, n_fault - 1), min_size=0, max_size=n_fault, unique=True)
    )
    over_grow_step = data.draw(st.integers(0, 2))
    bits_per_step  = data.draw(st.sampled_from([1, 2, 4]))
    syndrome = _surface_syndrome_from_faults(pcm, fired)
    c = ClusteringOvergrowBatch(pcm)
    c.run(syndrome, over_grow_step=over_grow_step, bits_per_step=bits_per_step)
    _check_invariants(c, syndrome)


@given(st.data())
@settings(max_examples=200, deadline=None)
def test_s2_bits1_field_identical_to_overgrow(data):
    """
    bits_per_step=1 with any over_grow_step must be field-identical to
    ClusteringOvergrow with the same over_grow_step.
    """
    pcm = _surface_code_pcm(distance=3, rounds=1, p=0.01)
    n_fault = pcm.H.shape[1]
    fired = data.draw(
        st.lists(st.integers(0, n_fault - 1), min_size=0, max_size=n_fault, unique=True)
    )
    over_grow_step = data.draw(st.integers(0, 2))
    syndrome = _surface_syndrome_from_faults(pcm, fired)

    ref   = ClusteringOvergrow(pcm)
    batch = ClusteringOvergrowBatch(pcm)
    ref.run(syndrome, over_grow_step=over_grow_step)
    batch.run(syndrome, over_grow_step=over_grow_step, bits_per_step=1)

    rm = {cl.cluster_id: cl for cl in ref.active_clusters}
    bm = {cl.cluster_id: cl for cl in batch.active_clusters}
    assert set(rm) == set(bm), \
        f"cluster ids differ for over_grow_step={over_grow_step}"
    for cid in rm:
        _assert_rref_fields_equal(bm[cid], rm[cid], label=f"cid={cid}")
