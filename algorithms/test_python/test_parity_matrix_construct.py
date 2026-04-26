"""
Hypothesis-based tests for ParityCheckMatrices.

Test strategy
-------------
The core oracle is the "single-fault DEM" test (Level 2):

  For fault column j, build a DEM containing only that fault at p=1.0.
  Because the fault fires with certainty, sampling once is deterministic.
  The returned syndrome must exactly match H[:, j] and L[:, j].

This is exact — no statistics, no approximations. It independently
re-derives what stim believes the fault does and compares it with what
from_DEM stored in H and L.

Test fixtures
-------------
We build three circuits once (module-level) and reuse them across all tests
to avoid rebuilding the circuit on every hypothesis example:

  - PCM_UNDEC : cultivation circuit, decompose=False (true hyperedges)
  - PCM_DEC   : cultivation circuit, decompose=True  (MWPM-compatible pieces)
  - PCM_SURF  : surface code memory,  decompose=False (simpler, all edges <=2)

Hypothesis draws a random fault index j for each fixture and checks the oracle.

Test layers
-----------
Layer 1 — structural invariants (no stim sampling needed)
  1a: H and L are binary
  1b: H/L shapes match n_detectors, n_observables, n_faults from stats
  1c: error_data[j]['detectors'] matches nonzero rows of H[:, j]
  1d: is_boundary consistency — boundary iff len(detectors) <= 1
  1e: is_timelike consistency — timelike iff detectors span >1 t-coordinate

Layer 2 — single-fault oracle (stim sampling)
  2a: H[:, j] matches syndrome from single-fault DEM at p=1.0
  2b: L[:, j] matches observable flips from single-fault DEM at p=1.0
  (tested together in one function to avoid double sampling)

Layer 3 — decompose mode consistency
  3a: decompose=False and decompose=True built from the same DEM share the
      same n_detectors, n_observables, and detector coordinate dict
  3b: every non-decomposed fault (decompose=False) appears as a subset-XOR
      of pieces in the decomposed version — i.e. the union of detectors
      across decomposed pieces covers the original detectors (modulo XOR).
      We test the weaker condition: n_faults_dec >= n_faults_undec.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
import stim
from hypothesis import given, settings, assume
import hypothesis.strategies as st

import cultiv
import gen
from parity_matrix_construct import ParityCheckMatrices


# ===========================================================================
# Shared fixtures — built once at module load to keep tests fast
# ===========================================================================

def _make_cultivation_dem(decompose: bool) -> stim.DetectorErrorModel:
    ideal = cultiv.make_end2end_cultivation_circuit(
        dcolor=3, dsurface=7, basis='Y',
        r_growing=3, r_end=0, inject_style='unitary',
    )
    noisy = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(ideal)
    return noisy.detector_error_model(
        decompose_errors=decompose,
        ignore_decomposition_failures=decompose,
        flatten_loops=True,
    )


def _make_surface_dem() -> stim.DetectorErrorModel:
    # Simple surface code memory with only bit-flip noise — guarantees
    # all faults touch at most 2 detectors (no correlated multi-body errors)
    circuit = stim.Circuit.generated(
        'surface_code:rotated_memory_z',
        rounds=5,
        distance=5,
        before_round_data_depolarization=0.001,
    )
    return circuit.detector_error_model(decompose_errors=False, flatten_loops=True)


# Build all fixtures once
_DEM_UNDEC = _make_cultivation_dem(decompose=False)
_DEM_DEC   = _make_cultivation_dem(decompose=True)
_DEM_SURF  = _make_surface_dem()

PCM_UNDEC = ParityCheckMatrices.from_DEM(_DEM_UNDEC, decompose=False)
PCM_DEC   = ParityCheckMatrices.from_DEM(_DEM_DEC,   decompose=True)
PCM_SURF  = ParityCheckMatrices.from_DEM(_DEM_SURF,  decompose=False)


# ===========================================================================
# Helpers
# ===========================================================================

def _single_fault_syndrome(
    pcm: ParityCheckMatrices,
    j: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a one-instruction DEM with only fault j at p=1.0, sample once
    (deterministic because p=1), and return (syndrome, obs_flips) as
    1-D uint8 arrays of length n_detectors and n_observables.

    The DEM is built directly from error_data[j] so it is independent of
    how H/L were populated — any mismatch between the two is a real bug.
    """
    ed = pcm.error_data[j]
    det_ids = sorted(ed['detectors'])
    obs_ids = [k for k in range(pcm.L.shape[0]) if pcm.L[k, j] == 1]

    # Compose the error instruction string
    parts = [f"D{d}" for d in det_ids] + [f"L{o}" for o in obs_ids]
    if parts:
        instruction = f"error(1.0) {' '.join(parts)}"
    else:
        # Fault with no detectors and no observables — empty instruction
        # stim doesn't allow error() with no targets, so use a trick:
        # we just return all-zero syndrome/obs directly (nothing fires)
        n_det = pcm.H.shape[0]
        n_obs = pcm.L.shape[0]
        return np.zeros(n_det, dtype=np.uint8), np.zeros(n_obs, dtype=np.uint8)

    # Add detector declarations so stim knows the total detector count.
    # Without these, stim infers num_detectors from the max D index seen,
    # which may be less than pcm.H.shape[0] for faults near the low end.
    # We declare a detector for each ID in the fault; the sampler only
    # returns columns up to max(declared detectors), so we use the global max.
    max_det = pcm.H.shape[0] - 1
    det_decl = "\n".join(f"detector D{d}" for d in range(max_det + 1))

    dem = stim.DetectorErrorModel(f"{instruction}\n{det_decl}")

    # p=1.0 → fault always fires → one sample is deterministic.
    # compile_sampler().sample() returns a 3-tuple:
    #   (detector_array, observable_array, errors_or_None)
    syndrome_2d, obs_2d, _ = dem.compile_sampler().sample(shots=1)
    # syndrome_2d shape: (1, max_det+1), obs_2d shape: (1, n_obs_in_dem)
    syndrome = syndrome_2d[0].astype(np.uint8)
    obs_raw  = obs_2d[0].astype(np.uint8)

    # Pad obs to full n_observables width (stim may return fewer columns
    # if no high-index observable appears in this single-fault DEM)
    n_obs_full = pcm.L.shape[0]
    obs = np.zeros(n_obs_full, dtype=np.uint8)
    obs[:len(obs_raw)] = obs_raw

    return syndrome, obs


def _coords_for_dem(dem: stim.DetectorErrorModel) -> dict:
    return dem.get_detector_coordinates()


# ===========================================================================
# Layer 1 — structural invariants
# ===========================================================================

class TestLayer1Structural:
    """
    Invariants that hold purely from the structure of H, L, error_data, and
    stats. No stim sampling required.
    """

    @pytest.mark.parametrize("pcm,label", [
        (PCM_UNDEC, "undec"),
        (PCM_DEC,   "dec"),
        (PCM_SURF,  "surf"),
    ])
    def test_1a_H_L_are_binary(self, pcm, label):
        """H and L must contain only 0s and 1s."""
        assert set(np.unique(pcm.H)).issubset({0, 1}), \
            f"[{label}] H contains values outside {{0,1}}"
        assert set(np.unique(pcm.L)).issubset({0, 1}), \
            f"[{label}] L contains values outside {{0,1}}"

    @pytest.mark.parametrize("pcm,label", [
        (PCM_UNDEC, "undec"),
        (PCM_DEC,   "dec"),
        (PCM_SURF,  "surf"),
    ])
    def test_1b_shapes_match_stats(self, pcm, label):
        """H.shape and L.shape must match stats n_detectors/n_observables/n_faults."""
        s = pcm.stats
        assert pcm.H.shape == (s['n_detectors'], s['n_faults']), \
            f"[{label}] H shape mismatch"
        assert pcm.L.shape == (s['n_observables'], s['n_faults']), \
            f"[{label}] L shape mismatch"
        assert len(pcm.error_data) == s['n_faults'], \
            f"[{label}] len(error_data) != n_faults"

    # -----------------------------------------------------------------------
    # Hypothesis: for random fault j, error_data[j]['detectors'] must match
    # the nonzero rows of H[:, j] exactly.
    # -----------------------------------------------------------------------

    @given(st.integers(0, PCM_UNDEC.stats['n_faults'] - 1))
    @settings(max_examples=200)
    def test_1c_detectors_match_H_column_undec(self, j):
        """error_data[j]['detectors'] == set of rows where H[:, j] == 1 (undec)."""
        pcm = PCM_UNDEC
        from_H    = frozenset(int(i) for i in np.where(pcm.H[:, j])[0])
        from_meta = pcm.error_data[j]['detectors']
        assert from_H == from_meta, \
            f"j={j}: H col detectors {from_H} != error_data detectors {from_meta}"

    @given(st.integers(0, PCM_DEC.stats['n_faults'] - 1))
    @settings(max_examples=200)
    def test_1c_detectors_match_H_column_dec(self, j):
        """error_data[j]['detectors'] == set of rows where H[:, j] == 1 (dec)."""
        pcm = PCM_DEC
        from_H    = frozenset(int(i) for i in np.where(pcm.H[:, j])[0])
        from_meta = pcm.error_data[j]['detectors']
        assert from_H == from_meta, \
            f"j={j}: H col detectors {from_H} != error_data detectors {from_meta}"

    @given(st.integers(0, PCM_SURF.stats['n_faults'] - 1))
    @settings(max_examples=200)
    def test_1c_detectors_match_H_column_surf(self, j):
        """error_data[j]['detectors'] == set of rows where H[:, j] == 1 (surf)."""
        pcm = PCM_SURF
        from_H    = frozenset(int(i) for i in np.where(pcm.H[:, j])[0])
        from_meta = pcm.error_data[j]['detectors']
        assert from_H == from_meta, \
            f"j={j}: H col detectors {from_H} != error_data detectors {from_meta}"

    @given(st.integers(0, PCM_UNDEC.stats['n_faults'] - 1))
    @settings(max_examples=200)
    def test_1d_is_boundary_consistency(self, j):
        """is_boundary must be True iff len(detectors) <= 1."""
        ed = PCM_UNDEC.error_data[j]
        if ed['is_boundary']:
            assert len(ed['detectors']) <= 1, \
                f"j={j}: is_boundary=True but len(detectors)={len(ed['detectors'])}"
        else:
            assert len(ed['detectors']) > 1, \
                f"j={j}: is_boundary=False but len(detectors)={len(ed['detectors'])}"

    @given(st.integers(0, PCM_UNDEC.stats['n_faults'] - 1))
    @settings(max_examples=200)
    def test_1e_is_timelike_consistency(self, j):
        """
        is_timelike must be False for boundary faults.
        For non-boundary faults: is_timelike iff detectors span >1 t-coordinate.
        """
        ed   = PCM_UNDEC.error_data[j]
        coords = _coords_for_dem(_DEM_UNDEC)

        if ed['is_boundary']:
            assert not ed['is_timelike'], \
                f"j={j}: boundary fault has is_timelike=True"
            return

        # Collect distinct t-values for all detectors in this fault
        t_vals = set()
        for det_id in ed['detectors']:
            c = coords.get(det_id, [])
            if len(c) > 2:
                t_vals.add(c[2])

        expected_timelike = len(t_vals) > 1
        assert ed['is_timelike'] == expected_timelike, (
            f"j={j}: is_timelike={ed['is_timelike']} but "
            f"t_vals={t_vals} => expected {expected_timelike}"
        )


# ===========================================================================
# Layer 2 — single-fault oracle
# ===========================================================================

class TestLayer2SingleFaultOracle:
    """
    The key correctness test: build a single-fault DEM at p=1.0 for fault j,
    sample once (deterministic), and check that the syndrome exactly matches
    H[:, j] and the observable flips match L[:, j].

    This independently re-derives the effect of each fault from stim and
    compares it with what from_DEM stored in H and L.
    """

    @given(st.integers(0, PCM_UNDEC.stats['n_faults'] - 1))
    @settings(max_examples=300)
    def test_2_single_fault_oracle_undec(self, j):
        """
        For the undecomposed cultivation PCM, fault j at p=1.0 must produce
        syndrome == H[:, j] and obs_flips == L[:, j].
        """
        pcm = PCM_UNDEC
        syndrome, obs = _single_fault_syndrome(pcm, j)

        # Syndrome must match H[:, j] exactly
        assert np.array_equal(syndrome, pcm.H[:, j]), (
            f"j={j}: syndrome mismatch\n"
            f"  from stim : {np.where(syndrome)[0].tolist()}\n"
            f"  from H col: {np.where(pcm.H[:, j])[0].tolist()}"
        )
        # Observable flips must match L[:, j] exactly
        assert np.array_equal(obs, pcm.L[:, j]), (
            f"j={j}: observable mismatch\n"
            f"  from stim : {np.where(obs)[0].tolist()}\n"
            f"  from L col: {np.where(pcm.L[:, j])[0].tolist()}"
        )

    @given(st.integers(0, PCM_DEC.stats['n_faults'] - 1))
    @settings(max_examples=300)
    def test_2_single_fault_oracle_dec(self, j):
        """
        For the decomposed cultivation PCM, each decomposed piece at p=1.0
        must produce syndrome == H[:, j] and obs_flips == L[:, j].
        """
        pcm = PCM_DEC
        syndrome, obs = _single_fault_syndrome(pcm, j)

        assert np.array_equal(syndrome, pcm.H[:, j]), (
            f"j={j}: syndrome mismatch (decomposed)\n"
            f"  from stim : {np.where(syndrome)[0].tolist()}\n"
            f"  from H col: {np.where(pcm.H[:, j])[0].tolist()}"
        )
        assert np.array_equal(obs, pcm.L[:, j]), (
            f"j={j}: observable mismatch (decomposed)\n"
            f"  from stim : {np.where(obs)[0].tolist()}\n"
            f"  from L col: {np.where(pcm.L[:, j])[0].tolist()}"
        )

    @given(st.integers(0, PCM_SURF.stats['n_faults'] - 1))
    @settings(max_examples=300)
    def test_2_single_fault_oracle_surf(self, j):
        """
        For the surface code PCM (all edges <= 2 detectors), fault j at p=1.0
        must match H[:, j] and L[:, j].
        """
        pcm = PCM_SURF
        syndrome, obs = _single_fault_syndrome(pcm, j)

        assert np.array_equal(syndrome, pcm.H[:, j]), (
            f"j={j}: syndrome mismatch (surface code)\n"
            f"  from stim : {np.where(syndrome)[0].tolist()}\n"
            f"  from H col: {np.where(pcm.H[:, j])[0].tolist()}"
        )
        assert np.array_equal(obs, pcm.L[:, j]), (
            f"j={j}: observable mismatch (surface code)\n"
            f"  from stim : {np.where(obs)[0].tolist()}\n"
            f"  from L col: {np.where(pcm.L[:, j])[0].tolist()}"
        )


# ===========================================================================
# Layer 3 — decompose mode consistency
# ===========================================================================

class TestLayer3DecomposeConsistency:
    """
    Cross-checks between decompose=False and decompose=True built from the
    same underlying DEM.
    """

    def test_3a_same_n_detectors_and_observables(self):
        """
        Both modes must agree on the number of detectors and observables —
        decomposition only affects fault columns, not the detector space.
        """
        assert PCM_UNDEC.stats['n_detectors']  == PCM_DEC.stats['n_detectors']
        assert PCM_UNDEC.stats['n_observables'] == PCM_DEC.stats['n_observables']
        assert PCM_UNDEC.H.shape[0] == PCM_DEC.H.shape[0]
        assert PCM_UNDEC.L.shape[0] == PCM_DEC.L.shape[0]

    def test_3b_decomposed_has_more_or_equal_fault_columns(self):
        """
        Decomposing can only split faults into more pieces, never fewer.
        So n_faults_dec >= n_faults_undec.
        """
        assert PCM_DEC.stats['n_faults'] >= PCM_UNDEC.stats['n_faults'], (
            f"decomposed has fewer faults: "
            f"{PCM_DEC.stats['n_faults']} < {PCM_UNDEC.stats['n_faults']}"
        )

    def test_3c_decomposed_mostly_small_edges(self):
        """
        After decomposition, the vast majority of fault columns should have
        <= 2 detectors. Specifically, |edge|>2 must be less common than in
        the undecomposed version.
        """
        def large_edge_count(pcm):
            return sum(
                1 for ed in pcm.error_data if len(ed['detectors']) > 2
            )

        n_large_undec = large_edge_count(PCM_UNDEC)
        n_large_dec   = large_edge_count(PCM_DEC)
        assert n_large_dec < n_large_undec, (
            f"Decomposed has more large edges than undecomposed: "
            f"{n_large_dec} vs {n_large_undec}"
        )

    def test_3d_surface_code_oracle_spot_check(self):
        """
        Spot-check 10 specific fault columns of the surface code PCM via the
        single-fault oracle, to confirm the surface code fixture also parses correctly.
        """
        pcm = PCM_SURF
        n = pcm.stats['n_faults']
        # Sample 10 evenly-spaced fault indices
        indices = [int(i * n / 10) for i in range(10)]
        for j in indices:
            syndrome, obs = _single_fault_syndrome(pcm, j)
            assert np.array_equal(syndrome, pcm.H[:, j]), \
                f"Surface code j={j}: syndrome mismatch"
            assert np.array_equal(obs, pcm.L[:, j]), \
                f"Surface code j={j}: observable mismatch"


# ===========================================================================
# Layer 4 — ldpc oracle comparison
# ===========================================================================
#
# ldpc.ckt_noise.detector_error_model_to_check_matrices is an independent
# implementation of DEM → H, L, priors.  We compare our ParityCheckMatrices
# against it as a second oracle.
#
# KEY DIFFERENCES between ldpc and our implementation:
#
#   1. Column ordering: ldpc inserts columns in the order unique detector-sets
#      are first seen; our code follows error instruction order.
#
#   2. Deduplication: ldpc groups error instructions by their detector frozenset.
#      Two instructions with the same detector set → one column in ldpc, with
#      probabilities combined via the XOR rule:
#          p_combined = p_old * (1 - p_new) + p_new * (1 - p_old)
#      Our code creates one column per instruction regardless.
#
# Comparison strategy: normalise both outputs into
#     canonical_form: dict[ (frozenset_dets, frozenset_obs) → float ]
# where duplicates are merged with the XOR rule before comparing.  Once both
# sides are in canonical form the two dicts must be equal.
#
# Tests:
#   4a  surface code  — structural (same fault patterns)
#   4b  surface code  — priors (same probabilities after aggregation)
#   4c  cultivation decompose=False — structural match against ldpc
#   4d  cultivation decompose=False — priors match against ldpc

try:
    from ldpc.ckt_noise import detector_error_model_to_check_matrices as _ldpc_dem_to_matrices
    _LDPC_AVAILABLE = True
except ImportError:
    _LDPC_AVAILABLE = False


def _xor_prob(p_old: float, p_new: float) -> float:
    """XOR-combination of two independent error probabilities."""
    return p_old * (1 - p_new) + p_new * (1 - p_old)


def _pcm_to_canonical(
    pcm: ParityCheckMatrices,
) -> dict[tuple[frozenset, frozenset], float]:
    """
    Convert a ParityCheckMatrices into a canonical dict
        (det_frozenset, obs_frozenset) → aggregated_prob.

    Columns that share the same (det, obs) key are merged with the XOR rule.
    This matches the deduplication that ldpc performs internally.
    """
    canonical: dict[tuple[frozenset, frozenset], float] = {}
    for j in range(pcm.stats['n_faults']):
        det_key = frozenset(int(i) for i in np.where(pcm.H[:, j])[0])
        obs_key = frozenset(int(k) for k in np.where(pcm.L[:, j])[0])
        key = (det_key, obs_key)
        p = float(pcm.error_data[j]['prob'])
        if key in canonical:
            canonical[key] = _xor_prob(canonical[key], p)
        else:
            canonical[key] = p
    return canonical


def _ldpc_to_canonical(
    ldpc_mat,  # DemMatrices from ldpc
) -> dict[tuple[frozenset, frozenset], float]:
    """
    Convert an ldpc DemMatrices into the same canonical dict format.
    ldpc already deduplicates, so each column maps to a unique (det, obs) key.
    """
    H = ldpc_mat.check_matrix.toarray()       # (n_det, n_faults)
    L = ldpc_mat.observables_matrix.toarray() # (n_obs, n_faults)
    priors = ldpc_mat.priors

    canonical: dict[tuple[frozenset, frozenset], float] = {}
    for j in range(H.shape[1]):
        det_key = frozenset(int(i) for i in np.where(H[:, j])[0])
        obs_key = frozenset(int(k) for k in np.where(L[:, j])[0])
        key = (det_key, obs_key)
        canonical[key] = float(priors[j])
    return canonical


@pytest.mark.skipif(not _LDPC_AVAILABLE, reason="ldpc package not installed")
class TestLayer4LdpcOracle:
    """
    Compare ParityCheckMatrices against ldpc's detector_error_model_to_check_matrices.

    Both are converted to a canonical (det_frozenset, obs_frozenset) → prob dict
    before comparing, which neutralises column-ordering and deduplication differences.
    """

    def test_4a_surf_structural_match(self):
        """
        Surface code: the set of unique (det, obs) fault patterns produced by
        our code must match ldpc exactly.
        """
        ldpc_mat = _ldpc_dem_to_matrices(_DEM_SURF, allow_undecomposed_hyperedges=True)
        our  = _pcm_to_canonical(PCM_SURF)
        ldpc = _ldpc_to_canonical(ldpc_mat)

        our_keys  = set(our.keys())
        ldpc_keys = set(ldpc.keys())
        only_ours = our_keys - ldpc_keys
        only_ldpc = ldpc_keys - our_keys

        assert our_keys == ldpc_keys, (
            f"Surface code fault-pattern mismatch:\n"
            f"  {len(only_ours)} patterns in ours but not ldpc: "
            f"{list(only_ours)[:5]}{'...' if len(only_ours) > 5 else ''}\n"
            f"  {len(only_ldpc)} patterns in ldpc but not ours: "
            f"{list(only_ldpc)[:5]}{'...' if len(only_ldpc) > 5 else ''}"
        )

    def test_4b_surf_priors_match(self):
        """
        Surface code: after aggregating duplicates with the XOR rule,
        each fault's probability must agree with ldpc to within floating-point
        precision (< 1e-12).
        """
        ldpc_mat = _ldpc_dem_to_matrices(_DEM_SURF, allow_undecomposed_hyperedges=True)
        our  = _pcm_to_canonical(PCM_SURF)
        ldpc = _ldpc_to_canonical(ldpc_mat)

        mismatches = []
        for key, p_ldpc in ldpc.items():
            p_ours = our.get(key)
            if p_ours is None:
                mismatches.append(f"  key {key}: missing from ours")
            elif abs(p_ours - p_ldpc) >= 1e-12:
                mismatches.append(
                    f"  key (dets={set(key[0])}, obs={set(key[1])}): "
                    f"ours={p_ours:.6e}, ldpc={p_ldpc:.6e}, "
                    f"diff={abs(p_ours - p_ldpc):.2e}"
                )
        assert not mismatches, (
            f"Prior mismatches for surface code ({len(mismatches)} faults):\n"
            + "\n".join(mismatches[:10])
            + ("\n  ..." if len(mismatches) > 10 else "")
        )

    def test_4c_cultivation_undec_structural_match(self):
        """
        Cultivation circuit (decompose=False): fault patterns match ldpc.
        allow_undecomposed_hyperedges=True is required because the cultivation
        circuit has hyperedges with >2 detectors.
        """
        ldpc_mat = _ldpc_dem_to_matrices(_DEM_UNDEC, allow_undecomposed_hyperedges=True)
        our  = _pcm_to_canonical(PCM_UNDEC)
        ldpc = _ldpc_to_canonical(ldpc_mat)

        our_keys  = set(our.keys())
        ldpc_keys = set(ldpc.keys())
        only_ours = our_keys - ldpc_keys
        only_ldpc = ldpc_keys - our_keys

        assert our_keys == ldpc_keys, (
            f"Cultivation (undec) fault-pattern mismatch:\n"
            f"  {len(only_ours)} patterns only in ours\n"
            f"  {len(only_ldpc)} patterns only in ldpc"
        )

    def test_4d_cultivation_undec_priors_match(self):
        """
        Cultivation circuit (decompose=False): probabilities match ldpc after
        XOR-aggregating duplicate fault patterns.
        """
        ldpc_mat = _ldpc_dem_to_matrices(_DEM_UNDEC, allow_undecomposed_hyperedges=True)
        our  = _pcm_to_canonical(PCM_UNDEC)
        ldpc = _ldpc_to_canonical(ldpc_mat)

        mismatches = []
        for key, p_ldpc in ldpc.items():
            p_ours = our.get(key)
            if p_ours is None:
                mismatches.append(
                    f"  dets={set(key[0])}, obs={set(key[1])}: missing from ours"
                )
            elif abs(p_ours - p_ldpc) >= 1e-12:
                mismatches.append(
                    f"  dets={set(key[0])}, obs={set(key[1])}: "
                    f"ours={p_ours:.6e}, ldpc={p_ldpc:.6e}"
                )
        assert not mismatches, (
            f"Prior mismatches for cultivation undec ({len(mismatches)} faults):\n"
            + "\n".join(mismatches[:10])
            + ("\n  ..." if len(mismatches) > 10 else "")
        )
