"""
prior_gap_estimation_use.py  —  Clustering-based gap estimator.

Extends PriorGapEstimator with a clustering engine and gap computation logic.
The gap is derived from logical-error null-space basis vectors produced by the
clustering algorithm, as an alternative or complement to the desaturation gap.

Gap types
---------
1  Binary       : -inf if any logical-error null-basis exists, else +inf.
2  Hamming      : minimum Hamming weight of logical-error null-basis vectors,
                  aggregated across clusters by 'min' or 'sum'.
4  Prior weight : minimum prior weight (sum of -log(p/(1-p)) for active faults)
                  of logical-error null-basis vectors, aggregated by 'min' or 'sum'.

Usage
-----
estimator = PriorGapEstimatorUse.from_desaturation(
    raw_dem,
    obs_det_type='per_obs',
    clustering_type='cplus_overgrow_batch',
    over_grow_step=15,
    bits_per_step=4,
)
gap, nonzero_count = estimator.execute(syndrome, gap_type=4, aggregate='min')
"""
from __future__ import annotations

import numpy as np

from prior_gap_estimation import PriorGapEstimator


# ---------------------------------------------------------------------------
# Local helpers (duplicated from algorithm_stats.py to keep file self-contained)
# ---------------------------------------------------------------------------

def _get_clustering_impl(clustering_type: str):
    if clustering_type == 'python_original':
        from clustering import Clustering as ClusteringImpl
        return ClusteringImpl
    if clustering_type == 'cplus_original':
        from clustering_cpp import Clustering as ClusteringImpl
        return ClusteringImpl
    if clustering_type == 'python_overgrow':
        from clustering_overgrow import Clustering as ClusteringImpl
        return ClusteringImpl
    if clustering_type == 'python_overgrow_batch':
        from clustering_overgrow_batch import ClusteringOvergrowBatch as ClusteringImpl
        return ClusteringImpl
    if clustering_type == 'cplus_overgrow_batch':
        from clustering_overgrow_batch_cpp import ClusteringOvergrowBatch as ClusteringImpl
        return ClusteringImpl
    raise ValueError(
        f"clustering_type must be one of 'python_original', 'cplus_original', "
        f"'python_overgrow', 'python_overgrow_batch', or 'cplus_overgrow_batch', "
        f"got {clustering_type!r}"
    )


def _make_run_kwargs(clustering_type: str, over_grow_step: int, bits_per_step: int) -> dict:
    if clustering_type in ('python_overgrow_batch', 'cplus_overgrow_batch'):
        return {'over_grow_step': over_grow_step, 'bits_per_step': bits_per_step}
    if clustering_type == 'python_overgrow':
        return {'over_grow_step': over_grow_step}
    return {}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PriorGapEstimatorUse(PriorGapEstimator):
    """
    PriorGapEstimator extended with clustering-based gap computation.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The modified DEM (output of modify_dem_as_desaturation).
    clustering_type : str
        Which clustering backend to use (see _get_clustering_impl).
    over_grow_step : int
        Passed to clustering.run() for overgrow variants.
    bits_per_step : int
        Passed to clustering.run() for batch overgrow variants.
    decompose : bool
        Forwarded to ParityCheckMatrices.from_DEM.
    """

    def __init__(
        self,
        dem,
        clustering_type: str = 'python_original',
        decompose: bool = False,
    ) -> None:
        super().__init__(dem, decompose=decompose)
        self.clustering_type = clustering_type

        ClusteringImpl  = _get_clustering_impl(clustering_type)
        self.Clustering = ClusteringImpl(self.PCM)

    @classmethod
    def from_desaturation(cls, dem, obs_det_type: str = 'per_obs', **kwargs):
        """
        Build from a raw DEM, applying the same desaturation pipeline as the
        base class.  Extra keyword arguments are forwarded to __init__.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
        obs_det_type : str
            Passed to modify_dem_as_desaturation.
        **kwargs
            clustering_type, decompose.
        """
        modified_dem = cls.modify_dem_as_desaturation(dem, obs_det_type=obs_det_type)
        return cls(modified_dem, **kwargs)

    # ------------------------------------------------------------------
    # Per-gap-type helpers (all operate on regions after run())
    # ------------------------------------------------------------------

    def _gap_binary(self, regions: dict) -> float:
        """Type 1: -inf if any logical-error null-basis exists, else +inf."""
        for groups in regions.values():
            if groups['logical_error']:
                return float('-inf')
        return float('inf')

    def _gap_hamming(self, regions: dict, aggregate: str) -> float:
        """
        Type 2: per cluster, minimum Hamming weight over logical-error null-basis
        vectors.  Aggregated across clusters by 'min' or 'sum'.
        """
        cluster_gaps = []
        for groups in regions.values():
            logical_vecs = groups['logical_error']
            if not logical_vecs:
                continue
            min_hw = min(int(z.sum()) for z, *_ in logical_vecs)
            cluster_gaps.append(float(min_hw))

        if not cluster_gaps:
            return float('inf')
        return min(cluster_gaps) if aggregate == 'min' else sum(cluster_gaps)

    def _gap_prior(self, regions: dict, aggregate: str) -> float:
        """
        Type 3: per cluster, minimum prior weight over logical-error null-basis
        vectors.  Prior weight of a vector z = sum of -log(p/(1-p)) for each
        fault index that is 1 in z.  Aggregated across clusters by 'min' or 'sum'.
        """
        weights  = self.Clustering.weights
        avc      = self.Clustering.active_valid_clusters
        cluster_gaps = []

        for cl_id, groups in regions.items():
            logical_vecs = groups['logical_error']
            if not logical_vecs:
                continue
            fault_map = avc[cl_id].cluster_fault_idx_to_pcm_fault_idx
            min_pw = min(
                float(sum(weights[fault_map[j]] for j in range(len(z)) if z[j]))
                for z, *_ in logical_vecs
            )
            cluster_gaps.append(min_pw)

        if not cluster_gaps:
            return float('inf')
        return min(cluster_gaps) if aggregate == 'min' else sum(cluster_gaps)

    def _gap_weight_diff(self, regions: dict, aggregate: str, asb: bool = False) -> float:
        """
        Per cluster: minimum signed weight difference weight(e2) - weight(e1) over
        all logical-error null-basis vectors z, where e1 is the RREF pivot solution
        and e2 = e1 ⊕ z.

        weight(e2) - weight(e1)
            = Σ w[j]  for j where z[j]=1 and e1[j]=0   (faults added by z)
            - Σ w[j]  for j where z[j]=1 and e1[j]=1   (faults removed by z)

        If asb=True, uses |weight(e2) - weight(e1)| instead — always non-negative.
        If asb=False (default), the signed value is returned; negative means the
        RREF chose the heavier (wrong) solution.
        Aggregated across clusters by 'min' or 'sum'.
        """
        weights = self.Clustering.weights
        avc     = self.Clustering.active_valid_clusters
        cluster_gaps = []

        for cl_id, groups in regions.items():
            logical_vecs = groups['logical_error']
            if not logical_vecs:
                continue
            cl        = avc[cl_id]
            fault_map = cl.cluster_fault_idx_to_pcm_fault_idx
            e1        = self._get_correction(cl)

            min_diff = min(
                float(abs(
                    sum(weights[fault_map[j]] for j in range(len(z)) if z[j] and not e1[j])
                  - sum(weights[fault_map[j]] for j in range(len(z)) if z[j] and     e1[j])
                ) if asb else (
                    sum(weights[fault_map[j]] for j in range(len(z)) if z[j] and not e1[j])
                  - sum(weights[fault_map[j]] for j in range(len(z)) if z[j] and     e1[j])
                ))
                for z, *_ in logical_vecs
            )
            cluster_gaps.append(min_diff)

        if not cluster_gaps:
            return float('inf')
        return min(cluster_gaps) if aggregate == 'min' else sum(cluster_gaps)

    def _compute_gap(self, regions: dict, gap_type: str, aggregate: str,
                     asb: bool = False) -> float:
        if gap_type == 'binary':
            return self._gap_binary(regions)
        if gap_type == 'hamming':
            return self._gap_hamming(regions, aggregate)
        if gap_type == 'prior_weight':
            return self._gap_prior(regions, aggregate)
        if gap_type == 'weight_diff':
            return self._gap_weight_diff(regions, aggregate, asb=asb)
        raise ValueError(
            f"gap_type must be 'binary', 'hamming', 'prior_weight', or 'weight_diff'; "
            f"got {gap_type!r}"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        syndrome: np.ndarray,
        gap_type: str = 'binary',
        aggregate: str = 'min',
        over_grow_step: int = 0,
        bits_per_step: int = 1,
        asb: bool = False,
        decode: bool = False,
    ):
        """
        Run clustering on one syndrome and return the clustering gap.

        Parameters
        ----------
        syndrome : np.ndarray of uint8, shape (n_detectors,)
        gap_type : {'binary', 'hamming', 'prior_weight', 'weight_diff'}
        aggregate : {'min', 'sum'}
            How to aggregate cluster-level gaps into a shot-level gap.
            Ignored for gap_type='binary'.
        over_grow_step : int
            Extra growth steps after valid_neutral.
        bits_per_step : int
            Faults per over-grow step for batch overgrow types.
        asb : bool
            If True, use |weight_diff| instead of the signed value.
            Only applies to gap_type='weight_diff'.
        decode : bool
            If True, also extract cluster corrections and return the overall
            logical flip (clustering-based decoded prediction).
            Must be True when gap_type='weight_diff'.

        Returns
        -------
        decode=False : (gap, nonzero_count)
            gap          : float
            nonzero_count: int

        decode=True  : (gap, overall_logical_flip, nonzero_count)
            gap                 : float
            overall_logical_flip: np.ndarray(n_obs,) uint8 — decoded prediction
            nonzero_count       : int
        """
        if gap_type == 'weight_diff' and not decode:
            raise ValueError("gap_type='weight_diff' requires decode=True")

        run_kwargs = _make_run_kwargs(self.clustering_type, over_grow_step, bits_per_step)
        regions = self.Clustering.run_and_create_degenerate_cycle_regions(
            syndrome, **run_kwargs
        )
        nonzero_count = sum(len(groups['logical_error']) for groups in regions.values())

        if decode:
            sol = self.get_cluster_solutions()
            overall_logical_flip = sol['overall_logical_flip']

        gap = self._compute_gap(regions, gap_type, aggregate, asb=asb)

        if decode:
            return gap, overall_logical_flip, nonzero_count
        return gap, nonzero_count

    # ------------------------------------------------------------------
    # Clustering-based decoder
    # ------------------------------------------------------------------

    @staticmethod
    def _get_correction(cl) -> np.ndarray:
        """Extract the RREF pivot solution for one cluster (local fault indices)."""
        corr = np.zeros(cl.rref.n_bits, dtype=np.uint8)
        for i, pm in enumerate(cl.rref.pivot_map):
            if pm is not None and cl.rref.s_prime[i] == 1:
                corr[pm] = 1
        return corr

    def get_cluster_solutions(self) -> dict:
        """
        Extract the local pivot solution for every active valid cluster.

        Must be called after execute() — reads from the clustering state
        already populated by that call.

        Returns
        -------
        dict with keys:
            'local_solutions'           : dict[cluster_id, np.ndarray(n_local_faults,)]
                Local correction vector for each cluster (local fault index space).
            'logical_flip_per_cluster'  : dict[cluster_id, np.ndarray(n_obs,)]
                Logical observable flip caused by each cluster's local correction.
            'overall_solution'          : np.ndarray(n_total_faults,)
                Global correction in PCM fault index space (union of all cluster
                corrections; clusters have disjoint fault sets by invariant).
            'overall_logical_flip'      : np.ndarray(n_obs,)
                XOR of all clusters' logical flips — the decoded prediction.
        """
        n_faults = self.PCM.H.shape[1]
        n_obs    = self.PCM.L.shape[0]

        local_solutions:          dict[int, np.ndarray] = {}
        logical_flip_per_cluster: dict[int, np.ndarray] = {}
        overall_solution     = np.zeros(n_faults, dtype=np.uint8)
        overall_logical_flip = np.zeros(n_obs,    dtype=np.uint8)

        for cl_id, cl in self.Clustering.active_valid_clusters.items():
            e1   = self._get_correction(cl)
            flip = (cl.L @ e1) % 2 if cl.L is not None else np.zeros(n_obs, dtype=np.uint8)

            local_solutions[cl_id]          = e1
            logical_flip_per_cluster[cl_id] = flip

            for local_j, val in enumerate(e1):
                if val:
                    overall_solution[cl.cluster_fault_idx_to_pcm_fault_idx[local_j]] = 1

            overall_logical_flip ^= flip

        return {
            'local_solutions':          local_solutions,
            'logical_flip_per_cluster': logical_flip_per_cluster,
            'overall_solution':         overall_solution,
            'overall_logical_flip':     overall_logical_flip,
        }
