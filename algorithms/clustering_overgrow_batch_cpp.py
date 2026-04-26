"""
clustering_overgrow_batch_cpp.py  —  C++ drop-in for ClusteringOvergrowBatch.

Wraps the ClusteringOvgBatchEngine pybind11 class to provide the same public
interface as clustering_overgrow_batch.py::ClusteringOvergrowBatch:

    from clustering_overgrow_batch_cpp import ClusteringOvergrowBatch

    c = ClusteringOvergrowBatch(pcm)
    c.run(syndrome, over_grow_step=2, bits_per_step=4)
    c.active_valid_clusters       # dict[int, ClusterStateOGB]
    c.active_clusters             # list[ClusterStateOGB]
    c.create_degenerate_cycle_regions()
    c.run_and_create_degenerate_cycle_regions(syndrome, over_grow_step, bits_per_step)

The logical matrix pcm.L (if present) is extracted once at construction and
precomputed into a compact column-packed form inside the C++ engine, so
create_degenerate_cycle_regions() needs no additional Python-side L handling.

Lifetime note: ClusterStateOGB objects returned by the engine are references
into C++ internal storage.  They are invalidated by the next call to run().
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Locate _clustering_cpp.so (built by prior_gap_estimation_cpp/setup.py).
_cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "prior_gap_estimation_cpp")
if _cpp_dir not in sys.path:
    sys.path.insert(0, _cpp_dir)

from _clustering_cpp import ClusteringOvgBatchEngine  # noqa: E402

from parity_matrix_construct import ParityCheckMatrices  # noqa: E402


class ClusteringOvergrowBatch:
    """
    C++ ClusteringOvgBatch with the same interface as the Python
    ClusteringOvergrowBatch.

    Construct once from a PCM; call run(syndrome, ...) for each shot.
    """

    def __init__(self, pcm: ParityCheckMatrices) -> None:
        self.pcm     = pcm
        self.n_det   = int(pcm.H.shape[0])
        self.n_fault = int(pcm.H.shape[1])

        check_to_faults, fault_to_checks, weights = self._build_adjacency()

        # n_logical: number of logical bridge check nodes (last n_logical
        # detector indices are excluded from the RREF in C++).
        # Prefer pcm.n_logical_check_nodes; fall back to pcm.L.shape[0].
        n_logical = getattr(pcm, 'n_logical_check_nodes', 0)

        # Extract logical matrix L if available.
        # pcm.L has shape (n_obs, n_fault); passed as list-of-lists.
        # n_obs should equal n_logical (one logical check node per observable).
        L_rows: list[list[int]] = []
        if hasattr(pcm, 'L') and pcm.L is not None:
            L_arr = np.asarray(pcm.L, dtype=np.uint8)
            if n_logical == 0:
                n_logical = L_arr.shape[0]   # fall back if attribute missing
            L_rows = [L_arr[i, :].tolist() for i in range(L_arr.shape[0])]

        self._engine = ClusteringOvgBatchEngine(
            self.n_det, self.n_fault,
            check_to_faults, fault_to_checks,
            weights.tolist(),
            n_logical,
            L_rows,
        )

        # Expose adjacency for test generators (mirrors clustering_cpp.py).
        self.check_to_faults = check_to_faults
        self.fault_to_checks = fault_to_checks
        self.weights         = weights

        self.active_valid_clusters: dict = {}

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def run(self, syndrome: np.ndarray,
            over_grow_step: int = 0,
            bits_per_step:  int = 1) -> None:
        """Run clustering for one syndrome vector."""
        self._engine.run(
            np.asarray(syndrome, dtype=np.uint8),
            over_grow_step,
            bits_per_step,
        )
        avc = self._engine.active_valid_clusters
        has_L = hasattr(self.pcm, 'L') and self.pcm.L is not None
        for cl in avc.values():
            cl.L = (
                self.pcm.L[:, cl.cluster_fault_idx_to_pcm_fault_idx]
                if has_L else None
            )
        self.active_valid_clusters = avc

    @property
    def active_clusters(self):
        return self._engine.active_clusters

    @property
    def clusters(self):
        return self._engine.active_clusters

    def create_degenerate_cycle_regions(self) -> dict[int, dict[str, list]]:
        """
        Classify each null-space basis vector in every active valid cluster
        as a logical error or a stabilizer.

        Uses the C++ implementation (precomputed L_col_packed inside the
        engine) — no Python-side L @ z computation.

        Returns the same dict structure as the Python version:
          { cluster_id: {'logical_error': [[z, flip_int], ...],
                         'stabilizer':    [[z], ...]} }
        """
        return self._engine.create_degenerate_cycle_regions()

    def run_and_create_degenerate_cycle_regions(
        self,
        syndrome: np.ndarray,
        over_grow_step: int = 0,
        bits_per_step:  int = 1,
    ) -> dict[int, dict[str, list]]:
        """run() then create_degenerate_cycle_regions() in one call."""
        result = self._engine.run_and_create_degenerate_cycle_regions(
            np.asarray(syndrome, dtype=np.uint8),
            over_grow_step,
            bits_per_step,
        )
        self.active_valid_clusters = self._engine.active_valid_clusters
        has_L = hasattr(self.pcm, 'L') and self.pcm.L is not None
        for cl in self.active_valid_clusters.values():
            cl.L = (
                self.pcm.L[:, cl.cluster_fault_idx_to_pcm_fault_idx]
                if has_L else None
            )
        return result

    def get_active_valid_clusters(self) -> None:
        avc = self._engine.active_valid_clusters
        has_L = hasattr(self.pcm, 'L') and self.pcm.L is not None
        for cl in avc.values():
            cl.L = (
                self.pcm.L[:, cl.cluster_fault_idx_to_pcm_fault_idx]
                if has_L else None
            )
        self.active_valid_clusters = avc

    # -----------------------------------------------------------------------
    # Build adjacency (identical to clustering_cpp.py)
    # -----------------------------------------------------------------------

    def _build_adjacency(self):
        n_det   = self.n_det
        n_fault = self.n_fault

        check_to_faults: list[list[int]] = [[] for _ in range(n_det)]
        fault_to_checks: list[list[int]] = [[] for _ in range(n_fault)]

        for j, ed in enumerate(self.pcm.error_data):
            for i in ed['detectors']:
                check_to_faults[i].append(j)
                fault_to_checks[j].append(i)

        weights = np.array(
            [np.log((1 - ed['prob']) / ed['prob'])
             if 0 < ed['prob'] < 1 else float('inf')
             for ed in self.pcm.error_data],
            dtype=np.float64,
        )
        return check_to_faults, fault_to_checks, weights
