"""
clustering_cpp.py  —  Drop-in C++ replacement for clustering.py.

Wraps the pybind11 _clustering_cpp extension to provide the same public
interface as the Python Clustering class:

    from clustering_cpp import Clustering

    c = Clustering(pcm)          # build once from PCM
    c.run(syndrome)              # per-shot
    c.active_valid_clusters      # dict[int, ClusterState]
    c.active_clusters            # list[ClusterState]
    c.create_degenerate_cycle_regions()  # same as Python version

ClusterState objects returned by the C++ engine expose the same attributes
as the Python ClusterState:
    .cluster_id, .active, .valid
    .fault_nodes, .check_nodes
    .cluster_fault_idx_to_pcm_fault_idx
    .cluster_check_idx_to_pcm_check_idx
    .rref.Z, .rref.s_prime, .rref.pivot_map, .rref.n_checks, .rref.n_bits,
    .rref.is_valid()
    .L   (numpy array or None, set after each run())

Lifetime note: ClusterState and IncrementalRREF objects are references into
the C++ engine's internal storage.  They are invalidated by the next call
to run().  Do not hold onto them across run() calls.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Add the directory that contains _clustering_cpp.so to sys.path.
_cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "prior_gap_estimation_cpp")
if _cpp_dir not in sys.path:
    sys.path.insert(0, _cpp_dir)

from _clustering_cpp import ClusteringEngine  # noqa: E402  (after sys.path tweak)

from parity_matrix_construct import ParityCheckMatrices  # noqa: E402


class Clustering:
    """
    C++ Clustering engine with the same interface as clustering.py::Clustering.

    Construct once from a PCM; call run(syndrome) for each shot.
    """

    def __init__(self, pcm: ParityCheckMatrices) -> None:
        self.pcm     = pcm
        self.n_det   = int(pcm.H.shape[0])
        self.n_fault = int(pcm.H.shape[1])

        check_to_faults, fault_to_checks, weights = self._build_adjacency()

        self._engine = ClusteringEngine(
            self.n_det, self.n_fault,
            check_to_faults, fault_to_checks,
            weights.tolist(),
        )

        # Mirrors Python Clustering: expose adjacency for generators / tests.
        self.check_to_faults = check_to_faults
        self.fault_to_checks = fault_to_checks
        self.weights         = weights

        self.active_valid_clusters: dict = {}

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def run(self, syndrome: np.ndarray) -> None:
        """Run the full clustering algorithm for the given syndrome."""
        self._engine.run(np.asarray(syndrome, dtype=np.uint8))

        # Populate active_valid_clusters and set .L on each cluster.
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
        """All currently active ClusterState objects (active and valid)."""
        return self._engine.active_clusters

    @property
    def clusters(self):
        """Alias: same as active_clusters (all active cluster states)."""
        return self._engine.active_clusters

    def get_active_valid_clusters(self) -> None:
        """Re-populate active_valid_clusters from the engine (idempotent)."""
        avc = self._engine.active_valid_clusters
        has_L = hasattr(self.pcm, 'L') and self.pcm.L is not None
        for cl in avc.values():
            cl.L = (
                self.pcm.L[:, cl.cluster_fault_idx_to_pcm_fault_idx]
                if has_L else None
            )
        self.active_valid_clusters = avc

    def create_degenerate_cycle_regions(self) -> dict[int, dict[str, list]]:
        """
        Classify each null-space basis vector in every active valid cluster as
        a logical error or a stabilizer.  Identical logic to clustering.py.
        """
        result: dict[int, dict[str, list]] = {}

        for cl_id, cl in self.active_valid_clusters.items():
            logical_errors: list = []
            stabilizers:    list = []

            for z in cl.rref.Z:
                if cl.L is None:
                    stabilizers.append([z])
                    continue

                flip_vec = cl.L @ z % 2
                if not np.any(flip_vec):
                    stabilizers.append([z])
                else:
                    flip_int = int(sum(int(flip_vec[k]) << k
                                       for k in range(len(flip_vec))))
                    logical_errors.append([z, flip_int])

            result[cl_id] = {
                'logical_error': logical_errors,
                'stabilizer':    stabilizers,
            }

        return result

    # -----------------------------------------------------------------------
    # Build adjacency lists from PCM  (identical to clustering.py)
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
