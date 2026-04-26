"""
prior_gap_estimator_cpp.py  —  C++ drop-in for PriorGapEstimatorUse.

Wraps PriorGapEstimatorEngine (pybind11) to provide the same execute()
interface as PriorGapEstimatorUse, plus a new execute_batch() method for
processing many shots in a single C++ call.

Usage
-----
    from prior_gap_estimator_cpp import PriorGapEstimatorUse

    estimator = PriorGapEstimatorUse(pcm)
    gap, nonzero_count = estimator.execute(syndrome, gap_type='binary')
    gap, flip, nonzero_count = estimator.execute(syndrome, gap_type='weight_diff',
                                                  decode=True)

    gaps, nonzero_counts, flips = estimator.execute_batch(
        syndromes,          # uint8 (n_shots, n_dets)
        gap_type='prior_weight',
        decode=True,
    )
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Locate _clustering_cpp.so (built by prior_gap_estimation_cpp/setup.py).
_cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'prior_gap_estimation_cpp')
if _cpp_dir not in sys.path:
    sys.path.insert(0, _cpp_dir)

from _clustering_cpp import PriorGapEstimatorEngine   # noqa: E402


class PriorGapEstimatorUse:
    """
    C++ PriorGapEstimator with the same execute() interface as
    PriorGapEstimatorUse (prior_gap_estimation_use.py).

    Construct from a ParityCheckMatrices object (same as ClusteringOvergrowBatch).

    Parameters
    ----------
    pcm : ParityCheckMatrices
        Must have .H, .error_data, .n_logical_check_nodes, and optionally .L.
    """

    def __init__(self, pcm) -> None:
        self.pcm     = pcm
        self.n_det   = int(pcm.H.shape[0])
        self.n_fault = int(pcm.H.shape[1])

        check_to_faults, fault_to_checks, weights = self._build_adjacency()

        n_logical: int = getattr(pcm, 'n_logical_check_nodes', 0)
        L_rows: list[list[int]] = []
        if hasattr(pcm, 'L') and pcm.L is not None:
            L_arr = np.asarray(pcm.L, dtype=np.uint8)
            if n_logical == 0:
                n_logical = int(L_arr.shape[0])
            L_rows = [L_arr[i, :].tolist() for i in range(L_arr.shape[0])]

        self._engine = PriorGapEstimatorEngine(
            self.n_det, self.n_fault,
            check_to_faults, fault_to_checks,
            weights.tolist(),
            n_logical,
            L_rows,
        )
        self.n_logical = n_logical

    # -----------------------------------------------------------------------
    # execute  — mirrors PriorGapEstimatorUse.execute exactly
    #
    # decode=False : returns (gap, nonzero_count)
    # decode=True  : returns (gap, overall_logical_flip, nonzero_count)
    #                overall_logical_flip is np.ndarray(n_logical,) uint8
    # -----------------------------------------------------------------------

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
        if gap_type == 'weight_diff' and not decode:
            raise ValueError("gap_type='weight_diff' requires decode=True")

        gap, nonzero_count, flip = self._engine.execute(
            np.asarray(syndrome, dtype=np.uint8),
            gap_type, aggregate,
            over_grow_step, bits_per_step,
            asb, decode,
        )

        if decode:
            # Match Python return order: (gap, overall_logical_flip, nonzero_count)
            return gap, flip, nonzero_count
        return gap, nonzero_count

    # -----------------------------------------------------------------------
    # execute_batch  — new method; no Python equivalent
    #
    # syndromes : uint8 (n_shots, n_dets)
    # Returns   : (gaps, nonzero_counts, flips)
    #   gaps          : float64 (n_shots,)
    #   nonzero_counts: int32   (n_shots,)
    #   flips         : uint8   (n_shots, n_logical), or None when
    #                   decode=False or n_logical=0
    # -----------------------------------------------------------------------

    def execute_batch(
        self,
        syndromes: np.ndarray,
        gap_type: str = 'binary',
        aggregate: str = 'min',
        over_grow_step: int = 0,
        bits_per_step: int = 1,
        asb: bool = False,
        decode: bool = False,
    ):
        if gap_type == 'weight_diff' and not decode:
            raise ValueError("gap_type='weight_diff' requires decode=True")

        return self._engine.execute_batch(
            np.asarray(syndromes, dtype=np.uint8, order='C'),
            gap_type, aggregate,
            over_grow_step, bits_per_step,
            asb, decode,
        )

    # -----------------------------------------------------------------------
    # Build adjacency (identical to clustering_overgrow_batch_cpp.py)
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
