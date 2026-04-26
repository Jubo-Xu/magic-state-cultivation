from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import stim


# Time coordinate index in stim detector coordinate tuples [x, y, t, ...]
_T_COORD_INDEX = 2


class ParityCheckMatrices:
    """
    Parity-check matrices and associated metadata extracted from a stim DEM.

    Attributes
    ----------
    H : np.ndarray, shape (n_detectors, n_faults), dtype uint8
        Parity-check matrix.  H[i, j] = 1 if fault j triggers detector i.

    L : np.ndarray, shape (n_observables, n_faults), dtype uint8
        Observable (logical) flip matrix.  L[k, j] = 1 if fault j flips
        observable k.

    error_data : list[dict]
        One entry per fault column j (same ordering as columns of H and L).
        Each dict contains:
            'prob'        : float  — physical probability of this fault
            'is_boundary' : bool   — True if fault touches 0 or 1 detectors
                                     (half-edge or observable-only fault)
            'is_timelike' : bool   — True if fault connects detectors from
                                     different rounds (t-coordinates differ);
                                     always False for boundary faults
            'detectors'   : frozenset[int]  — detector IDs this fault touches

    stats : dict
        Summary statistics collected during construction:
            'n_detectors'      : int
            'n_observables'    : int
            'n_faults'         : int
            'decomposed'       : bool  — whether ^ splitting was applied
            'size_distribution': Counter  — {hyperedge_size: count}
            'n_boundary'       : int
            'n_spacelike'      : int
            'n_timelike'       : int
            'prob_min'         : float
            'prob_max'         : float
            'detectors_per_round': dict  — {t: n_detectors_at_that_round}
    """

    def __init__(
        self,
        H: np.ndarray,
        L: np.ndarray,
        error_data: list[dict],
        stats: dict,
        n_logical_check_nodes: int = 0,
    ) -> None:
        self.H = H
        self.L = L
        self.error_data = error_data
        self.stats = stats
        self.n_logical_check_nodes = n_logical_check_nodes

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def from_DEM(
        dem: stim.DetectorErrorModel,
        decompose: bool = False,
    ) -> "ParityCheckMatrices":
        """
        Construct a ParityCheckMatrices from a stim DetectorErrorModel.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            Must already be flattened (no repeat blocks) — call
            dem.detector_error_model(flatten_loops=True) before passing in,
            or use stim's flatten_loops=True option when extracting the DEM.

        decompose : bool, default False
            If False: each error instruction becomes one fault column
            (true hyperedge, size up to ~10 for cultivation circuits).

            If True: error instructions that contain ^ separator targets are
            split into their decomposed MWPM-compatible pieces, and each piece
            becomes its own fault column (most pieces have <= 2 detectors).
            Instructions that stim could not decompose are kept as-is
            (ignore_decomposition_failures must have been set when extracting
            the DEM, otherwise stim already raised an error before this point).

        Returns
        -------
        ParityCheckMatrices
        """
        n_detectors   = dem.num_detectors
        n_observables = dem.num_observables

        # Detector spacetime coordinates: {det_id: [x, y, t, ...]}
        coords = dem.get_detector_coordinates()

        # ------------------------------------------------------------------
        # Step 1: collect detector round distribution (for stats)
        # ------------------------------------------------------------------
        detectors_per_round: dict[float, int] = defaultdict(int)
        for det_id, c in coords.items():
            if len(c) > _T_COORD_INDEX:
                detectors_per_round[c[_T_COORD_INDEX]] += 1

        # ------------------------------------------------------------------
        # Step 2: parse error instructions into (prob, targets) pieces.
        #
        # decompose=False: one piece per instruction (ignore ^ if present).
        # decompose=True:  split at ^ separators; one piece per sub-edge.
        # ------------------------------------------------------------------
        all_pieces: list[tuple[float, list[stim.DemTarget]]] = []

        for inst in dem:
            if inst.type != "error":
                continue

            p = inst.args_copy()[0]

            if not decompose:
                # Treat whole instruction as a single piece; skip any ^
                targets = [t for t in inst.targets_copy() if not t.is_separator()]
                all_pieces.append((p, targets))
            else:
                # Split at ^ into separate pieces
                pieces: list[list[stim.DemTarget]] = []
                current: list[stim.DemTarget] = []
                for t in inst.targets_copy():
                    if t.is_separator():
                        pieces.append(current)
                        current = []
                    else:
                        current.append(t)
                pieces.append(current)
                for piece in pieces:
                    all_pieces.append((p, piece))

        n_faults = len(all_pieces)

        # ------------------------------------------------------------------
        # Step 3: build H, L, error_data
        # ------------------------------------------------------------------
        H          = np.zeros((n_detectors,  n_faults), dtype=np.uint8)
        L          = np.zeros((n_observables, n_faults), dtype=np.uint8)
        error_data: list[dict] = []

        for j, (p, targets) in enumerate(all_pieces):
            det_set: set[int] = set()
            for t in targets:
                if t.is_relative_detector_id():
                    H[t.val, j] = 1
                    det_set.add(t.val)
                elif t.is_logical_observable_id():
                    L[t.val, j] = 1

            detectors = frozenset(det_set)

            # Boundary: touches 0 or 1 detectors — timelike undefined
            if len(detectors) <= 1:
                is_boundary = True
                is_timelike = False
            else:
                is_boundary = False
                # Collect distinct t-coordinates of all detectors in this fault
                t_vals: set[float] = set()
                for det_id in detectors:
                    c = coords.get(det_id, [])
                    if len(c) > _T_COORD_INDEX:
                        t_vals.add(c[_T_COORD_INDEX])
                is_timelike = len(t_vals) > 1

            error_data.append({
                "prob":        p,
                "is_boundary": is_boundary,
                "is_timelike": is_timelike,
                "detectors":   detectors,
            })

        # ------------------------------------------------------------------
        # Step 4: collect stats
        # ------------------------------------------------------------------
        size_distribution: Counter = Counter(len(d["detectors"]) for d in error_data)
        n_boundary  = sum(1 for d in error_data if d["is_boundary"])
        n_timelike  = sum(1 for d in error_data if not d["is_boundary"] and d["is_timelike"])
        n_spacelike = sum(1 for d in error_data if not d["is_boundary"] and not d["is_timelike"])
        probs_arr   = np.array([d["prob"] for d in error_data], dtype=np.float64)

        stats = {
            "n_detectors":       n_detectors,
            "n_observables":     n_observables,
            "n_faults":          n_faults,
            "decomposed":        decompose,
            "size_distribution": size_distribution,
            "n_boundary":        n_boundary,
            "n_spacelike":       n_spacelike,
            "n_timelike":        n_timelike,
            "prob_min":          float(probs_arr.min()) if n_faults > 0 else 0.0,
            "prob_max":          float(probs_arr.max()) if n_faults > 0 else 0.0,
            "detectors_per_round": dict(sorted(detectors_per_round.items())),
        }

        # Logical check-node detectors (added by _dem_with_obs_detector /
        # _dem_with_logical_check_nodes) are identified by a t-coordinate of -10.
        # Physical detectors always have non-negative t-coordinates.
        n_logical_check_nodes = sum(
            1 for c in coords.values()
            if len(c) > _T_COORD_INDEX and c[_T_COORD_INDEX] == -10
        )

        return ParityCheckMatrices(H=H, L=L, error_data=error_data, stats=stats,
                                   n_logical_check_nodes=n_logical_check_nodes)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def print_H(self, max_cols: int = 60, max_col_trunc: bool = True) -> None:
        """
        Print H as a 2-D grid of 0s and 1s.

        Rows are detectors (0..n_detectors-1), columns are faults.

        Parameters
        ----------
        max_cols : int, default 60
            Maximum number of fault columns to display. Ignored when
            max_col_trunc=False.
        max_col_trunc : bool, default True
            If True, truncate output to max_cols columns and print a notice.
            If False, print all columns regardless of max_cols.
        """
        n_det, n_faults = self.H.shape
        cols = min(n_faults, max_cols) if max_col_trunc else n_faults
        truncated = max_col_trunc and n_faults > max_cols

        print(f"H  ({n_det} detectors x {n_faults} faults)"
              + (f"  [showing first {cols} columns]" if truncated else ""))
        for i in range(n_det):
            row = "".join(str(self.H[i, j]) for j in range(cols))
            print(f"  D{i:>4d} | {row}")
        if truncated:
            print(f"  ... ({n_faults - cols} more columns not shown)")

    def print_L(self, max_cols: int = 60, max_col_trunc: bool = True) -> None:
        """
        Print L as a 2-D grid of 0s and 1s.

        Rows are observables (0..n_observables-1), columns are faults.

        Parameters
        ----------
        max_cols : int, default 60
            Maximum number of fault columns to display. Ignored when
            max_col_trunc=False.
        max_col_trunc : bool, default True
            If True, truncate output to max_cols columns and print a notice.
            If False, print all columns regardless of max_cols.
        """
        n_obs, n_faults = self.L.shape
        cols = min(n_faults, max_cols) if max_col_trunc else n_faults
        truncated = max_col_trunc and n_faults > max_cols

        print(f"L  ({n_obs} observables x {n_faults} faults)"
              + (f"  [showing first {cols} columns]" if truncated else ""))
        for k in range(n_obs):
            row = "".join(str(self.L[k, j]) for j in range(cols))
            print(f"  L{k:>4d} | {row}")
        if truncated:
            print(f"  ... ({n_faults - cols} more columns not shown)")

    def print_stats(self) -> None:
        """Print the summary statistics collected during from_DEM."""
        s = self.stats
        print(f"decomposed           : {s['decomposed']}")
        print(f"n_detectors          : {s['n_detectors']}")
        print(f"n_observables        : {s['n_observables']}")
        print(f"n_faults             : {s['n_faults']}")
        print(f"H shape              : {self.H.shape}")
        print(f"L shape              : {self.L.shape}")
        print(f"prob range           : [{s['prob_min']:.3e}, {s['prob_max']:.3e}]")
        print()
        print("Hyperedge size distribution (detectors per fault column):")
        for sz in sorted(s["size_distribution"]):
            print(f"  |edge|={sz:2d}  ->  {s['size_distribution'][sz]} faults")
        print()
        print(f"Boundary faults (<=1 detector) : {s['n_boundary']}")
        print(f"Spacelike faults (same t)       : {s['n_spacelike']}")
        print(f"Timelike faults (diff t)        : {s['n_timelike']}")
        print()
        print("Detectors per round (t coordinate):")
        for t, count in s["detectors_per_round"].items():
            print(f"  t={t:.1f}  ->  {count} detectors")
