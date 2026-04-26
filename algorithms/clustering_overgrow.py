"""
clustering_overgrow.py  —  Clustering with configurable over-grow.

After a cluster first reaches valid_neutral (syndrome in image(H_cluster)),
it continues growing for `over_grow_step` additional _grow_one_step calls
before being declared truly valid.  This forces the cluster to absorb fault
nodes near the logical boundary, which can turn a zero null-basis into a
non-zero one for ambiguous shots.

Set over_grow_step=0 to recover the original stopping behaviour exactly.
"""

from __future__ import annotations

import heapq
import numpy as np

# Import the original Clustering and ClusterState from clustering.py.
# We alias Clustering to avoid shadowing it with the subclass defined below.
from clustering import Clustering as _OriginalClustering
from clustering import ClusterState


class ClusterStateOvergrow(ClusterState):
    """ClusterState extended with an over-grow countdown."""

    __slots__ = ('overgrow_budget',)

    def __init__(self, cluster_id: int) -> None:
        super().__init__(cluster_id)
        # -1  : valid_neutral not yet reached
        # > 0 : steps still remaining in over-grow phase
        # 0   : over-grow complete; cluster is truly valid
        self.overgrow_budget: int = -1


class Clustering(_OriginalClustering):
    """
    Drop-in replacement for the original Clustering with over-grow support.

    over_grow_step is passed to run() rather than __init__(), so the same
    object can be reused across shots with different step counts.

    Usage
    -----
    clustering = Clustering(pcm)
    clustering.run(syndrome, over_grow_step=5)
    """

    def __init__(self, pcm) -> None:
        super().__init__(pcm)
        self.over_grow_step: int = 0  # set at the start of each run()

    def run(self, syndrome: np.ndarray, over_grow_step: int = 0) -> None:
        """
        Run clustering with over-grow.

        Parameters
        ----------
        syndrome : np.ndarray of uint8
        over_grow_step : int
            Extra _grow_one_step calls after each cluster first reaches
            valid_neutral.  0 = identical to original behaviour.
        """
        self.over_grow_step = over_grow_step
        super().run(syndrome)

    def _init_each_cluster(self, seed: int) -> ClusterStateOvergrow:
        """Same as original but creates ClusterStateOvergrow instead of ClusterState."""
        cl = ClusterStateOvergrow(seed)

        cl.check_nodes.add(seed)
        cl.boundary_check_nodes.add(seed)
        cl.enclosed_syndromes.add(seed)
        self.global_check_membership[seed] = cl

        for j in self.check_to_faults[seed]:
            w = float(self.weights[j])
            if w < cl.dist.get(j, float('inf')):
                cl.dist[j] = w
                heapq.heappush(cl.heap, (w, j))

        return cl

    def _grow_one_step(self, cl: ClusterStateOvergrow) -> ClusterStateOvergrow:
        """Same as original but applies over-grow countdown after valid_neutral."""
        result = super()._grow_one_step(cl)

        # result.valid was set to rref.is_valid() by the parent.
        is_neutral = result.valid

        if not is_neutral:
            # Invalid (including after a merge that breaks a previously-done
            # cluster): reset so we wait for valid_neutral again.
            result.overgrow_budget = -1
        elif result.overgrow_budget < 0:
            # First time reaching valid_neutral: start countdown.
            result.overgrow_budget = self.over_grow_step
        elif result.overgrow_budget > 0:
            # Count down on every step.
            result.overgrow_budget -= 1

        result.valid = (result.overgrow_budget == 0)
        return result
