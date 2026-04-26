"""
clustering_overgrow_batch.py  —  Over-grow clustering with batch fault growth.

Combines two orthogonal extensions to the base Clustering:

  Over-grow (from clustering_overgrow.py)
    After a cluster first reaches valid_neutral (syndrome in image(H_cluster))
    it continues growing for `over_grow_step` additional _grow_one_step calls
    before being declared truly valid.

  Batch (new)
    Up to `bits_per_step` fault nodes are popped and processed per growth
    step.  Free faults are added in a single add_columns call; collision
    faults are all passed together to a batch merge.  Validity is checked
    exactly once after ALL faults in the step are processed.

Set over_grow_step=0 and bits_per_step=1 to recover the original stopping
behaviour exactly (field-identical results for T, U, H, s, s_prime).
"""

from __future__ import annotations

import heapq
from typing import Optional

import numpy as np

from clustering_overgrow import Clustering as _OriginalClusteringOvergrow
from clustering_overgrow import ClusterStateOvergrow
from clustering import ClusterState
from incremental_rref_batch import IncrementalRREFBatch


class ClusteringOvergrowBatch(_OriginalClusteringOvergrow):
    """
    Drop-in replacement for ClusteringOvergrow with batch fault growth.

    Usage
    -----
    clustering = ClusteringOvergrowBatch(pcm)
    clustering.run(syndrome, over_grow_step=5, bits_per_step=4)
    """

    def __init__(self, pcm) -> None:
        super().__init__(pcm)
        self.bits_per_step: int = 1  # set at the start of each run()

    def run(
        self,
        syndrome: np.ndarray,
        over_grow_step: int = 0,
        bits_per_step: int = 1,
    ) -> None:
        """
        Run clustering with over-grow and batch fault growth.

        Parameters
        ----------
        syndrome : np.ndarray of uint8
        over_grow_step : int
            Extra _grow_one_step calls after first reaching valid_neutral.
            0 = identical to original behaviour.
        bits_per_step : int
            Fault nodes to pop and process per growth step.
            1 = identical to original behaviour.
        """
        if bits_per_step < 1:
            raise ValueError(f"bits_per_step must be >= 1, got {bits_per_step}")
        self.bits_per_step = bits_per_step
        super().run(syndrome, over_grow_step=over_grow_step)

    def _init_each_cluster(self, seed: int) -> ClusterStateOvergrow:
        """Same as parent but uses IncrementalRREFBatch for the cluster RREF."""
        cl = super()._init_each_cluster(seed)
        cl.rref = IncrementalRREFBatch()
        return cl

    # -----------------------------------------------------------------------
    # Batch growth step (replaces both clustering._grow_one_step and the
    # overgrow wrapper in clustering_overgrow._grow_one_step)
    # -----------------------------------------------------------------------

    def _grow_one_step(self, cl: ClusterStateOvergrow) -> ClusterStateOvergrow:
        """
        Pop up to `bits_per_step` faults, batch-add free ones, batch-merge
        collision ones, then apply the over-grow countdown.

        Validity is checked once after all faults are processed — this is
        the semantic difference from bits_per_step=1 step-by-step checking.
        """
        fault_to_checks         = self.fault_to_checks
        global_check_membership = self.global_check_membership
        global_fault_membership = self.global_fault_membership

        free_faults:      list[tuple[float, int]] = []
        collision_faults: list[tuple[float, int]] = []
        merge_set:        set[ClusterState]       = set()

        actions_taken = 0
        while cl.heap and actions_taken < self.bits_per_step:
            vw, j = heapq.heappop(cl.heap)

            # Skip stale heap entries (lazy deletion).
            if vw > cl.dist.get(j, float('inf')):
                continue
            # Skip faults already absorbed into this cluster.
            if global_fault_membership[j] is cl:
                continue

            colliding: set[ClusterState] = set()
            for c in fault_to_checks[j]:
                owner = global_check_membership[c]
                if owner is not None and owner is not cl:
                    colliding.add(owner)

            if colliding:
                collision_faults.append((vw, j))
                merge_set.update(colliding)
            else:
                free_faults.append((vw, j))

            actions_taken += 1

        # Merge first so free faults are added to the (potentially larger)
        # surviving cluster.
        if merge_set:
            connecting_js  = [j  for _,  j in collision_faults]
            connecting_vws = [vw for vw, _ in collision_faults]
            cl = self._merge_batch(cl, merge_set, connecting_js, connecting_vws)

        if free_faults:
            self._add_fault_batch(cl, free_faults)

        # Over-grow countdown (identical logic to clustering_overgrow.py).
        is_neutral = cl.rref.is_valid()
        if not is_neutral:
            cl.overgrow_budget = -1
        elif cl.overgrow_budget < 0:
            cl.overgrow_budget = self.over_grow_step
        elif cl.overgrow_budget > 0:
            cl.overgrow_budget -= 1

        cl.valid = (cl.overgrow_budget == 0)
        return cl

    # -----------------------------------------------------------------------
    # Batch free-fault addition
    # -----------------------------------------------------------------------

    def _add_fault_batch(
        self,
        cl: ClusterStateOvergrow,
        faults: list[tuple[float, int]],
    ) -> None:
        """
        Absorb multiple free faults into cl in a single add_columns call.

        Builds a unified check-index map covering all new RREF rows across
        all faults, constructs their column vectors, and calls add_columns
        once.  A single-element list produces a result field-identical to
        _add_fault (add_columns([h], s_extra) == add_column(h, s_extra)).
        """
        if not faults:
            return

        syndrome                = self.syndrome
        fault_to_checks         = self.fault_to_checks
        check_to_faults         = self.check_to_faults
        weights                 = self.weights
        global_check_membership = self.global_check_membership
        global_fault_membership = self.global_fault_membership

        # Classify each fault's checks (regular RREF rows vs logical bridge).
        regular_per_fault: list[list[int]] = []
        logical_per_fault: list[list[int]] = []
        for _, j in faults:
            regular_j: list[int] = []
            logical_j: list[int] = []
            for c in fault_to_checks[j]:
                if c in self.logical_check_node_set:
                    logical_j.append(c)
                else:
                    regular_j.append(c)
            regular_per_fault.append(regular_j)
            logical_per_fault.append(logical_j)

        # Union of new RREF check rows across all faults (discovery order).
        all_new_for_rref: list[int] = []
        seen_new: set[int] = set()
        for idx in range(len(faults)):
            for c in regular_per_fault[idx]:
                if c not in cl.pcm_check_idx_to_cluster_check_idx and c not in seen_new:
                    all_new_for_rref.append(c)
                    seen_new.add(c)

        # Assign local RREF indices and update check ownership.
        n_existing = cl.rref.n_checks
        for local_offset, c in enumerate(all_new_for_rref):
            local = n_existing + local_offset
            cl.pcm_check_idx_to_cluster_check_idx[c] = local
            cl.cluster_check_idx_to_pcm_check_idx.append(c)
            if c not in cl.check_nodes:
                cl.check_nodes.add(c)
                cl.boundary_check_nodes.add(c)
                global_check_membership[c] = cl

        # Claim logical check node ownership.
        for idx in range(len(faults)):
            for c in logical_per_fault[idx]:
                if c not in cl.check_nodes:
                    cl.check_nodes.add(c)
                    global_check_membership[c] = cl

        # Build column vectors against the unified check-index map.
        n_full = n_existing + len(all_new_for_rref)
        columns: list[np.ndarray] = []
        for idx in range(len(faults)):
            h_j = np.zeros(n_full, dtype=np.uint8)
            for c in regular_per_fault[idx]:
                h_j[cl.pcm_check_idx_to_cluster_check_idx[c]] = 1
            columns.append(h_j)

        s_extra = (
            np.array([syndrome[c] for c in all_new_for_rref], dtype=np.uint8)
            if all_new_for_rref else None
        )

        cl.rref.add_columns(columns, s_extra)

        # Register faults and push Dijkstra candidates.
        for vw, j in faults:
            cl.fault_nodes.add(j)
            cl.cluster_fault_idx_to_pcm_fault_idx.append(j)
            global_fault_membership[j] = cl

        for vw, j in faults:
            for c in fault_to_checks[j]:
                for k in check_to_faults[c]:
                    if global_fault_membership[k] is not cl:
                        new_vw = vw + float(weights[k])
                        if new_vw < cl.dist.get(k, float('inf')):
                            cl.dist[k] = new_vw
                            heapq.heappush(cl.heap, (new_vw, k))

    # -----------------------------------------------------------------------
    # Batch merge (single or multiple connecting faults)
    # -----------------------------------------------------------------------

    def _merge_batch(
        self,
        cl: ClusterStateOvergrow,
        others: set[ClusterState],
        connecting_js: list[int],
        connecting_vws: list[float],
    ) -> ClusterState:
        """
        Merge cl and all clusters in `others` using multiple connecting faults.

        Identical to the parent _merge for a single connecting fault, but
        adds ALL connecting faults via one add_columns call rather than
        sequential add_column calls.

        Pre-condition (maintained by _grow_one_step):
        - cl and every cluster in `others` have pairwise disjoint check sets.
        - Each fault in connecting_js is not yet in any cluster's RREF.
        """
        syndrome                = self.syndrome
        fault_to_checks         = self.fault_to_checks
        check_to_faults         = self.check_to_faults
        weights                 = self.weights
        global_check_membership = self.global_check_membership
        global_fault_membership = self.global_fault_membership

        active_others = [o for o in others if o.active]

        # Determine surviving cluster (largest by fault count).
        all_involved = [cl] + active_others
        all_involved.sort(key=lambda c: len(c.fault_nodes), reverse=True)
        larger  = all_involved[0]
        smaller = all_involved[1:]

        # Step 1: block-diagonal RREF merge of all involved clusters.
        unified_check_list: list[int]      = list(larger.cluster_check_idx_to_pcm_check_idx)
        unified_check_map:  dict[int, int] = dict(larger.pcm_check_idx_to_cluster_check_idx)
        offset = larger.rref.n_checks
        merged_rref = larger.rref

        for other in smaller:
            for c, local in other.pcm_check_idx_to_cluster_check_idx.items():
                unified_check_map[c] = offset + local
            unified_check_list.extend(other.cluster_check_idx_to_pcm_check_idx)
            merged_rref = IncrementalRREFBatch.merge(merged_rref, other.rref, connecting_edges=[])
            offset += other.rref.n_checks

        # Step 2: add ALL connecting faults via a single add_columns call.
        # Classify each connecting fault's checks.
        regular_per_cf: list[list[int]] = []
        logical_per_cf: list[list[int]] = []
        for j in connecting_js:
            regular_j: list[int] = []
            logical_j: list[int] = []
            for c in fault_to_checks[j]:
                if c in self.logical_check_node_set:
                    logical_j.append(c)
                else:
                    regular_j.append(c)
            regular_per_cf.append(regular_j)
            logical_per_cf.append(logical_j)

        # Union of new check rows across all connecting faults (discovery order).
        all_new_checks: list[int] = []
        seen_new: set[int] = set()
        for cf_idx in range(len(connecting_js)):
            for c in regular_per_cf[cf_idx]:
                if c not in unified_check_map and c not in seen_new:
                    all_new_checks.append(c)
                    seen_new.add(c)

        for local_offset, c in enumerate(all_new_checks):
            unified_check_map[c] = offset + local_offset
            unified_check_list.append(c)

        # Build column vectors against the unified check-index map.
        n_total = offset + len(all_new_checks)
        columns: list[np.ndarray] = []
        for cf_idx in range(len(connecting_js)):
            h_j = np.zeros(n_total, dtype=np.uint8)
            for c in regular_per_cf[cf_idx]:
                h_j[unified_check_map[c]] = 1
            columns.append(h_j)

        s_extra = (
            np.array([syndrome[c] for c in all_new_checks], dtype=np.uint8)
            if all_new_checks else None
        )

        # Claim logical check ownership for all connecting faults.
        for cf_idx in range(len(connecting_js)):
            for c in logical_per_cf[cf_idx]:
                larger.check_nodes.add(c)
                global_check_membership[c] = larger

        merged_rref.add_columns(columns, s_extra)

        # Step 3: update surviving cluster's state.
        larger.rref = merged_rref
        larger.cluster_check_idx_to_pcm_check_idx = unified_check_list
        larger.pcm_check_idx_to_cluster_check_idx = unified_check_map

        # Absorb smaller clusters into larger.
        for other in smaller:
            for j2 in other.cluster_fault_idx_to_pcm_fault_idx:
                larger.fault_nodes.add(j2)
                larger.cluster_fault_idx_to_pcm_fault_idx.append(j2)
                global_fault_membership[j2] = larger
            for c in other.check_nodes:
                larger.check_nodes.add(c)
                global_check_membership[c] = larger
            for c in other.boundary_check_nodes:
                larger.boundary_check_nodes.add(c)
            for s in other.enclosed_syndromes:
                larger.enclosed_syndromes.add(s)
            for entry in other.heap:
                heapq.heappush(larger.heap, entry)
            for k, d in other.dist.items():
                if d < larger.dist.get(k, float('inf')):
                    larger.dist[k] = d
            other.active = False

        # Register all connecting faults in the surviving cluster.
        for j in connecting_js:
            larger.fault_nodes.add(j)
            larger.cluster_fault_idx_to_pcm_fault_idx.append(j)
            global_fault_membership[j] = larger

        # Update check ownership for any new checks from connecting faults.
        for c in all_new_checks:
            larger.check_nodes.add(c)
            larger.boundary_check_nodes.add(c)
            global_check_membership[c] = larger

        # Push Dijkstra candidates reachable through all connecting faults.
        for vw, j in zip(connecting_vws, connecting_js):
            for c in fault_to_checks[j]:
                for k in check_to_faults[c]:
                    if global_fault_membership[k] is not larger:
                        new_vw = vw + float(weights[k])
                        if new_vw < larger.dist.get(k, float('inf')):
                            larger.dist[k] = new_vw
                            heapq.heappush(larger.heap, (new_vw, k))

        return larger
