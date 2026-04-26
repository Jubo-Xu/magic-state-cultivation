"""
clustering.py  —  Tanner-graph clustering for gap estimation.

One cluster is created per non-trivial detector.  Each cluster grows by
Dijkstra virtual weight (accumulated path cost from the seed detector).
When a chosen fault node's checks span two clusters the clusters merge,
with that fault as the single connecting edge in IncrementalRREF.merge.

Growth stops once every cluster is valid: syndrome ∈ image(H_cluster),
checked via IncrementalRREF.is_valid().  After clustering, rref.Z holds
the null-space basis for gap estimation and rref.s / rref.s_prime hold
the (transformed) syndrome restricted to the cluster's check nodes.

Relationship to LSD
-------------------
The clustering logic mirrors LSD (Localized Statistics Decoder) with two
differences:
  - Virtual weights (Dijkstra accumulated) instead of raw BP LLRs.
  - IncrementalRREF instead of PLU decomposition; no error-chain recovery.

Key invariant
-------------
Before every merge, the two clusters have disjoint check sets and disjoint
fault sets.  This holds because grow() checks ALL of a candidate fault's
adjacent checks before adding it: if any check belongs to another cluster
the fault is held back as the connecting edge and the merge is triggered.
Faults already in a cluster therefore only touch that cluster's checks,
making IncrementalRREF.merge's block-diagonal construction exact.
"""

from __future__ import annotations

import heapq
from typing import Optional

import numpy as np

from incremental_rref import IncrementalRREF
from parity_matrix_construct import ParityCheckMatrices


# ---------------------------------------------------------------------------
# Cluster state
# ---------------------------------------------------------------------------

class ClusterState:
    """
    All mutable state for one growing cluster.

    Attributes
    ----------
    cluster_id : int
        Index of the seed non-trivial detector.
    active : bool
        False once this cluster has been absorbed into a larger one.
    valid : bool
        True once rref.is_valid() — syndrome is in image(H_cluster).

    Node sets  (all store global PCM indices)
    ---------
    fault_nodes          : fault nodes absorbed into this cluster.
    check_nodes          : check nodes owned by this cluster.
    boundary_check_nodes : checks whose adjacent faults are not all absorbed.
    enclosed_syndromes   : non-trivial detectors inside this cluster.

    Local index maps
    ----------------
    cluster_check_idx_to_pcm_check_idx : list[int]
        cluster_check_idx_to_pcm_check_idx[local_i] = global_check_i.
        Ordered by the time each check first entered the RREF.
    pcm_check_idx_to_cluster_check_idx : dict[int, int]
        Inverse of the above.
        NOTE: a check can be in check_nodes but absent from this dict —
        it enters the dict (and the RREF) only when the first fault
        touching it is added.  The seed detector is the primary example.
    cluster_fault_idx_to_pcm_fault_idx : list[int]
        cluster_fault_idx_to_pcm_fault_idx[local_j] = global_fault_j.

    Dijkstra state
    --------------
    heap : min-heap of (virtual_weight, fault_idx).
    dist : dict mapping fault_idx → best virtual weight seen so far.
           Used to skip stale heap entries (lazy deletion).

    RREF
    ----
    rref : IncrementalRREF tracking H_cluster, s_cluster, null space Z.
    """

    __slots__ = (
        'cluster_id', 'active', 'valid',
        'fault_nodes', 'check_nodes', 'boundary_check_nodes', 'enclosed_syndromes',
        'cluster_check_idx_to_pcm_check_idx', 'pcm_check_idx_to_cluster_check_idx',
        'cluster_fault_idx_to_pcm_fault_idx',
        'heap', 'dist',
        'rref',
        'L',
    )

    def __init__(self, cluster_id: int) -> None:
        self.cluster_id = cluster_id
        self.active: bool = True
        self.valid:  bool = False
        self.L: Optional[np.ndarray] = None  # sub-logical matrix, set by get_active_valid_clusters()

        self.fault_nodes:          set[int] = set()
        self.check_nodes:          set[int] = set()
        self.boundary_check_nodes: set[int] = set()
        self.enclosed_syndromes:   set[int] = set()

        self.cluster_check_idx_to_pcm_check_idx: list[int]      = []
        self.pcm_check_idx_to_cluster_check_idx: dict[int, int] = {}
        self.cluster_fault_idx_to_pcm_fault_idx:     list[int]      = []

        self.heap: list[tuple[float, int]] = []
        self.dist: dict[int, float]        = {}

        self.rref: IncrementalRREF = IncrementalRREF()


# ---------------------------------------------------------------------------
# Clustering process
# ---------------------------------------------------------------------------

class Clustering:
    """
    Tanner-graph clustering for gap estimation.

    Construct with a PCM (computed once from the DEM), then call run(syndrome)
    for each shot.

    Parameters
    ----------
    pcm : ParityCheckMatrices
        Parity-check matrices and fault metadata.  Built once and reused
        across many syndromes.

    After calling run(syndrome), the results are accessible via:
      clusters              — all ClusterState objects (active and inactive)
      active_clusters       — property returning only active ones
      active_valid_clusters — dict[cluster_id, ClusterState] for active valid clusters,
                              each with cl.L set to its sub-logical matrix (or None).
    Each active cluster's rref.Z gives the null-space basis for gap estimation.
    """

    def __init__(self, pcm: ParityCheckMatrices) -> None:
        self.pcm     = pcm
        self.n_det   = pcm.H.shape[0]
        self.n_fault = pcm.H.shape[1]

        n_logical = getattr(pcm, 'n_logical_check_nodes', 0)
        self.logical_check_node_set: frozenset[int] = (
            frozenset(range(self.n_det - n_logical, self.n_det))
            if n_logical > 0 else frozenset()
        )

        self.check_to_faults, self.fault_to_checks, self.weights = \
            self._build_adjacency()

        # These are reset per syndrome in clusters_initialization.
        self.syndrome: np.ndarray = np.zeros(0, dtype=np.uint8)
        self.global_check_membership: list[Optional[ClusterState]] = [None] * self.n_det
        self.global_fault_membership: list[Optional[ClusterState]] = [None] * self.n_fault
        self.clusters: list[ClusterState] = []

        # Populated by get_active_valid_clusters() at the end of run().
        # Maps cluster_id → ClusterState (with cl.L set to its sub-logical matrix).
        self.active_valid_clusters: dict[int, ClusterState] = {}

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def clusters_initialization(self, syndrome: np.ndarray) -> None:
        """
        Reset all clustering state and initialise one cluster per non-trivial
        detector in `syndrome`.  Called at the start of run().

        Parameters
        ----------
        syndrome : np.ndarray of uint8, shape (n_detectors,)
            Binary syndrome: 1 at each non-trivial detector.
        """
        self.syndrome = syndrome
        self.global_check_membership = [None] * self.n_det
        self.global_fault_membership = [None] * self.n_fault
        self.clusters = []

        for i in range(self.n_det):
            if syndrome[i] and i not in self.logical_check_node_set:
                cl = self._init_each_cluster(i)
                self.clusters.append(cl)

    def run(self, syndrome: np.ndarray) -> None:
        """Run the full clustering algorithm for the given syndrome."""
        self.active_valid_clusters = {}
        self.clusters_initialization(syndrome)

        # Grow until all clusters are valid.
        # Rebuild invalid list from all clusters after each round to pick up
        # surviving clusters that were not in the previous invalid list
        # (e.g. after a merge where the surviving cluster was already valid
        # but the merged result is not).
        invalid: list[ClusterState] = [cl for cl in self.clusters if not cl.valid]

        while invalid:
            for cl in invalid:
                if cl.active and not cl.valid:
                    self._grow_one_step(cl)
            invalid = [cl for cl in self.clusters if cl.active and not cl.valid]

        self.get_active_valid_clusters()

    def get_active_valid_clusters(self) -> None:
        """
        Populate self.active_valid_clusters for every active valid cluster, and
        set each cluster's .L attribute to its sub-logical matrix.

        For each active valid cluster, L_C is the restriction of pcm.L to the
        fault columns belonging to the cluster, in the same column order as the
        cluster's RREF (i.e. cluster_fault_idx_to_pcm_fault_idx).  Component j
        of any z in cl.rref.Z corresponds to column j of cl.L.

        If pcm has no L attribute (e.g. unit-test stubs), cl.L is left as None.

        Result
        ------
        self.active_valid_clusters : dict[int, ClusterState]
            cluster_id  →  ClusterState, with cl.L set.
        """
        self.active_valid_clusters = {}
        has_L = hasattr(self.pcm, 'L') and self.pcm.L is not None
        for cl in self.clusters:
            if cl.active and cl.valid:
                cl.L = (
                    self.pcm.L[:, cl.cluster_fault_idx_to_pcm_fault_idx]
                    if has_L else None
                )
                self.active_valid_clusters[cl.cluster_id] = cl

    def create_degenerate_cycle_regions(self) -> dict[int, dict[str, list]]:
        """
        Classify each null-space basis vector in every active valid cluster as
        either a logical error or a stabilizer by checking the sub-logical matrix.

        For each basis vector z in cl.rref.Z:
          - Compute flip_vec = (cl.L @ z) % 2.
          - If flip_vec is all-zero  → stabilizer  (no logical observable is flipped).
          - Otherwise               → logical error (one or more observables flipped).

        For logical error vectors, the flip pattern is encoded as an integer
        (basis_logical_flip_int) where bit k corresponds to logical observable k
        (least-significant bit = observable 0).

        If cl.L is None (pcm has no L, e.g. unit-test stubs), all basis vectors
        are placed in the stabilizer group.

        Parameters
        ----------
        (none — operates on self.active_valid_clusters populated by run())

        Returns
        -------
        dict[int, dict[str, list]]
            cluster_id →
                {
                  "logical_error": [[z0, flip_int0], [z1, flip_int1], ...],
                  "stabilizer":    [[z0], [z1], ...],
                }
            where z is a np.ndarray (uint8, 1-D) and flip_int is an int.
        """
        result: dict[int, dict[str, list]] = {}

        for cl_id, cl in self.active_valid_clusters.items():
            logical_errors: list = []
            stabilizers:    list = []

            for z in cl.rref.Z:
                if cl.L is None:
                    stabilizers.append([z])
                    continue

                flip_vec = cl.L @ z % 2   # shape (n_obs,), uint8
                if not np.any(flip_vec):
                    stabilizers.append([z])
                else:
                    # Pack the flip pattern into an integer: bit k = observable k.
                    flip_int = int(sum(int(flip_vec[k]) << k for k in range(len(flip_vec))))
                    logical_errors.append([z, flip_int])

            result[cl_id] = {
                'logical_error': logical_errors,
                'stabilizer':    stabilizers,
            }

        return result

    def run_and_create_degenerate_cycle_regions(
        self, syndrome: np.ndarray, **kwargs
    ) -> dict[int, dict[str, list]]:
        """
        Convenience: run(syndrome, **kwargs) then create_degenerate_cycle_regions().

        All keyword arguments are forwarded to run(), so over_grow_step and
        bits_per_step work transparently for subclasses (ClusteringOvergrow,
        ClusteringOvergrowBatch) without needing to override this method.

        Returns
        -------
        Same dict as create_degenerate_cycle_regions().
        """
        self.run(syndrome, **kwargs)
        return self.create_degenerate_cycle_regions()

    @property
    def active_clusters(self) -> list[ClusterState]:
        return [cl for cl in self.clusters if cl.active]

    # -----------------------------------------------------------------------
    # Build adjacency lists from PCM
    # -----------------------------------------------------------------------

    def _build_adjacency(
        self,
    ) -> tuple[list[list[int]], list[list[int]], np.ndarray]:
        """
        Return (check_to_faults, fault_to_checks, weights).

        check_to_faults[i] = list of fault indices touching detector i.
        fault_to_checks[j] = list of detector indices touched by fault j.
        weights[j]         = -log(p_j)  (log-likelihood weight, >= 0).
        """
        # pcm = self.pcm
        n_det   = self.pcm.H.shape[0]
        n_fault = self.pcm.H.shape[1]

        check_to_faults: list[list[int]] = [[] for _ in range(n_det)]
        fault_to_checks: list[list[int]] = [[] for _ in range(n_fault)]

        for j, ed in enumerate(self.pcm.error_data):
            for i in ed['detectors']:
                check_to_faults[i].append(j)
                fault_to_checks[j].append(i)

        weights = np.array(
            [np.log((1 - ed['prob']) / ed['prob']) if 0 < ed['prob'] < 1 else float('inf')
             for ed in self.pcm.error_data],
            dtype=np.float64,
        )
        return check_to_faults, fault_to_checks, weights

    # -----------------------------------------------------------------------
    # Initialise one cluster
    # -----------------------------------------------------------------------

    def _init_each_cluster(self, seed: int) -> ClusterState:
        """
        Create a cluster seeded at non-trivial detector `seed`.

        The seed is registered in check_nodes and global_check_membership but
        NOT yet in the RREF (n_checks starts at 0).  It enters the RREF as a
        new row when the first fault touching it is added via _add_fault.
        """
        cl = ClusterState(seed)

        cl.check_nodes.add(seed)
        cl.boundary_check_nodes.add(seed)
        cl.enclosed_syndromes.add(seed)         # seed is non-trivial by definition
        self.global_check_membership[seed] = cl

        # Seed heap: all faults directly adjacent to the seed detector.
        for j in self.check_to_faults[seed]:
            w = float(self.weights[j])
            if w < cl.dist.get(j, float('inf')):
                cl.dist[j] = w
                heapq.heappush(cl.heap, (w, j))

        return cl

    # -----------------------------------------------------------------------
    # Add a free fault to a cluster
    # -----------------------------------------------------------------------

    def _add_fault(self, cl: ClusterState, j: int, vw: float) -> None:
        """
        Absorb fault j (already confirmed free — no collisions) into cl.

        Steps
        -----
        1. Identify checks that are new to the RREF (not yet in the local
           index map).  This includes the seed detector on its first visit
           and any unclaimed checks j introduces.
        2. Assign consecutive local RREF indices starting from rref.n_checks.
        3. Build h_j in local index space and call rref.add_column.
        4. Register fault membership and push new Dijkstra candidates.
        """
        syndrome             = self.syndrome
        fault_to_checks      = self.fault_to_checks
        check_to_faults      = self.check_to_faults
        weights              = self.weights
        global_check_membership = self.global_check_membership
        global_fault_membership = self.global_fault_membership

        # Split fault_to_checks[j] into regular checks and logical bridge nodes.
        # Logical nodes participate in ownership/collision only, never in RREF.
        regular_checks_j: list[int] = []
        logical_checks_j: list[int] = []
        for c in fault_to_checks[j]:
            if c in self.logical_check_node_set:
                logical_checks_j.append(c)
            else:
                regular_checks_j.append(c)

        # Checks that need new RREF rows (not yet in local index map).
        # Note: a check can be in check_nodes (e.g. seed) but absent from
        # pcm_check_idx_to_cluster_check_idx — it is treated as a new row.
        new_for_rref: list[int] = [
            c for c in regular_checks_j
            if c not in cl.pcm_check_idx_to_cluster_check_idx
        ]

        # Assign local indices and update check ownership for truly new checks.
        n_existing = cl.rref.n_checks
        for idx, c in enumerate(new_for_rref):
            local = n_existing + idx
            cl.pcm_check_idx_to_cluster_check_idx[c] = local
            cl.cluster_check_idx_to_pcm_check_idx.append(c)
            if c not in cl.check_nodes:          # unclaimed check — take ownership
                cl.check_nodes.add(c)
                cl.boundary_check_nodes.add(c)
                global_check_membership[c] = cl
                # if syndrome[c]:
                #     cl.enclosed_syndromes.add(c)
            # If c is already in check_nodes (e.g. seed detector) we only
            # needed to add it to the index map, which is done above.

        # Claim ownership of logical check nodes (no RREF rows).
        for c in logical_checks_j:
            if c not in cl.check_nodes:
                cl.check_nodes.add(c)
                global_check_membership[c] = cl

        # Build h_j in local RREF index space (regular checks only).
        h_j = np.zeros(n_existing + len(new_for_rref), dtype=np.uint8)
        for c in regular_checks_j:
            h_j[cl.pcm_check_idx_to_cluster_check_idx[c]] = 1

        # s_extra: syndrome values for the newly introduced RREF rows only.
        s_extra = (
            np.array([syndrome[c] for c in new_for_rref], dtype=np.uint8)
            if new_for_rref else None
        )

        cl.rref.add_column(h_j, s_extra)

        # Register fault.
        cl.fault_nodes.add(j)
        cl.cluster_fault_idx_to_pcm_fault_idx.append(j)
        global_fault_membership[j] = cl

        # Push new Dijkstra candidates from all checks adjacent to j.
        for c in fault_to_checks[j]:
            for k in check_to_faults[c]:
                if global_fault_membership[k] is not cl:
                    new_vw = vw + float(weights[k])
                    if new_vw < cl.dist.get(k, float('inf')):
                        cl.dist[k] = new_vw
                        heapq.heappush(cl.heap, (new_vw, k))

    # -----------------------------------------------------------------------
    # Merge multiple clusters
    # -----------------------------------------------------------------------

    def _merge(
        self,
        cl: ClusterState,
        others: set[ClusterState],
        connecting_j: int,
        connecting_vw: float,
    ) -> ClusterState:
        """
        Merge cl and all clusters in `others` into the largest one, using
        connecting_j as the single connecting fault edge.

        Pre-condition (maintained by grow)
        ------------------------------------
        - cl and every cluster in `others` have pairwise disjoint check sets.
        - connecting_j is not yet in any cluster's RREF.
        - connecting_j's checks span cl and/or clusters in `others` (and
          possibly some unclaimed checks).

        Algorithm
        ---------
        Step 1  Block-diagonal RREF merge of all clusters (valid by the
                disjoint-check-set invariant).  No connecting edges yet.
        Step 2  Add connecting_j via a single add_column call on the merged
                RREF, with any unclaimed checks introduced as new rows.
        Step 3  Update node sets, memberships, and Dijkstra heaps of the
                surviving (largest) cluster.
        """
        syndrome                = self.syndrome
        fault_to_checks         = self.fault_to_checks
        check_to_faults         = self.check_to_faults
        weights                 = self.weights
        global_check_membership = self.global_check_membership
        global_fault_membership = self.global_fault_membership

        active_others = [o for o in others if o.active]

        # --- Determine surviving cluster (largest by fault count) -------------
        all_involved = [cl] + active_others
        all_involved.sort(key=lambda c: len(c.fault_nodes), reverse=True)
        larger  = all_involved[0]
        smaller = all_involved[1:]

        # --- Step 1: block-diagonal RREF merge --------------------------------
        # Build a unified check map spanning all involved clusters.
        unified_check_list: list[int]      = list(larger.cluster_check_idx_to_pcm_check_idx)
        unified_check_map:  dict[int, int] = dict(larger.pcm_check_idx_to_cluster_check_idx)
        offset = larger.rref.n_checks
        merged_rref = larger.rref

        for other in smaller:
            # Extend unified map: other's local indices shift by current offset.
            for c, local in other.pcm_check_idx_to_cluster_check_idx.items():
                unified_check_map[c] = offset + local
            unified_check_list.extend(other.cluster_check_idx_to_pcm_check_idx)

            # Block-diagonal merge — valid because check sets are disjoint.
            merged_rref = IncrementalRREF.merge(merged_rref, other.rref, connecting_edges=[])
            offset += other.rref.n_checks

        # --- Step 2: add connecting_j as the connecting edge ------------------
        # Split connecting_j's checks into regular and logical bridge nodes.
        regular_checks_connecting_j: list[int] = []
        logical_checks_connecting_j: list[int] = []
        for c in fault_to_checks[connecting_j]:
            if c in self.logical_check_node_set:
                logical_checks_connecting_j.append(c)
            else:
                regular_checks_connecting_j.append(c)

        # Any regular checks not yet in the unified map are new RREF rows.
        new_checks_j: list[int] = [
            c for c in regular_checks_connecting_j
            if c not in unified_check_map
        ]
        for idx, c in enumerate(new_checks_j):
            unified_check_map[c] = offset + idx
            unified_check_list.append(c)

        h_j = np.zeros(offset + len(new_checks_j), dtype=np.uint8)
        for c in regular_checks_connecting_j:
            h_j[unified_check_map[c]] = 1

        s_extra_j = (
            np.array([syndrome[c] for c in new_checks_j], dtype=np.uint8)
            if new_checks_j else None
        )
        merged_rref.add_column(h_j, s_extra_j)

        # Claim ownership of logical check nodes for connecting_j.
        for c in logical_checks_connecting_j:
            larger.check_nodes.add(c)
            global_check_membership[c] = larger

        # --- Step 3: update surviving cluster's state -------------------------
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
            # Lazy heap merge: push all entries; stale ones filtered on pop.
            for entry in other.heap:
                heapq.heappush(larger.heap, entry)
            for k, d in other.dist.items():
                if d < larger.dist.get(k, float('inf')):
                    larger.dist[k] = d
            other.active = False

        # Register connecting_j in the surviving cluster.
        larger.fault_nodes.add(connecting_j)
        larger.cluster_fault_idx_to_pcm_fault_idx.append(connecting_j)
        global_fault_membership[connecting_j] = larger

        # Update ownership for new checks from connecting_j.
        for c in new_checks_j:
            larger.check_nodes.add(c)
            larger.boundary_check_nodes.add(c)
            global_check_membership[c] = larger
            # if syndrome[c]:                   # dead code in non-JIT case
            #     larger.enclosed_syndromes.add(c)

        # Push new Dijkstra candidates reachable through connecting_j.
        for c in fault_to_checks[connecting_j]:
            for k in check_to_faults[c]:
                if global_fault_membership[k] is not larger:
                    new_vw = connecting_vw + float(weights[k])
                    if new_vw < larger.dist.get(k, float('inf')):
                        larger.dist[k] = new_vw
                        heapq.heappush(larger.heap, (new_vw, k))

        return larger

    # -----------------------------------------------------------------------
    # Single growth step
    # -----------------------------------------------------------------------

    def _grow_one_step(self, cl: ClusterState) -> ClusterState:
        """
        Pop the cheapest valid candidate fault from the heap and either:
          - Add it directly (free case: all its checks in cl or unclaimed).
          - Trigger a merge  (collision: some check belongs to another cluster).

        Exactly one action is taken per call (mirrors LSD bits_per_step=1).
        Returns the surviving cluster (may differ from cl after a merge).
        """
        fault_to_checks         = self.fault_to_checks
        global_check_membership = self.global_check_membership
        global_fault_membership = self.global_fault_membership

        merge_list:    set[ClusterState] = set()
        connecting_j:  Optional[int]     = None
        connecting_vw: Optional[float]   = None

        while cl.heap:
            vw, j = heapq.heappop(cl.heap)

            # --- Skip stale heap entries --------------------------------------
            if vw > cl.dist.get(j, float('inf')):
                continue

            # --- Skip faults already in this cluster --------------------------
            # (This is key since a lazy merge would cause connecting_j be in the heap of merged cluster again)
            if global_fault_membership[j] is cl:
                continue

            # --- Check for collisions BEFORE doing anything -------------------
            colliding: set[ClusterState] = set()
            for c in fault_to_checks[j]:
                owner = global_check_membership[c]
                if owner is not None and owner is not cl:
                    colliding.add(owner)

            if colliding:
                # Collision: j spans this cluster and others.
                # j is the connecting edge — do NOT add it to any cluster yet.
                merge_list    = colliding
                connecting_j  = j
                connecting_vw = vw
            else:
                # Free: all checks are in cl or unclaimed.
                self._add_fault(cl, j, vw)

            break   # one action per step

        # --- Resolve merges ---------------------------------------------------
        if merge_list:
            cl = self._merge(cl, merge_list, connecting_j, connecting_vw)

        cl.valid = cl.rref.is_valid()
        return cl
