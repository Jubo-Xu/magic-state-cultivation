"""
prior_gap_estimation.py  —  Prior-gap estimator.

Provides PriorGapEstimator, whose main entry point is:

    modify_dem_as_desaturation(dem, obs_det_type='single')

which transforms a stim DEM in the same way as DesaturationSampler.from_task
(virtual pair nodes, X/Z decomposition, postselection clipping), then appends
logical check-node detector(s) according to obs_det_type:

  'single'  — one shared detector for all logical-affecting faults (default);
              matches DesaturationSampler.gap_dem exactly.
  'per_obs' — one detector per distinct observable index; finer-grained.
  None      — no logical check-node detector added; null(H_C) then includes
              fault combinations that flip logicals.
"""
from __future__ import annotations

import collections
import dataclasses
import heapq
import math
from typing import Literal, cast, Any, AbstractSet

import stim

from parity_matrix_construct import ParityCheckMatrices
from clustering import Clustering


# ---------------------------------------------------------------------------
# Minimal helpers copied from cultiv._decoding._desaturation_sampler so that
# this file stays self-contained (cultiv may not be on sys.path here).
# ---------------------------------------------------------------------------

def _int_to_flipped_bits(mask: int) -> list[int]:
    """Return indices of set bits in *mask*, LSB first."""
    bits: list[int] = []
    i = 0
    while mask:
        if mask & 1:
            bits.append(i)
        mask >>= 1
        i += 1
    return bits


@dataclasses.dataclass(frozen=True)
class _DemError:
    p: float
    det_set: frozenset[int]
    obs_mask: int

    @staticmethod
    def from_error_instruction(instruction: stim.DemInstruction) -> '_DemError':
        p = instruction.args_copy()[0]
        det_list: list[int] = []
        obs_mask = 0
        for target in instruction.targets_copy():
            if target.is_logical_observable_id():
                obs_mask ^= 1 << target.val
            elif target.is_relative_detector_id():
                det_list.append(target.val)
            elif target.is_separator():
                pass
            else:
                raise NotImplementedError(f'{instruction}')
        # XOR-reduce: a detector appearing twice cancels (GF(2)).
        counts = collections.Counter(det_list)
        det_set = frozenset(d for d, c in counts.items() if c % 2 == 1)
        return _DemError(p=p, det_set=det_set, obs_mask=obs_mask)

    def to_instruction(self) -> stim.DemInstruction:
        targets = []
        for d in self.det_set:
            targets.append(stim.target_relative_detector_id(d))
        for b in _int_to_flipped_bits(self.obs_mask)[::-1]:
            targets.append(stim.target_logical_observable_id(b))
        return stim.DemInstruction('error', [self.p], targets)

    @staticmethod
    def to_separated_instruction(parts: list['_DemError']) -> stim.DemInstruction:
        assert len(parts) >= 1
        assert len({p.p for p in parts}) == 1
        targets = []
        for k, part in enumerate(parts):
            if k:
                targets.append(stim.target_separator())
            for d in part.det_set:
                targets.append(stim.target_relative_detector_id(d))
            for b in _int_to_flipped_bits(part.obs_mask)[::-1]:
                targets.append(stim.target_logical_observable_id(b))
        return stim.DemInstruction('error', [parts[0].p], targets)


def _clipped_matchable_dem(
    flat_dem: stim.DetectorErrorModel,
    clip: AbstractSet[int],
) -> stim.DetectorErrorModel:
    """
    Remove edges whose detector sets contain ≥2 clipped detectors, and replace
    isolated clipped nodes with boundary edges (boundary = path to virtual
    boundary via Dijkstra over the unclipped graph).
    """
    neighbors: dict[int, dict[int, tuple[float, int]]] = collections.defaultdict(dict)
    heap: list[tuple[float, int, int]] = []
    boundaries: set[int] = set()

    for inst in flat_dem:
        if inst.type != 'error':
            continue
        if any(t.is_separator() for t in inst.targets_copy()):
            continue
        err = _DemError.from_error_instruction(inst)
        w = -math.log(err.p / (1 - err.p))
        if len(err.det_set) == 1:
            (a,) = err.det_set
            heapq.heappush(heap, (w, a, err.obs_mask))
            boundaries.add(a)
        elif len(err.det_set) == 2:
            a, b = err.det_set
            neighbors[a][b] = (w, err.obs_mask)
            neighbors[b][a] = (w, err.obs_mask)

    classification: dict[int, tuple[int, float]] = {}
    while heap:
        cost, node, obs = heapq.heappop(heap)
        if node in classification:
            continue
        classification[node] = (obs, cost)
        for neighbor, (extra_cost, extra_obs) in neighbors[node].items():
            if neighbor not in classification:
                heapq.heappush(heap, (cost + extra_cost, neighbor, obs ^ extra_obs))

    new_dem = stim.DetectorErrorModel()
    for inst in flat_dem:
        clipped_count = sum(
            t.is_relative_detector_id() and t.val in clip
            for t in inst.targets_copy()
        )
        if inst.type != 'error' or clipped_count < 2:
            new_dem.append(inst)
    for c in clip:
        if c not in boundaries and c in classification:
            obs, w = classification[c]
            p = math.exp(-w) / (math.exp(-w) + 1)
            targets = [stim.target_relative_detector_id(c)]
            for b in _int_to_flipped_bits(obs):
                targets.append(stim.target_logical_observable_id(b))
            new_dem.append('error', [p], targets)
    return new_dem


def _dem_with_obs_detector(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """
    Adds a single virtual detector that all errors with any logical observable
    effect connect to.  This is the standard desaturation "gap detector" used
    to compute the gap as the matching-weight difference with/without it active.
    """
    obs_det = stim.target_relative_detector_id(dem.num_detectors)
    new_dem = stim.DetectorErrorModel()
    new_dem.append('detector', [-10, -10, -10, -10, -10], [obs_det])
    for inst in dem:
        if inst.type == 'error':
            targets = inst.targets_copy()
            new_targets = []
            for t in targets:
                if t.is_logical_observable_id():
                    new_targets.append(obs_det)
                new_targets.append(t)
            new_dem.append('error', inst.args_copy(), new_targets)
        else:
            new_dem.append(inst)
    return new_dem


def _dem_with_logical_check_nodes(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """
    Stage 3 augmentation: adds one virtual logical check-node detector per
    distinct logical observable index found in *dem*.

    Every fault that flips observable L_i gets connected to the check node for
    L_i, immediately before the original L_i target (same insertion pattern as
    _dem_with_obs_detector).  Faults that flip multiple observables therefore
    connect to multiple logical check nodes.  The original L_ targets are kept.

    Different observables have separate check nodes; this is more fine-grained
    than _dem_with_obs_detector (which uses a single shared detector for all
    observables).
    """
    # Pass 1: collect all observable indices that appear.
    obs_ids_used: set[int] = set()
    for inst in dem:
        if inst.type == 'error':
            for t in inst.targets_copy():
                if t.is_logical_observable_id():
                    obs_ids_used.add(t.val)

    if not obs_ids_used:
        return dem.copy()

    base_det = dem.num_detectors
    # Map observable index → new detector target (sorted for determinism).
    obs_id_to_det: dict[int, stim.DemTarget] = {
        obs_id: stim.target_relative_detector_id(base_det + i)
        for i, obs_id in enumerate(sorted(obs_ids_used))
    }

    new_dem = stim.DetectorErrorModel()
    # Prepend the new logical check-node detector declarations.
    for i, _ in enumerate(sorted(obs_ids_used)):
        new_dem.append(
            'detector',
            [-10 - i, -10, -10, -10, -10],
            [stim.target_relative_detector_id(base_det + i)],
        )

    # Pass 2: copy all instructions, inserting the per-observable logical
    # check-node target immediately before each matching L_ target.
    for inst in dem:
        if inst.type == 'error':
            new_targets = []
            for t in inst.targets_copy():
                if t.is_logical_observable_id():
                    new_targets.append(obs_id_to_det[t.val])
                new_targets.append(t)
            new_dem.append('error', inst.args_copy(), new_targets)
        else:
            new_dem.append(inst)

    return new_dem


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PriorGapEstimator:
    """
    Prior-gap estimator.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        decompose: bool = False,
    ) -> None:
        self.dem = dem
        self.dem_origin = dem
        self.PCM = ParityCheckMatrices.from_DEM(dem, decompose=decompose)
        self.Clustering = Clustering(self.PCM)

    @classmethod
    def from_desaturation(
        cls,
        dem: stim.DetectorErrorModel,
        obs_det_type: str | None = 'single',
    ) -> 'PriorGapEstimator':
        """
        Build a PriorGapEstimator from a DEM after applying desaturation.

        Parameters
        ----------
        dem :
            Raw stim.DetectorErrorModel from the noisy circuit.
        obs_det_type :
            Controls which logical check-node detectors are appended.
            See modify_dem_as_desaturation for details.
        """
        modified_dem = cls.modify_dem_as_desaturation(dem, obs_det_type=obs_det_type)
        return cls(modified_dem, decompose=False)

    @staticmethod
    def modify_dem_as_desaturation(
        dem: stim.DetectorErrorModel,
        obs_det_type: str | None = 'single',
    ) -> stim.DetectorErrorModel:
        """
        Transforms *dem* exactly as DesaturationSampler.from_task does, then
        optionally appends logical check-node detector(s).

        Stage 1+2 (always applied)
        --------------------------
        1. Parse per-detector color/basis annotations from coords[4]:
             index 0-2 → X basis, colors r/g/b
             index 3-5 → Z basis, colors r/g/b
             index 6-7 → mixed/boundary ('_' color)
             coords[4] == -9 or missing → postselected (hidden from matcher)
        2. Identify postselected detectors:
             - Detectors annotated with -9 → hidden from matcher.
             - Detectors that are part of 2-det, same-basis, obs-flipping errors
               → also hidden.
             - rX and gZ color/basis detectors → visible-to-matcher postselected.
        3. Build virtual pair nodes for 3-body RGB errors and boundary 2-body
           color-pair errors, appending new virtual detectors to the DEM.
        4. Construct a matchable DEM:
             - Boundary color-pair errors  → kept + a virtual-pair-node copy.
             - 3-body RGB bulk errors      → split into 3 node-to-virtual-pair edges.
             - Simple 1–2-det same-basis   → kept as-is.
             - Mixed X/Z (≤2 X, ≤2 Z)     → decomposed into separated X ^ Z parts.
             - Anything else               → dropped.
        5. Clip the matchable DEM: remove edges with ≥2 postselected endpoints;
           replace isolated postselected nodes with Dijkstra-shortest boundary edges.

        Logical check-node step (controlled by obs_det_type)
        -----------------------------------------------------
        'single'  — add one shared detector for all logical-affecting faults
                    (_dem_with_obs_detector).  Matches DesaturationSampler exactly.
        'per_obs' — add one detector per distinct observable index
                    (_dem_with_logical_check_nodes).  Finer-grained: faults
                    flipping L_i connect only to det_i.
        None      — do not add any logical check-node detector.  null(H_C) then
                    includes fault combinations that flip logicals.

        Parameters
        ----------
        dem :
            A stim.DetectorErrorModel.  Detectors must carry 5-coordinate tuples
            where coords[4] encodes the color/basis annotation.
        obs_det_type :
            One of 'single', 'per_obs', or None (see above).

        Returns
        -------
        stim.DetectorErrorModel
        """
        dem = dem.flattened()
        num_dets = dem.num_detectors

        # ------------------------------------------------------------------ #
        # Parse color/basis annotations and identify postselected detectors.  #
        # ------------------------------------------------------------------ #
        det_coords = dem.get_detector_coordinates()
        det_bases: list[Literal['X', 'Z', '!']] = []
        det_colors: list[Literal['r', 'g', 'b', '_']] = []
        postselected_hidden: set[int] = set()
        postselected_visible: set[int] = set()

        for d in range(num_dets):
            coords = det_coords[d]
            if len(coords) <= 4 or coords[4] == -9:
                postselected_hidden.add(d)
                det_bases.append('!')
                det_colors.append('_')
                continue

            coord_annotation = int(coords[4])
            basis = cast(Any, 'XXXZZZXZ'[coord_annotation])
            color = cast(Any, 'rgbrgb__'[coord_annotation])
            det_bases.append(basis)
            det_colors.append(color)
            if (color == 'r' and basis == 'X') or (color == 'g' and basis == 'Z'):
                postselected_visible.add(d)

        # ------------------------------------------------------------------ #
        # Parse errors.                                                        #
        # ------------------------------------------------------------------ #
        errors: list[_DemError] = []
        for inst in dem:
            if inst.type != 'error':
                continue
            errors.append(_DemError.from_error_instruction(inst))

        # Classify single-basis 2-det errors for X/Z decomposition later.
        dets_to_obs: dict[frozenset[int], int] = {}
        for err in errors:
            bases = {det_bases[d] for d in err.det_set}
            if len(bases) == 1 and len(err.det_set) == 2 and err.obs_mask:
                a, b = err.det_set
                postselected_hidden.add(a)
                postselected_hidden.add(b)
            if len(bases) == 1:
                dets_to_obs[err.det_set] = err.obs_mask

        # ------------------------------------------------------------------ #
        # Identify virtual pair nodes for 3-body and boundary 2-body errors.  #
        # ------------------------------------------------------------------ #
        virtual_pair_nodes: set[frozenset[int]] = set()
        for err in errors:
            colors = {det_colors[d] for d in err.det_set}
            bases = {det_bases[d] for d in err.det_set}
            if len(err.det_set) == 3:
                if len(bases) == 1 and colors == {'r', 'g', 'b'}:
                    a, b, c = err.det_set
                    virtual_pair_nodes.add(frozenset([a, b]))
                    virtual_pair_nodes.add(frozenset([a, c]))
                    virtual_pair_nodes.add(frozenset([b, c]))
            elif (
                len(err.det_set) == 2
                and len(bases) == 1
                and colors in ({'r', 'g'}, {'r', 'b'}, {'b', 'g'})
            ):
                a, b = err.det_set
                virtual_pair_nodes.add(frozenset([a, b]))

        pair2virtual: dict[frozenset[int], int] = {}
        for pair in sorted(virtual_pair_nodes, key=lambda e: tuple(sorted(e))):
            k = len(pair2virtual) + num_dets
            pair2virtual[pair] = k
            a, b = pair
            if a != -1 and b != -1:
                det_coords[k] = [(x + y) / 2 for x, y in list(zip(det_coords[a], det_coords[b]))[:3]]
            else:
                c = a if a != -1 else b
                coords_c = list(det_coords[c][:3])
                coords_c[0] += 0.25
                coords_c[1] += 0.25
                coords_c[2] += 0.25
                det_coords[k] = coords_c

        # ------------------------------------------------------------------ #
        # Build matchable DEM.                                                 #
        # ------------------------------------------------------------------ #
        matchable_dem = stim.DetectorErrorModel()
        for k in range(num_dets + len(pair2virtual)):
            matchable_dem.append('detector', det_coords[k], [stim.target_relative_detector_id(k)])

        for err in errors:
            colors = collections.Counter(det_colors[d] for d in err.det_set)
            bases = collections.Counter(det_bases[d] for d in err.det_set)

            if len(err.det_set) == 2 and err.det_set in virtual_pair_nodes:
                # Boundary color-pair error: add normal edge + virtual-node boundary edge.
                virtual_err = _DemError(
                    p=err.p,
                    det_set=frozenset([pair2virtual[err.det_set]]),
                    obs_mask=err.obs_mask,
                )
                matchable_dem.append(err.to_instruction())
                matchable_dem.append(virtual_err.to_instruction())

            elif len(err.det_set) == 3 and len(bases) == 1 and colors == collections.Counter('rgb'):
                # 3-body bulk color-code error: split into 3 node-to-virtual-pair edges.
                assert err.obs_mask == 0
                for solo in err.det_set:
                    virtual_err = _DemError(
                        p=err.p,
                        obs_mask=err.obs_mask,
                        det_set=frozenset([solo, pair2virtual[err.det_set ^ frozenset([solo])]]),
                    )
                    matchable_dem.append(virtual_err.to_instruction())

            elif len(err.det_set) <= 2 and (bases.keys() == {'X'} or bases.keys() == {'Z'}):
                # Simple single-basis matchable edge.
                matchable_dem.append(err.to_instruction())

            elif bases['X'] <= 2 and bases['Z'] <= 2:
                # Mixed X/Z error: decompose into separated X and Z parts.
                xs = frozenset(d for d in err.det_set if det_bases[d] == 'X')
                zs = frozenset(d for d in err.det_set if det_bases[d] == 'Z')
                if xs not in dets_to_obs or zs not in dets_to_obs:
                    continue  # cannot decompose without knowing individual effects
                obs_x = dets_to_obs[xs]
                obs_z = dets_to_obs[zs]
                if obs_x ^ obs_z != err.obs_mask:
                    continue  # inconsistent decomposition (e.g. distance-3 logical)
                x_part = _DemError(p=err.p, det_set=xs, obs_mask=obs_x)
                z_part = _DemError(p=err.p, det_set=zs, obs_mask=obs_z)
                matchable_dem.append(_DemError.to_separated_instruction([x_part, z_part]))

            # else: too complicated to decompose — skip.

        # ------------------------------------------------------------------ #
        # Clip postselected detectors.                                         #
        # ------------------------------------------------------------------ #
        clipped = _clipped_matchable_dem(matchable_dem, postselected_hidden)

        # ------------------------------------------------------------------ #
        # Logical check-node step (controlled by obs_det_type).              #
        # ------------------------------------------------------------------ #
        if obs_det_type is None:
            result = clipped
        elif obs_det_type == 'single':
            result = _dem_with_obs_detector(clipped)
        elif obs_det_type == 'per_obs':
            result = _dem_with_logical_check_nodes(clipped)
        else:
            raise ValueError(
                f"obs_det_type must be 'single', 'per_obs', or None; got {obs_det_type!r}"
            )

        return result
