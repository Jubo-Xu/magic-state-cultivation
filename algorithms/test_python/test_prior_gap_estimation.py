"""
test_prior_gap_estimation.py  —  Tests for PriorGapEstimator.

Test strategy
-------------
Layer 1  —  stage3=False DEM matches CompiledDesaturationSampler.gap_dem exactly
    The DEM produced by PriorGapEstimator.from_desaturation(dem, stage3=False).dem
    must be string-identical to CompiledDesaturationSampler.from_task(task).gap_dem.
    Tested across several circuit parameter sets (dcolor/dsurface/r_growing/r_end)
    to give thorough coverage of the various error decomposition branches.

Layer 2  —  stage3=True structural checks
    2a: stage3=True output has strictly more detectors than stage3=False.
    2b: the number of extra detectors equals the count of distinct observable
        indices in the stage3=False DEM (one new detector per observable).
    2c: every error instruction that had at least one L_ target in the
        stage3=False DEM has a corresponding logical check-node target inserted
        immediately before each L_ target in the stage3=True DEM.
    2d: error instructions with no L_ targets are unchanged by stage3.
    2e: non-error instructions (detector declarations etc.) are preserved.

Fixtures
--------
Six cultivation-circuit parameter sets are built once at module load:
  CASES = list of (label, raw_dem, task) tuples

The raw_dem is extracted without decompose_errors so it feeds directly into
both modify_dem_as_desaturation and DesaturationSampler/from_task.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import sinter
import stim

import cultiv
import gen
from cultiv._decoding._desaturation_sampler import DesaturationSampler
from prior_gap_estimation import PriorGapEstimator


# ===========================================================================
# Shared fixtures — built once at module load
# ===========================================================================

def _make_case(
    dcolor: int,
    dsurface: int,
    r_growing: int,
    r_end: int,
    basis: str = 'Y',
    inject_style: str = 'unitary',
    noise: float = 1e-3,
) -> tuple[str, stim.DetectorErrorModel, sinter.Task]:
    """
    Build a noisy cultivation circuit and return
    (label, raw_dem, sinter_task).

    raw_dem  — unflattened, no decompose_errors; fed to both decoders.
    task     — sinter.Task used by CompiledDesaturationSampler.from_task.
    """
    label = f'd{dcolor}_s{dsurface}_rg{r_growing}_re{r_end}_{basis}'
    ideal = cultiv.make_end2end_cultivation_circuit(
        dcolor=dcolor,
        dsurface=dsurface,
        basis=basis,
        r_growing=r_growing,
        r_end=r_end,
        inject_style=inject_style,
    )
    noisy = gen.NoiseModel.uniform_depolarizing(noise).noisy_circuit_skipping_mpp_boundaries(ideal)
    raw_dem = noisy.detector_error_model()
    task = sinter.Task(circuit=noisy, detector_error_model=raw_dem)
    return label, raw_dem, task


# Six parameter combinations: varying dcolor, r_growing, r_end, and basis.
CASES: list[tuple[str, stim.DetectorErrorModel, sinter.Task]] = [
    _make_case(dcolor=3, dsurface=7,  r_growing=2, r_end=1),
    _make_case(dcolor=3, dsurface=7,  r_growing=3, r_end=0),
    _make_case(dcolor=3, dsurface=7,  r_growing=4, r_end=1),
    _make_case(dcolor=5, dsurface=11, r_growing=2, r_end=1),
    _make_case(dcolor=5, dsurface=11, r_growing=4, r_end=0),
    _make_case(dcolor=5, dsurface=11, r_growing=6, r_end=1),
]
CASE_IDS = [c[0] for c in CASES]


# ===========================================================================
# Layer 1: stage3=False must produce the identical DEM as desaturation
# ===========================================================================

class TestMatchesDesaturation:
    """
    PriorGapEstimator.from_desaturation(dem, obs_det_type='single').dem must be
    string-identical to CompiledDesaturationSampler.from_task(task).gap_dem.
    """

    @pytest.mark.parametrize('label,raw_dem,task', CASES, ids=CASE_IDS)
    def test_dem_string_identical_to_gap_dem(
        self,
        label: str,
        raw_dem: stim.DetectorErrorModel,
        task: sinter.Task,
    ) -> None:
        """obs_det_type='single' output must exactly match CompiledDesaturationSampler.gap_dem."""
        expected = DesaturationSampler().compiled_sampler_for_task(task).gap_dem
        got = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type='single').dem
        assert str(got) == str(expected), (
            f'[{label}] DEM mismatch.\n'
            f'--- expected (desaturation) ---\n{expected}\n'
            f'--- got (PriorGapEstimator) ---\n{got}\n'
        )

    @pytest.mark.parametrize('label,raw_dem,task', CASES, ids=CASE_IDS)
    def test_num_detectors_matches(
        self,
        label: str,
        raw_dem: stim.DetectorErrorModel,
        task: sinter.Task,
    ) -> None:
        """Detector count must match."""
        expected_ndets = DesaturationSampler().compiled_sampler_for_task(task).gap_dem.num_detectors
        got_ndets = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type='single').dem.num_detectors
        assert got_ndets == expected_ndets, (
            f'[{label}] detector count: expected {expected_ndets}, got {got_ndets}'
        )

    @pytest.mark.parametrize('label,raw_dem,task', CASES, ids=CASE_IDS)
    def test_static_call_equals_instance_dem(
        self,
        label: str,
        raw_dem: stim.DetectorErrorModel,
        task: sinter.Task,
    ) -> None:
        """Calling modify_dem_as_desaturation as a static method must give the same
        result as accessing .dem on a from_desaturation instance."""
        via_static = PriorGapEstimator.modify_dem_as_desaturation(raw_dem, obs_det_type='single')
        via_instance = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type='single').dem
        assert str(via_static) == str(via_instance), f'[{label}] static vs instance mismatch'


# ===========================================================================
# Layer 2: stage3=True structural checks
# ===========================================================================

def _count_distinct_obs_ids(dem: stim.DetectorErrorModel) -> int:
    """Return the number of distinct logical observable indices used in *dem*."""
    obs_ids: set[int] = set()
    for inst in dem:
        if inst.type == 'error':
            for t in inst.targets_copy():
                if t.is_logical_observable_id():
                    obs_ids.add(t.val)
    return len(obs_ids)


class TestPerObsAugmentation:
    """
    Structural invariants for obs_det_type='per_obs' output.
    Uses obs_det_type=None as the base (clipped DEM with no obs_det added),
    so that per_obs adds exactly n_obs new detectors.
    """

    @pytest.mark.parametrize('label,raw_dem,task', CASES, ids=CASE_IDS)
    def test_2a_per_obs_has_more_detectors(
        self,
        label: str,
        raw_dem: stim.DetectorErrorModel,
        task: sinter.Task,
    ) -> None:
        """obs_det_type='per_obs' must have strictly more detectors than None."""
        base = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type=None).dem
        augmented = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type='per_obs').dem
        assert augmented.num_detectors > base.num_detectors, (
            f'[{label}] per_obs should add detectors but got same count: '
            f'{augmented.num_detectors}'
        )

    @pytest.mark.parametrize('label,raw_dem,task', CASES, ids=CASE_IDS)
    def test_2b_extra_detectors_equal_distinct_observables(
        self,
        label: str,
        raw_dem: stim.DetectorErrorModel,
        task: sinter.Task,
    ) -> None:
        """Number of added detectors must equal distinct observable indices in
        the raw_dem (one logical check node per observable)."""
        base = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type=None).dem
        augmented = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type='per_obs').dem
        n_extra = augmented.num_detectors - base.num_detectors
        n_obs = _count_distinct_obs_ids(raw_dem)
        assert n_extra == n_obs, (
            f'[{label}] expected {n_obs} extra detectors (one per observable), '
            f'got {n_extra}'
        )

    @pytest.mark.parametrize('label,raw_dem,task', CASES, ids=CASE_IDS)
    def test_2c_logical_errors_have_check_node_inserted(
        self,
        label: str,
        raw_dem: stim.DetectorErrorModel,
        task: sinter.Task,
    ) -> None:
        """
        For every error instruction that has an L_ target in the None DEM,
        the corresponding instruction in 'per_obs' must have a detector target
        inserted immediately before each L_ target.
        That detector's id must fall in the range of newly added detectors.
        """
        base = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type=None).dem
        augmented = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type='per_obs').dem

        base_errors = [inst for inst in base if inst.type == 'error']
        aug_errors  = [inst for inst in augmented if inst.type == 'error']
        assert len(base_errors) == len(aug_errors), (
            f'[{label}] error instruction count changed: '
            f'{len(base_errors)} → {len(aug_errors)}'
        )

        logical_det_range = range(base.num_detectors, augmented.num_detectors)

        for idx, (b_inst, a_inst) in enumerate(zip(base_errors, aug_errors)):
            b_targets = b_inst.targets_copy()
            a_targets = a_inst.targets_copy()

            # Walk both target lists in lockstep.
            # Augmented list has extra det targets before each L_ target.
            a_pos = 0
            for b_t in b_targets:
                assert a_pos < len(a_targets), (
                    f'[{label}] instruction {idx}: ran out of augmented targets'
                )
                if b_t.is_logical_observable_id():
                    # Expect a new detector target before the L_ target.
                    inserted = a_targets[a_pos]
                    assert inserted.is_relative_detector_id(), (
                        f'[{label}] instruction {idx}: expected detector before L_{b_t.val}, '
                        f'got {inserted}'
                    )
                    assert inserted.val in logical_det_range, (
                        f'[{label}] instruction {idx}: inserted detector {inserted.val} '
                        f'not in logical-check-node range {list(logical_det_range)}'
                    )
                    a_pos += 1  # consume the inserted detector
                # Now the original target must match.
                assert a_targets[a_pos] == b_t, (
                    f'[{label}] instruction {idx}: target mismatch at position {a_pos}: '
                    f'expected {b_t}, got {a_targets[a_pos]}'
                )
                a_pos += 1

    @pytest.mark.parametrize('label,raw_dem,task', CASES, ids=CASE_IDS)
    def test_2d_non_logical_errors_unchanged(
        self,
        label: str,
        raw_dem: stim.DetectorErrorModel,
        task: sinter.Task,
    ) -> None:
        """Error instructions with no L_ targets must be identical in both DEMs."""
        base = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type=None).dem
        augmented = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type='per_obs').dem

        base_errors = [inst for inst in base if inst.type == 'error']
        aug_errors  = [inst for inst in augmented if inst.type == 'error']

        for idx, (b_inst, a_inst) in enumerate(zip(base_errors, aug_errors)):
            has_logical = any(t.is_logical_observable_id() for t in b_inst.targets_copy())
            if not has_logical:
                assert str(b_inst) == str(a_inst), (
                    f'[{label}] instruction {idx}: non-logical error changed by per_obs.\n'
                    f'  base:      {b_inst}\n'
                    f'  augmented: {a_inst}'
                )

    @pytest.mark.parametrize('label,raw_dem,task', CASES, ids=CASE_IDS)
    def test_2e_non_error_instructions_preserved(
        self,
        label: str,
        raw_dem: stim.DetectorErrorModel,
        task: sinter.Task,
    ) -> None:
        """
        All non-error instructions (detector declarations, etc.) from None
        must appear in 'per_obs' in the same order; 'per_obs' may only
        prepend additional detector declarations at the front.
        """
        base = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type=None).dem
        augmented = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type='per_obs').dem

        base_non_errors = [str(inst) for inst in base if inst.type != 'error']
        aug_non_errors  = [str(inst) for inst in augmented if inst.type != 'error']

        # per_obs prepends extra detector declarations; the tail must match.
        n_extra = augmented.num_detectors - base.num_detectors
        assert aug_non_errors[n_extra:] == base_non_errors, (
            f'[{label}] non-error instructions do not match after skipping '
            f'{n_extra} prepended logical-check-node detectors.\n'
            f'  base:      {base_non_errors}\n'
            f'  augmented: {aug_non_errors[n_extra:]}'
        )
