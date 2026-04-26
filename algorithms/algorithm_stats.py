"""
algorithm_stats.py  —  Statistics and diagnostics for clustering algorithms.

Functions
---------
dem_sample_and_pcm_gen(dem, shots)
    Build a ParityCheckMatrices from *dem* and draw *shots* syndrome samples.
    Returns (pcm, det_results) for reuse across multiple test functions.

check_clustering_nullbasis_num(pcm, det_results)
    Run the clustering algorithm on pre-sampled syndromes and collect the
    distribution of null-space basis sizes across all final active valid clusters.

check_clustering_nullbasis_num_plot(counter, ax=None)
    Bar-plot the distribution returned by check_clustering_nullbasis_num.

check_clustering_total_nullbasis_num_per_shot(pcm, det_results, collect_type)
    For each shot, compute a single scalar from all active valid clusters:
    sum or max of their null-space basis sizes.

check_clustering_total_nullbasis_num_per_shot_plot(per_shot, plt_type, ...)
    Plot the per-shot values as a history line or a distribution bar chart.

check_gap_vs_nullbasis_count(compiled, estimator, threshold, shots, collect_type)
    Joint analysis: for each shot compute both the DesaturationSampler gap and
    the clustering null-space basis count, then split results into accepted
    (gap >= threshold) and gap-discarded (gap < threshold) groups.

check_gap_vs_nullbasis_count_plot(result, axs=None, ...)
    Side-by-side bar charts of null-basis distributions for accepted vs
    gap-discarded shots, with % of shots having null_dim > 0 annotated.

sample_gap_shots(compiled, estimator, shots)
    Draw syndrome samples from the gap circuit once and pre-compute all
    shot-level quantities (gaps, desaturation/PyMatching correctness) that
    do not depend on clustering parameters.  Use together with
    check_gap_vs_logical_error_nullbasis_count_from_samples.

check_gap_vs_logical_error_nullbasis_count_from_samples(samples, estimator, threshold, ...)
    Run clustering analysis on pre-sampled data from sample_gap_shots.
    Call this multiple times with different clustering_type / over_grow_step /
    bits_per_step to compare algorithms on the exact same syndrome samples.

check_gap_vs_logical_error_nullbasis_count(compiled, estimator, threshold, shots, collect_type)
    Convenience wrapper: draw fresh samples and run analysis in one call.
    Same as check_gap_vs_nullbasis_count but counts only the logical-error
    null-space basis vectors per cluster (those that flip at least one logical
    observable), using Clustering.create_degenerate_cycle_regions().

check_gap_vs_logical_error_nullbasis_count_plot(result, axs=None, ...)
    Side-by-side bar charts of logical-error null-basis distributions for
    accepted vs gap-discarded shots.
"""
from __future__ import annotations

import collections

import numpy as np
import matplotlib.pyplot as plt
import pymatching
import stim

from parity_matrix_construct import ParityCheckMatrices


def _plain_dem_from_gap_dem(gap_dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Return the matchable DEM with the obs_det augmentation stripped out.

    _dem_with_obs_detector inserts obs_det before each logical-observable target
    but keeps the original L_i targets.  Reversing it is just filtering out every
    relative_detector_id(obs_det_id) target and dropping the obs_det detector
    instruction.
    """
    obs_det_id = gap_dem.num_detectors - 1
    plain_dem = stim.DetectorErrorModel()
    for inst in gap_dem:
        if inst.type == 'detector':
            if any(t.is_relative_detector_id() and t.val == obs_det_id
                   for t in inst.targets_copy()):
                continue
            plain_dem.append(inst)
        elif inst.type == 'error':
            new_targets = [t for t in inst.targets_copy()
                           if not (t.is_relative_detector_id() and t.val == obs_det_id)]
            plain_dem.append('error', inst.args_copy(), new_targets)
        else:
            plain_dem.append(inst)
    return plain_dem


def _get_clustering_impl(clustering_type: str):
    """
    Resolve the clustering implementation backend.

    Parameters
    ----------
    clustering_type : str
        'python_original'      uses algorithms/clustering.py.
        'cplus_original'       uses algorithms/clustering_cpp.py.
        'python_overgrow'      uses algorithms/clustering_overgrow.py.
        'python_overgrow_batch' uses algorithms/clustering_overgrow_batch.py.
        'cplus_overgrow_batch' uses algorithms/clustering_overgrow_batch_cpp.py
                               (C++ engine; fastest).
    """
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
    """Build the keyword-argument dict for clustering.run() based on implementation type."""
    if clustering_type in ('python_overgrow_batch', 'cplus_overgrow_batch'):
        return {'over_grow_step': over_grow_step, 'bits_per_step': bits_per_step}
    if clustering_type == 'python_overgrow':
        return {'over_grow_step': over_grow_step}
    return {}


def _get_prior_gap_estimator_impl(prior_gap_estimate_type: str):
    """
    Resolve the PriorGapEstimatorUse implementation.

    Parameters
    ----------
    prior_gap_estimate_type : str
        'python' : PriorGapEstimatorUse from prior_gap_estimation_use.py.
                   Constructor takes a DEM.  execute() runs a Python-level loop.
        'cpp'    : PriorGapEstimatorUse from prior_gap_estimator_cpp.py.
                   Constructor takes a PCM.  execute_batch() runs entirely in C++.
    """
    if prior_gap_estimate_type == 'python':
        from prior_gap_estimation_use import PriorGapEstimatorUse
        return PriorGapEstimatorUse
    if prior_gap_estimate_type == 'cpp':
        from prior_gap_estimator_cpp import PriorGapEstimatorUse
        return PriorGapEstimatorUse
    raise ValueError(
        f"prior_gap_estimate_type must be 'python' or 'cpp', "
        f"got {prior_gap_estimate_type!r}"
    )


def dem_sample_and_pcm_gen(
    dem: stim.DetectorErrorModel,
    shots: int,
    decompose: bool = False,
) -> tuple[ParityCheckMatrices, np.ndarray]:
    """
    Build a ParityCheckMatrices from *dem* and draw *shots* syndrome samples.

    Separating this from individual test functions allows the same PCM and
    sampled syndromes to be reused across multiple analyses without re-sampling.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The (possibly modified) DEM.  Typically the output of
        PriorGapEstimator.modify_dem_as_desaturation.
    shots : int
        Number of syndrome samples to draw.
    decompose : bool, default=False
        Whether to decompose the parity-check matrices.

    Returns
    -------
    pcm : ParityCheckMatrices
        Parity-check matrices built from *dem*.
    det_results : np.ndarray, shape (shots, num_detectors), dtype bool
        Sampled detector outcomes.  Each row is one syndrome.
    """
    pcm = ParityCheckMatrices.from_DEM(dem, decompose=decompose)

    # CompiledDemSampler.sample() returns (det_results, obs_results, errors).
    # Observables are already encoded as detectors in the modified DEM via
    # _dem_with_obs_detector, so we only need det_results.
    det_results, _, _ = dem.compile_sampler().sample(shots)

    return pcm, det_results

def check_clustering_nullbasis_num(
    pcm: ParityCheckMatrices,
    det_results: np.ndarray,
    clustering_type: str = 'python_original',
    over_grow_step: int = 0,
    bits_per_step: int = 1,
) -> collections.Counter:
    """
    Run the clustering algorithm on pre-sampled syndromes and record the number
    of null-space basis vectors (len(cl.rref.Z)) for every final active valid
    cluster.

    Parameters
    ----------
    pcm : ParityCheckMatrices
        Output of dem_sample_and_pcm_gen.
    det_results : np.ndarray, shape (shots, num_detectors)
        Sampled detector outcomes, output of dem_sample_and_pcm_gen.
    clustering_type : {'python_original', 'cplus_original', 'python_overgrow', 'python_overgrow_batch'}
        Which clustering implementation to use.
    over_grow_step : int, default=0
        Passed to run() for 'python_overgrow' and 'python_overgrow_batch'.
    bits_per_step : int, default=1
        Passed to run() for 'python_overgrow_batch'.

    Returns
    -------
    collections.Counter
        Maps  null_basis_count (int)  →  total number of clusters (summed over
        all shots) that had exactly that many null-space basis vectors.
    """
    ClusteringImpl = _get_clustering_impl(clustering_type)
    clustering = ClusteringImpl(pcm)
    run_kwargs = _make_run_kwargs(clustering_type, over_grow_step, bits_per_step)
    counter: collections.Counter = collections.Counter()

    for i in range(len(det_results)):
        syndrome = det_results[i].astype(np.uint8)
        clustering.run(syndrome, **run_kwargs)
        for cl in clustering.active_clusters:
            if cl.valid:
                counter[len(cl.rref.Z)] += 1

    return counter

def check_clustering_nullbasis_num_plot(
    counter: collections.Counter,
    ax: plt.Axes | None = None,
    title: str = 'Null-space basis size distribution across clusters',
) -> plt.Axes:
    """
    Bar-plot the distribution of null-space basis sizes returned by
    check_clustering_nullbasis_num.

    Parameters
    ----------
    counter : collections.Counter
        Output of check_clustering_nullbasis_num.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if None.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    if not counter:
        ax.set_title(title)
        ax.set_xlabel('Number of null-space basis vectors per cluster')
        ax.set_ylabel('Number of clusters')
        return ax

    max_k = max(counter.keys())
    xs = list(range(max_k + 1))
    ys = [counter.get(k, 0) for k in xs]

    ax.bar(xs, ys, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Number of null-space basis vectors per cluster')
    ax.set_ylabel('Number of clusters (total over all shots)')
    ax.set_title(title)
    ax.set_xticks(xs)

    return ax


def check_clustering_total_nullbasis_num_per_shot(
    pcm: ParityCheckMatrices,
    det_results: np.ndarray,
    collect_type: str = 'sum',
    clustering_type: str = 'python_original',
    over_grow_step: int = 0,
    bits_per_step: int = 1,
) -> list[int]:
    """
    For each shot, compute a single scalar summarising the null-space basis
    sizes across all final active valid clusters.

    Parameters
    ----------
    pcm : ParityCheckMatrices
        Output of dem_sample_and_pcm_gen.
    det_results : np.ndarray, shape (shots, num_detectors)
        Sampled detector outcomes, output of dem_sample_and_pcm_gen.
    collect_type : {'sum', 'max'}
        'sum' — value per shot = sum of len(cl.rref.Z) over all active valid clusters.
        'max' — value per shot = max of len(cl.rref.Z) over all active valid clusters
                (0 if there are no active valid clusters).
    clustering_type : {'python_original', 'cplus_original', 'python_overgrow', 'python_overgrow_batch'}
        Which clustering implementation to use.
    over_grow_step : int, default=0
        Passed to run() for 'python_overgrow' and 'python_overgrow_batch'.
    bits_per_step : int, default=1
        Passed to run() for 'python_overgrow_batch'.

    Returns
    -------
    list[int]
        One entry per shot.
    """
    if collect_type not in ('sum', 'max'):
        raise ValueError(f"collect_type must be 'sum' or 'max', got {collect_type!r}")

    ClusteringImpl = _get_clustering_impl(clustering_type)
    clustering = ClusteringImpl(pcm)
    run_kwargs = _make_run_kwargs(clustering_type, over_grow_step, bits_per_step)
    per_shot: list[int] = []

    for i in range(len(det_results)):
        syndrome = det_results[i].astype(np.uint8)
        clustering.run(syndrome, **run_kwargs)
        sizes = [len(cl.rref.Z) for cl in clustering.active_clusters if cl.valid]
        if collect_type == 'sum':
            per_shot.append(sum(sizes))
        else:  # 'max'
            per_shot.append(max(sizes) if sizes else 0)

    return per_shot


def check_clustering_total_nullbasis_num_per_shot_plot(
    per_shot: list[int],
    plt_type: str = 'history',
    ax: plt.Axes | None = None,
    title: str = 'Null-space basis summary per shot',
    collect_type: str = '',
) -> plt.Axes:
    """
    Plot the per-shot null-space basis summary returned by
    check_clustering_total_nullbasis_num_per_shot.

    Parameters
    ----------
    per_shot : list[int]
        Output of check_clustering_total_nullbasis_num_per_shot.
    plt_type : {'history', 'distribution'}
        'history'      — x = shot index, y = value for that shot (line plot).
        'distribution' — x = value, y = number of shots with that value (bar plot).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if None.
    title : str
        Plot title.
    collect_type : str, optional
        If provided, appended to axis labels to clarify whether values are
        sums or maxima (e.g. pass the same collect_type used when collecting).

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if plt_type not in ('history', 'distribution'):
        raise ValueError(f"plt_type must be 'history' or 'distribution', got {plt_type!r}")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    suffix = f' ({collect_type})' if collect_type else ''

    if plt_type == 'history':
        ax.plot(range(len(per_shot)), per_shot, linewidth=0.8, color='steelblue')
        ax.set_xlabel('Shot index')
        ax.set_ylabel(f'Null-space basis count per shot{suffix}')
        ax.set_title(title)

    else:  # 'distribution'
        counter = collections.Counter(per_shot)
        max_k = max(counter.keys()) if counter else 0
        xs = list(range(max_k + 1))
        ys = [counter.get(k, 0) for k in xs]
        ax.bar(xs, ys, color='steelblue', edgecolor='white', linewidth=0.5)
        ax.set_xlabel(f'Null-space basis count per shot{suffix}')
        ax.set_ylabel('Number of shots')
        ax.set_title(title)
        ax.set_xticks(xs)

    return ax


def check_gap_vs_nullbasis_count(
    compiled,
    estimator,
    threshold: int,
    shots: int,
    collect_type: str = 'max',
    clustering_type: str = 'python_original',
    over_grow_step: int = 0,
    bits_per_step: int = 1,
) -> dict:
    """
    Joint analysis: for each shot draw a syndrome from the gap circuit, compute
    the DesaturationSampler gap, and run clustering to get the null-space basis
    count.  Shots are split into three groups:

      - postselection-discarded : a postselected detector fired (excluded).
      - gap-discarded            : gap < threshold.
      - accepted                 : gap >= threshold.

    Parameters
    ----------
    compiled :
        CompiledDesaturationSampler for the task.  Must expose
        `gap_circuit`, `gap_circuit_sampler`, `postselected_detectors`,
        and `_decode_batch_overwrite_last_byte`.
    estimator :
        PriorGapEstimator built with obs_det_type=None so that null(H_C)
        contains logical error vectors.  Its PCM must have exactly
        compiled.gap_circuit.num_detectors - 1  rows.
    threshold : int
        Accept if gap >= threshold, gap-discard otherwise.
    shots : int
        Number of syndrome samples to draw.
    collect_type : {'max', 'sum'}
        Aggregation across clusters per shot ('max' recommended).
    clustering_type : {'python_original', 'cplus_original', 'python_overgrow', 'python_overgrow_batch'}
        Which clustering implementation to use.
    over_grow_step : int, default=0
        Passed to run() for 'python_overgrow' and 'python_overgrow_batch'.
    bits_per_step : int, default=1
        Passed to run() for 'python_overgrow_batch'.

    Returns
    -------
    dict with keys:
        'accept_nullbasis'  : list[int]  null-basis counts for accepted shots.
        'discard_nullbasis' : list[int]  null-basis counts for gap-discarded shots.
        'postsel_discards'  : int        shots removed by postselection.
        'threshold'         : int        threshold used.
        'accept_gaps'       : list[int]  gap values for accepted shots.
        'discard_gaps'      : list[int]  gap values for gap-discarded shots.
    """
    if collect_type not in ('sum', 'max'):
        raise ValueError(f"collect_type must be 'sum' or 'max', got {collect_type!r}")

    n_cluster_dets  = estimator.PCM.H.shape[0]
    n_logical       = estimator.PCM.n_logical_check_nodes
    n_original_dets = n_cluster_dets - n_logical
    n_gap_dets      = compiled.gap_circuit.num_detectors
    assert n_gap_dets == n_original_dets + 1, (
        f"gap_circuit has {n_gap_dets} detectors but expected "
        f"estimator.PCM rows ({n_cluster_dets}) - n_logical ({n_logical}) + 1 (obs_det) "
        f"= {n_original_dets + 1}."
    )

    # ------------------------------------------------------------------ #
    # Sample syndromes from the gap circuit (unpacked for easy indexing). #
    # ------------------------------------------------------------------ #
    dets_all, _ = compiled.gap_circuit_sampler.sample(
        shots, separate_observables=True, bit_packed=False,
    )  # shape: (shots, n_gap_dets)

    # Postselection: drop shots where any postselected detector fired.
    keep = np.ones(shots, dtype=bool)
    for d in compiled.postselected_detectors:
        keep &= ~dets_all[:, d].astype(bool)
    postsel_discards = int(np.sum(~keep))

    dets_kept = dets_all[keep]  # shape: (n_kept, n_gap_dets)

    if len(dets_kept) == 0:
        return {
            'accept_nullbasis': [],
            'discard_nullbasis': [],
            'postsel_discards': postsel_discards,
            'threshold': threshold,
            'accept_gaps': [],
            'discard_gaps': [],
        }

    # ------------------------------------------------------------------ #
    # Compute per-shot gap using the desaturation decoder.                #
    # ------------------------------------------------------------------ #
    dets_packed = np.packbits(dets_kept, bitorder='little', axis=1)
    _, gaps_float = compiled._decode_batch_overwrite_last_byte(dets_packed)
    gaps = np.round(gaps_float).astype(int)
    accept_mask = gaps >= threshold

    # ------------------------------------------------------------------ #
    # Clustering on the first n_original_dets columns (no obs_det).      #
    # Logical check node columns are zero-padded (never fire in samples). #
    # ------------------------------------------------------------------ #
    dets_cluster = np.zeros((len(dets_kept), n_cluster_dets), dtype=np.uint8)
    dets_cluster[:, :n_original_dets] = dets_kept[:, :n_original_dets]
    ClusteringImpl = _get_clustering_impl(clustering_type)
    clustering = ClusteringImpl(estimator.PCM)
    run_kwargs = _make_run_kwargs(clustering_type, over_grow_step, bits_per_step)
    nullbasis_per_shot: list[int] = []
    for i in range(len(dets_cluster)):
        syndrome = dets_cluster[i].astype(np.uint8)
        clustering.run(syndrome, **run_kwargs)
        sizes = [len(cl.rref.Z) for cl in clustering.active_clusters if cl.valid]
        if collect_type == 'max':
            nullbasis_per_shot.append(max(sizes) if sizes else 0)
        else:
            nullbasis_per_shot.append(sum(sizes))

    nullbasis_arr = np.array(nullbasis_per_shot)

    return {
        'accept_nullbasis':  nullbasis_arr[accept_mask].tolist(),
        'discard_nullbasis': nullbasis_arr[~accept_mask].tolist(),
        'postsel_discards':  postsel_discards,
        'threshold':         threshold,
        'accept_gaps':       gaps[accept_mask].tolist(),
        'discard_gaps':      gaps[~accept_mask].tolist(),
    }


def check_gap_vs_nullbasis_count_plot(
    result: dict,
    axs=None,
    title: str = 'Null-space basis count: accepted vs gap-discarded',
    collect_type: str = '',
):
    """
    Side-by-side bar charts of null-space basis count distributions for accepted
    and gap-discarded shots.

    Parameters
    ----------
    result : dict
        Output of check_gap_vs_nullbasis_count.
    axs : array-like of two Axes, optional
        If None, a new (1×2) figure is created.
    title : str
        Super-title for the figure.
    collect_type : str, optional
        Appended to axis labels (e.g. 'max' or 'sum').

    Returns
    -------
    axs : np.ndarray of matplotlib.axes.Axes
    """
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(12, 4))

    suffix = f' ({collect_type})' if collect_type else ''
    colors = {'Accepted': 'steelblue', 'Gap-discarded': 'tomato'}

    for ax, key, label, color in [
        (axs[0], 'accept_nullbasis',  'Accepted',      colors['Accepted']),
        (axs[1], 'discard_nullbasis', 'Gap-discarded', colors['Gap-discarded']),
    ]:
        vals = result[key]
        n_total = len(vals)
        if n_total == 0:
            ax.set_title(f'{label}\n(no shots)')
            ax.set_xlabel(f'Null-space basis count{suffix}')
            ax.set_ylabel('Number of shots')
            continue

        counter = collections.Counter(vals)
        max_k = max(counter.keys())
        xs = list(range(max_k + 1))
        ys = [counter.get(k, 0) for k in xs]
        ax.bar(xs, ys, color=color, edgecolor='white', linewidth=0.5)
        ax.set_xlabel(f'Null-space basis count{suffix}')
        ax.set_ylabel('Number of shots')
        n_nonzero = sum(v for k, v in counter.items() if k > 0)
        ax.set_title(
            f'{label}  (gap {"≥" if key == "accept_nullbasis" else "<"} '
            f'{result["threshold"]})\n'
            f'n={n_total}  |  null_dim > 0 : {n_nonzero} ({100 * n_nonzero / n_total:.1f}%)'
        )
        if max_k <= 30:
            ax.set_xticks(xs)

    axs[0].get_figure().suptitle(title, y=1.02, fontsize=11)
    return axs


def sample_gap_shots(
    compiled,
    estimator,
    shots: int,
) -> dict:
    """
    Draw syndrome samples from the gap circuit and pre-compute all
    shot-level quantities that do not depend on clustering parameters.

    Call this once, then pass the result to
    check_gap_vs_logical_error_nullbasis_count_from_samples with different
    clustering_type / over_grow_step / bits_per_step to compare algorithms
    on the exact same set of samples.

    Parameters
    ----------
    compiled :
        CompiledDesaturationSampler for the task.
    estimator :
        PriorGapEstimator built with obs_det_type=None.
    shots : int
        Number of syndrome samples to draw before postselection.

    Returns
    -------
    dict:
        'dets_kept'        : np.ndarray (n_kept, n_gap_dets) uint8
        'obs_kept'         : np.ndarray (n_kept, n_obs) uint8
        'gaps_float'       : np.ndarray (n_kept,) float  — desaturation gap per shot
        'actual_flip'      : np.ndarray (n_kept,) bool   — true observable flip
        'incorrect'        : np.ndarray (n_kept,) bool   — desaturation incorrect correction
        'plain_incorrect'  : np.ndarray (n_kept,) bool   — plain PyMatching incorrect
        'postsel_discards' : int   — shots dropped by postselection
        'n_original_dets'  : int   — detector count in the plain DEM (no obs_det)
        'n_cluster_dets'   : int   — row count of estimator.PCM.H
    """
    n_cluster_dets  = estimator.PCM.H.shape[0]
    n_logical       = estimator.PCM.n_logical_check_nodes
    n_original_dets = n_cluster_dets - n_logical
    n_gap_dets      = compiled.gap_circuit.num_detectors
    assert n_gap_dets == n_original_dets + 1, (
        f"gap_circuit has {n_gap_dets} detectors but expected "
        f"estimator.PCM rows ({n_cluster_dets}) - n_logical ({n_logical}) + 1 (obs_det) "
        f"= {n_original_dets + 1}."
    )

    dets_all, obs_all = compiled.gap_circuit_sampler.sample(
        shots, separate_observables=True, bit_packed=False,
    )

    keep = np.ones(shots, dtype=bool)
    for d in compiled.postselected_detectors:
        keep &= ~dets_all[:, d].astype(bool)
    postsel_discards = int(np.sum(~keep))

    dets_kept = dets_all[keep]
    obs_kept  = obs_all[keep]

    if len(dets_kept) == 0:
        return {
            'dets_kept':        dets_kept,
            'obs_kept':         obs_kept,
            'gaps_float':       np.array([], dtype=np.float64),
            'actual_flip':      np.array([], dtype=bool),
            'incorrect':        np.array([], dtype=bool),
            'plain_incorrect':  np.array([], dtype=bool),
            'postsel_discards': postsel_discards,
            'n_original_dets':  n_original_dets,
            'n_cluster_dets':   n_cluster_dets,
        }

    # Desaturation decode — copy so in-place bit-toggling does not corrupt dets_kept.
    dets_packed = np.packbits(dets_kept, bitorder='little', axis=1).copy()
    predictions, gaps_float = compiled._decode_batch_overwrite_last_byte(dets_packed)

    actual_flip = obs_kept.any(axis=1)
    incorrect   = predictions.astype(bool) ^ actual_flip

    # Plain PyMatching decode on the non-obs_det columns.
    plain_dem     = _plain_dem_from_gap_dem(compiled.gap_dem)
    plain_decoder = pymatching.Matching.from_detector_error_model(plain_dem)
    dets_plain_packed = np.packbits(
        dets_kept[:, :n_original_dets], bitorder='little', axis=1
    )
    plain_preds_raw, _ = plain_decoder.decode_batch(
        dets_plain_packed,
        bit_packed_shots=True,
        bit_packed_predictions=False,
        return_weights=True,
    )
    plain_incorrect = np.any(
        plain_preds_raw.astype(bool) ^ obs_kept.astype(bool), axis=1
    )

    return {
        'dets_kept':        dets_kept,
        'obs_kept':         obs_kept,
        'gaps_float':       gaps_float,
        'actual_flip':      actual_flip,
        'incorrect':        incorrect,
        'plain_incorrect':  plain_incorrect,
        'postsel_discards': postsel_discards,
        'n_original_dets':  n_original_dets,
        'n_cluster_dets':   n_cluster_dets,
    }


def check_gap_vs_logical_error_nullbasis_count_from_samples(
    samples: dict,
    estimator,
    threshold: int,
    collect_type: str = 'max',
    clustering_type: str = 'python_original',
    over_grow_step: int = 0,
    bits_per_step: int = 1,
) -> dict:
    """
    Run clustering analysis on pre-sampled shot data from sample_gap_shots.

    Use this with sample_gap_shots to compare clustering algorithms on the
    exact same syndrome samples — draw once, analyse many times:

        samples = sample_gap_shots(compiled, estimator, shots)
        r1 = check_gap_vs_logical_error_nullbasis_count_from_samples(
                 samples, estimator, threshold, clustering_type='python_original')
        r2 = check_gap_vs_logical_error_nullbasis_count_from_samples(
                 samples, estimator, threshold, clustering_type='python_overgrow_batch',
                 over_grow_step=2, bits_per_step=4)

    Parameters
    ----------
    samples : dict
        Output of sample_gap_shots.
    estimator :
        PriorGapEstimator used when calling sample_gap_shots.
    threshold : int
        Accept if gap >= threshold.
    collect_type : {'max', 'sum'}
        How to aggregate logical-error basis counts across clusters per shot.
    clustering_type : {'python_original', 'cplus_original', 'python_overgrow', 'python_overgrow_batch'}
        Which clustering implementation to use.
    over_grow_step : int, default=0
        Passed to run() for 'python_overgrow' and 'python_overgrow_batch'.
    bits_per_step : int, default=1
        Passed to run() for 'python_overgrow_batch'.

    Returns
    -------
    Same dict as check_gap_vs_logical_error_nullbasis_count.
    """
    if collect_type not in ('sum', 'max'):
        raise ValueError(f"collect_type must be 'sum' or 'max', got {collect_type!r}")

    dets_kept        = samples['dets_kept']
    gaps_float       = samples['gaps_float']
    actual_flip      = samples['actual_flip']
    incorrect        = samples['incorrect']
    plain_incorrect  = samples['plain_incorrect']
    postsel_discards = samples['postsel_discards']
    n_original_dets  = samples['n_original_dets']
    n_cluster_dets   = samples['n_cluster_dets']

    _empty: dict = {
        'log_err_nullbasis':         [],
        'gaps':                      [],
        'accepted':                  [],
        'incorrect_correction':      [],
        'postsel_discards':          postsel_discards,
        'threshold':                 threshold,
        'accept_log_err_nullbasis':  [],
        'discard_log_err_nullbasis': [],
        'accept_gaps':               [],
        'discard_gaps':              [],
        'accept_incorrect':          [],
        'discard_incorrect':         [],
    }

    if len(dets_kept) == 0:
        return _empty

    accept_mask = gaps_float >= threshold

    # ------------------------------------------------------------------ #
    # Clustering on the first n_original_dets columns (no obs_det).      #
    # ------------------------------------------------------------------ #
    dets_cluster = np.zeros((len(dets_kept), n_cluster_dets), dtype=np.uint8)
    dets_cluster[:, :n_original_dets] = dets_kept[:, :n_original_dets]
    ClusteringImpl = _get_clustering_impl(clustering_type)
    clustering = ClusteringImpl(estimator.PCM)
    run_kwargs = _make_run_kwargs(clustering_type, over_grow_step, bits_per_step)
    log_err_per_shot: list[int] = []

    for i in range(len(dets_cluster)):
        syndrome = dets_cluster[i].astype(np.uint8)
        regions = clustering.run_and_create_degenerate_cycle_regions(syndrome, **run_kwargs)
        counts = [len(groups['logical_error']) for groups in regions.values()]
        if collect_type == 'max':
            log_err_per_shot.append(max(counts) if counts else 0)
        else:
            log_err_per_shot.append(sum(counts))

    log_err_arr = np.array(log_err_per_shot)

    # ------------------------------------------------------------------ #
    # Per-group summary statistics.                                        #
    # ------------------------------------------------------------------ #
    def _group_summary(mask: np.ndarray, label: str) -> dict:
        n = int(mask.sum())
        if n == 0:
            summary = {
                'n_shots':               0,
                'n_actual_flip':         0,
                'n_incorrect':           0,
                'gap_min_incorrect':     None,
                'gap_max_incorrect':     None,
                'n_plain_incorrect':     0,
                'n_nonzero_basis':       0,
                'basis_min':             None,
                'basis_max':             None,
                'zero_basis_n_actual_flip':       0,
                'zero_basis_n_incorrect':         0,
                'zero_basis_n_plain_incorrect':   0,
                'nonzero_basis_n_actual_flip':    0,
                'nonzero_basis_n_incorrect':      0,
                'nonzero_basis_n_plain_incorrect': 0,
            }
            _print_summary(summary, label, threshold)
            return summary

        g_gaps         = gaps_float[mask]
        g_actual       = actual_flip[mask]
        g_incorrect    = incorrect[mask]
        g_plain_incorr = plain_incorrect[mask]
        g_basis        = log_err_arr[mask]
        zero_mask      = g_basis == 0
        nonzero_mask   = g_basis > 0
        incorrect_gaps = g_gaps[g_incorrect]

        summary = {
            'n_shots':               n,
            'n_actual_flip':         int(g_actual.sum()),
            'n_incorrect':           int(g_incorrect.sum()),
            'gap_min_incorrect':     float(incorrect_gaps.min()) if len(incorrect_gaps) else None,
            'gap_max_incorrect':     float(incorrect_gaps.max()) if len(incorrect_gaps) else None,
            'n_plain_incorrect':     int(g_plain_incorr.sum()),
            'n_nonzero_basis':       int(nonzero_mask.sum()),
            'basis_min':             int(g_basis.min()),
            'basis_max':             int(g_basis.max()),
            'zero_basis_n_actual_flip':        int(g_actual[zero_mask].sum()),
            'zero_basis_n_incorrect':          int(g_incorrect[zero_mask].sum()),
            'zero_basis_n_plain_incorrect':    int(g_plain_incorr[zero_mask].sum()),
            'nonzero_basis_n_actual_flip':     int(g_actual[nonzero_mask].sum()) if nonzero_mask.any() else 0,
            'nonzero_basis_n_incorrect':       int(g_incorrect[nonzero_mask].sum()) if nonzero_mask.any() else 0,
            'nonzero_basis_n_plain_incorrect': int(g_plain_incorr[nonzero_mask].sum()) if nonzero_mask.any() else 0,
        }
        _print_summary(summary, label, threshold)
        return summary

    def _print_summary(s: dict, label: str, thr: int) -> None:
        n = s['n_shots']
        print(f"\n=== {label} (gap {'≥' if 'Accepted' in label else '<'} {thr}) ===")
        if n == 0:
            print("  (no shots in this group)")
            return
        pct = lambda k: f"{100 * s[k] / n:.1f}%"
        print(f"  shots              : {n}")
        print(f"  actual obs flipped : {s['n_actual_flip']}  ({pct('n_actual_flip')})")
        print(f"  incorrect decoding : {s['n_incorrect']}  ({pct('n_incorrect')})")
        if s['gap_min_incorrect'] is not None:
            print(f"    gap range (incorrect) : [{s['gap_min_incorrect']:.2f}, {s['gap_max_incorrect']:.2f}] dB")
        else:
            print(f"    gap range (incorrect) : n/a")
        print(f"  plain PyM incorrect: {s['n_plain_incorrect']}  ({pct('n_plain_incorrect')})")
        print(f"  basis count range  : [{s['basis_min']}, {s['basis_max']}]")
        n_zero    = n - s['n_nonzero_basis']
        n_nonzero = s['n_nonzero_basis']
        print(f"  zero basis shots   : {n_zero}  ({100 * n_zero / n:.1f}%)")
        print(f"    actual flip      : {s['zero_basis_n_actual_flip']}")
        print(f"    incorrect        : {s['zero_basis_n_incorrect']}")
        print(f"    plain incorrect  : {s['zero_basis_n_plain_incorrect']}")
        print(f"  nonzero basis shots: {n_nonzero}  ({100 * n_nonzero / n:.1f}%)")
        print(f"    actual flip      : {s['nonzero_basis_n_actual_flip']}")
        print(f"    incorrect        : {s['nonzero_basis_n_incorrect']}")
        print(f"    plain incorrect  : {s['nonzero_basis_n_plain_incorrect']}")

    n_kept = len(dets_kept)
    print(f"\n{'='*55}")
    print(f"check_gap_vs_logical_error_nullbasis_count  "
          f"[collect_type={collect_type!r}]")
    print(f"  total shots drawn  : {n_kept + postsel_discards}")
    print(f"  postsel discards   : {postsel_discards}")
    print(f"  kept shots         : {n_kept}")

    accept_summary  = _group_summary(accept_mask,  'Accepted')
    discard_summary = _group_summary(~accept_mask, 'Discarded')
    print()

    return {
        'log_err_nullbasis':           log_err_arr.tolist(),
        'gaps':                        gaps_float.tolist(),
        'accepted':                    accept_mask.tolist(),
        'incorrect_correction':        incorrect.tolist(),
        'plain_incorrect_correction':  plain_incorrect.tolist(),
        'postsel_discards':            postsel_discards,
        'threshold':                   threshold,
        'accept_log_err_nullbasis':    log_err_arr[accept_mask].tolist(),
        'discard_log_err_nullbasis':   log_err_arr[~accept_mask].tolist(),
        'accept_gaps':                 gaps_float[accept_mask].tolist(),
        'discard_gaps':                gaps_float[~accept_mask].tolist(),
        'accept_incorrect':            incorrect[accept_mask].tolist(),
        'discard_incorrect':           incorrect[~accept_mask].tolist(),
        'accept_plain_incorrect':      plain_incorrect[accept_mask].tolist(),
        'discard_plain_incorrect':     plain_incorrect[~accept_mask].tolist(),
        'accept_summary':              accept_summary,
        'discard_summary':             discard_summary,
    }


def check_gap_vs_logical_error_nullbasis_count(
    compiled,
    estimator,
    threshold: int,
    shots: int,
    collect_type: str = 'max',
    clustering_type: str = 'python_original',
    over_grow_step: int = 0,
    bits_per_step: int = 1,
) -> dict:
    """
    Convenience wrapper: draw fresh samples and run analysis in one call.

    For fair comparison across clustering algorithms on the same syndrome
    samples, call sample_gap_shots once and reuse with
    check_gap_vs_logical_error_nullbasis_count_from_samples:

        samples = sample_gap_shots(compiled, estimator, shots)
        r1 = check_gap_vs_logical_error_nullbasis_count_from_samples(
                 samples, estimator, threshold, clustering_type='python_original')
        r2 = check_gap_vs_logical_error_nullbasis_count_from_samples(
                 samples, estimator, threshold,
                 clustering_type='python_overgrow_batch', over_grow_step=2)
    """
    samples = sample_gap_shots(compiled, estimator, shots)
    return check_gap_vs_logical_error_nullbasis_count_from_samples(
        samples, estimator, threshold,
        collect_type=collect_type,
        clustering_type=clustering_type,
        over_grow_step=over_grow_step,
        bits_per_step=bits_per_step,
    )



def check_gap_vs_logical_error_nullbasis_count_plot(
    result: dict,
    axs=None,
    title: str = 'Logical-error null-basis count: accepted vs gap-discarded',
    collect_type: str = '',
):
    """
    Side-by-side bar charts of logical-error null-space basis count distributions
    for accepted and gap-discarded shots.

    Parameters
    ----------
    result : dict
        Output of check_gap_vs_logical_error_nullbasis_count.
    axs : array-like of two Axes, optional
        If None, a new (1×2) figure is created.
    title : str
        Super-title for the figure.
    collect_type : str, optional
        Appended to axis labels (e.g. 'max' or 'sum').

    Returns
    -------
    axs : np.ndarray of matplotlib.axes.Axes
    """
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(12, 4))

    suffix = f' ({collect_type})' if collect_type else ''
    colors = {'Accepted': 'steelblue', 'Gap-discarded': 'tomato'}

    for ax, key, label, color in [
        (axs[0], 'accept_log_err_nullbasis',  'Accepted',      colors['Accepted']),
        (axs[1], 'discard_log_err_nullbasis', 'Gap-discarded', colors['Gap-discarded']),
    ]:
        vals = result[key]
        n_total = len(vals)
        if n_total == 0:
            ax.set_title(f'{label}\n(no shots)')
            ax.set_xlabel(f'Logical-error null-basis count{suffix}')
            ax.set_ylabel('Number of shots')
            continue

        counter = collections.Counter(vals)
        max_k = max(counter.keys())
        xs = list(range(max_k + 1))
        ys = [counter.get(k, 0) for k in xs]
        ax.bar(xs, ys, color=color, edgecolor='white', linewidth=0.5)
        ax.set_xlabel(f'Logical-error null-basis count{suffix}')
        ax.set_ylabel('Number of shots')
        n_nonzero = sum(v for k, v in counter.items() if k > 0)
        ax.set_title(
            f'{label}  (gap {"≥" if key == "accept_log_err_nullbasis" else "<"} '
            f'{result["threshold"]})\n'
            f'n={n_total}  |  log_err_dim > 0 : {n_nonzero} ({100 * n_nonzero / n_total:.1f}%)'
        )
        if max_k <= 30:
            ax.set_xticks(xs)

    axs[0].get_figure().suptitle(title, y=1.02, fontsize=11)
    return axs



def check_complementary_gap_vs_out_gap_analysis_from_samples(
    samples: dict,
    estimator_use,
    threshold: int,
    gap_type: str = 'binary',
    aggregate: str = 'min',
    over_grow_step: int = 0,
    bits_per_step: int = 1,
    decode: bool = False,
    asb: bool = False,
) -> dict:
    """
    Joint analysis comparing the desaturation complementary gap with the
    clustering-based 'out gap' (from PriorGapEstimatorUse.execute).

    Uses pre-sampled shot data from sample_gap_shots, so the same syndrome
    samples can be reused across different gap_type / over_grow_step settings.

    For each shot, records:
      - desaturation complementary gap
      - clustering out gap (gap_type / aggregate / over_grow_step)
      - nonzero logical-error null-basis count (sum across clusters)

    Reports per group (Accepted / Discarded):
      - Overall desaturation gap range and out gap range
      - Desaturation gap range for incorrect shots
      - Per subgroup (zero-basis / nonzero-basis):
          desaturation gap range, out gap range, incorrect counts

    Parameters
    ----------
    samples : dict
        Output of sample_gap_shots.
    estimator_use : PriorGapEstimatorUse
        Provides both PCM dimensions and the clustering execute() method.
    threshold : int
        Accept if desaturation gap >= threshold.
    gap_type : {'binary', 'hamming', 'prior_weight'}
        Clustering gap metric passed to execute().
    aggregate : {'min', 'sum'}
        Cluster aggregation passed to execute().
    over_grow_step : int
        Passed to execute() for overgrow clustering types.
    bits_per_step : int
        Passed to execute() for batch overgrow clustering types.

    Returns
    -------
    dict with keys:
        'log_err_nullbasis'       : list[int]    per-shot nonzero null-basis count.
        'out_gaps'                : list[float]  per-shot clustering gap.
        'desat_gaps'              : list[float]  per-shot desaturation gap.
        'accepted'                : list[bool]
        'incorrect_correction'    : list[bool]
        'plain_incorrect_correction': list[bool]
        'postsel_discards'        : int
        'threshold'               : int
        'accept_summary'          : dict
        'discard_summary'         : dict
        Plus convenience sub-lists split by accept/discard mask.
    """
    dets_kept        = samples['dets_kept']
    gaps_float       = samples['gaps_float']
    actual_flip      = samples['actual_flip']
    incorrect        = samples['incorrect']
    plain_incorrect  = samples['plain_incorrect']
    postsel_discards = samples['postsel_discards']
    n_original_dets  = samples['n_original_dets']
    n_cluster_dets   = samples['n_cluster_dets']

    if len(dets_kept) == 0:
        return {
            'log_err_nullbasis': [], 'out_gaps': [], 'desat_gaps': [],
            'accepted': [], 'incorrect_correction': [],
            'plain_incorrect_correction': [],
            'postsel_discards': postsel_discards, 'threshold': threshold,
            'accept_summary': {}, 'discard_summary': {},
        }

    accept_mask = gaps_float >= threshold

    # ------------------------------------------------------------------ #
    # Per-shot clustering: execute() returns (out_gap, nonzero_count).    #
    # ------------------------------------------------------------------ #
    dets_cluster = np.zeros((len(dets_kept), n_cluster_dets), dtype=np.uint8)
    dets_cluster[:, :n_original_dets] = dets_kept[:, :n_original_dets]

    out_gap_per_shot:     list[float] = []
    log_err_per_shot:     list[int]   = []
    our_incorrect_per_shot: list[bool] = []

    for i in range(len(dets_cluster)):
        syndrome = dets_cluster[i].astype(np.uint8)
        if decode:
            out_gap, our_flip, nonzero_count = estimator_use.execute(
                syndrome,
                gap_type=gap_type, aggregate=aggregate,
                over_grow_step=over_grow_step, bits_per_step=bits_per_step,
                asb=asb, decode=True,
            )
            our_incorrect_per_shot.append(bool(np.any(our_flip)) != bool(actual_flip[i]))
        else:
            out_gap, nonzero_count = estimator_use.execute(
                syndrome,
                gap_type=gap_type, aggregate=aggregate,
                over_grow_step=over_grow_step, bits_per_step=bits_per_step,
                asb=asb,
            )
        out_gap_per_shot.append(out_gap)
        log_err_per_shot.append(nonzero_count)

    out_gaps_arr       = np.array(out_gap_per_shot, dtype=float)
    log_err_arr        = np.array(log_err_per_shot)
    our_incorrect_arr  = np.array(our_incorrect_per_shot, dtype=bool) if decode else None

    # ------------------------------------------------------------------ #
    # Helper: range of finite values in an array, or None if none.        #
    # ------------------------------------------------------------------ #
    def _finite_range(arr):
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return None
        return [float(finite.min()), float(finite.max())]

    def _fmt(r):
        return f"[{r[0]:.3f}, {r[1]:.3f}]" if r is not None else "n/a"

    # ------------------------------------------------------------------ #
    # Per-group summary.                                                   #
    # ------------------------------------------------------------------ #
    def _group_summary(mask: np.ndarray, label: str) -> dict:
        n = int(mask.sum())
        if n == 0:
            s = {
                'n_shots': 0, 'n_actual_flip': 0,
                'n_incorrect': 0, 'n_plain_incorrect': 0, 'n_our_incorrect': 0,
                'desat_gap_range': None, 'desat_gap_range_incorrect': None,
                'incorrect_out_gap_range': None,
                'plain_incorrect_desat_gap_range': None,
                'plain_incorrect_out_gap_range': None,
                'out_gap_range': None,
                'n_nonzero_basis': 0, 'basis_range': None,
                'zero_basis_n': 0,
                'zero_basis_n_incorrect': 0, 'zero_basis_n_plain_incorrect': 0,
                'zero_basis_n_our_incorrect': 0,
                'zero_basis_desat_gap_range': None, 'zero_basis_out_gap_range': None,
                'nonzero_basis_n': 0,
                'nonzero_basis_n_incorrect': 0, 'nonzero_basis_n_plain_incorrect': 0,
                'nonzero_basis_n_our_incorrect': 0,
                'nonzero_basis_desat_gap_range': None, 'nonzero_basis_out_gap_range': None,
                'our_incorrect_desat_gap_range': None, 'our_incorrect_out_gap_range': None,
            }
            _print_summary(s, label, threshold)
            return s

        g_desat        = gaps_float[mask]
        g_out          = out_gaps_arr[mask]
        g_actual       = actual_flip[mask]
        g_incorrect    = incorrect[mask]
        g_plain_incorr = plain_incorrect[mask]
        g_our_incorr   = our_incorrect_arr[mask] if decode else np.zeros(n, dtype=bool)
        g_basis        = log_err_arr[mask]
        zero_mask      = g_basis == 0
        nonzero_mask   = g_basis > 0

        s = {
            'n_shots':                     n,
            'n_actual_flip':               int(g_actual.sum()),
            'n_incorrect':                 int(g_incorrect.sum()),
            'n_plain_incorrect':           int(g_plain_incorr.sum()),
            'n_our_incorrect':             int(g_our_incorr.sum()),
            'desat_gap_range':             [float(g_desat.min()), float(g_desat.max())],
            'desat_gap_range_incorrect':   _finite_range(g_desat[g_incorrect]),
            'incorrect_out_gap_range':     _finite_range(g_out[g_incorrect]),
            'plain_incorrect_desat_gap_range': _finite_range(g_desat[g_plain_incorr]),
            'plain_incorrect_out_gap_range':   _finite_range(g_out[g_plain_incorr]),
            'out_gap_range':               _finite_range(g_out),
            'our_incorrect_desat_gap_range': _finite_range(g_desat[g_our_incorr]) if decode else None,
            'our_incorrect_out_gap_range':   _finite_range(g_out[g_our_incorr])   if decode else None,
            'n_nonzero_basis':             int(nonzero_mask.sum()),
            'basis_range':                 [int(g_basis.min()), int(g_basis.max())],
            # zero-basis subgroup
            'zero_basis_n':                int(zero_mask.sum()),
            'zero_basis_n_incorrect':      int(g_incorrect[zero_mask].sum()),
            'zero_basis_n_plain_incorrect':int(g_plain_incorr[zero_mask].sum()),
            'zero_basis_n_our_incorrect':  int(g_our_incorr[zero_mask].sum()),
            'zero_basis_desat_gap_range':  _finite_range(g_desat[zero_mask]) if zero_mask.any() else None,
            'zero_basis_out_gap_range':    _finite_range(g_out[zero_mask]) if zero_mask.any() else None,
            # nonzero-basis subgroup
            'nonzero_basis_n':                  int(nonzero_mask.sum()),
            'nonzero_basis_n_incorrect':        int(g_incorrect[nonzero_mask].sum()) if nonzero_mask.any() else 0,
            'nonzero_basis_n_plain_incorrect':  int(g_plain_incorr[nonzero_mask].sum()) if nonzero_mask.any() else 0,
            'nonzero_basis_n_our_incorrect':    int(g_our_incorr[nonzero_mask].sum()) if nonzero_mask.any() else 0,
            'nonzero_basis_desat_gap_range':    _finite_range(g_desat[nonzero_mask]) if nonzero_mask.any() else None,
            'nonzero_basis_out_gap_range':      _finite_range(g_out[nonzero_mask]) if nonzero_mask.any() else None,
        }
        _print_summary(s, label, threshold)
        return s

    def _print_summary(s: dict, label: str, thr: int) -> None:
        n = s['n_shots']
        sign = '≥' if 'Accepted' in label else '<'
        print(f"\n=== {label} (complementary gap {sign} {thr}) ===")
        if n == 0:
            print("  (no shots in this group)")
            return
        pct = lambda k: f"{100 * s[k] / n:.1f}%"
        print(f"  shots                    : {n}")
        print(f"  actual obs flipped       : {s['n_actual_flip']}  ({pct('n_actual_flip')})")
        print(f"  incorrect decoding       : {s['n_incorrect']}  ({pct('n_incorrect')})")
        print(f"    desat gap range        : {_fmt(s['desat_gap_range_incorrect'])} dB")
        print(f"    out gap range          : {_fmt(s['incorrect_out_gap_range'])}")
        print(f"  plain PyM incorrect      : {s['n_plain_incorrect']}  ({pct('n_plain_incorrect')})")
        print(f"    desat gap range        : {_fmt(s['plain_incorrect_desat_gap_range'])} dB")
        print(f"    out gap range          : {_fmt(s['plain_incorrect_out_gap_range'])}")
        if decode:
            print(f"  our method incorrect     : {s['n_our_incorrect']}  ({pct('n_our_incorrect')})")
            print(f"    desat gap range        : {_fmt(s['our_incorrect_desat_gap_range'])} dB")
            print(f"    out gap range          : {_fmt(s['our_incorrect_out_gap_range'])}")
        print(f"  desat gap range          : {_fmt(s['desat_gap_range'])} dB")
        print(f"  out gap range (finite)   : {_fmt(s['out_gap_range'])}")
        print(f"  basis count range        : {s['basis_range']}")
        n_zero    = s['zero_basis_n']
        n_nonzero = s['nonzero_basis_n']
        print(f"  zero basis shots         : {n_zero}  ({100 * n_zero / n:.1f}%)")
        print(f"    incorrect              : {s['zero_basis_n_incorrect']}")
        print(f"    plain incorrect        : {s['zero_basis_n_plain_incorrect']}")
        if decode:
            print(f"    our method incorrect   : {s['zero_basis_n_our_incorrect']}")
        print(f"    desat gap range        : {_fmt(s['zero_basis_desat_gap_range'])} dB")
        print(f"    out gap range          : {_fmt(s['zero_basis_out_gap_range'])}")
        print(f"  nonzero basis shots      : {n_nonzero}  ({100 * n_nonzero / n:.1f}%)")
        print(f"    incorrect              : {s['nonzero_basis_n_incorrect']}")
        print(f"    plain incorrect        : {s['nonzero_basis_n_plain_incorrect']}")
        if decode:
            print(f"    our method incorrect   : {s['nonzero_basis_n_our_incorrect']}")
        print(f"    desat gap range        : {_fmt(s['nonzero_basis_desat_gap_range'])} dB")
        print(f"    out gap range          : {_fmt(s['nonzero_basis_out_gap_range'])}")

    n_kept = len(dets_kept)
    print(f"\n{'='*60}")
    print(f"check_complementary_gap_vs_out_gap_analysis")
    print(f"  gap_type={gap_type!r}  aggregate={aggregate!r}  "
          f"over_grow_step={over_grow_step}  bits_per_step={bits_per_step}")
    print(f"  total shots drawn  : {n_kept + postsel_discards}")
    print(f"  postsel discards   : {postsel_discards}")
    print(f"  kept shots         : {n_kept}")

    accept_summary  = _group_summary(accept_mask,  'Accepted')
    discard_summary = _group_summary(~accept_mask, 'Discarded')
    print()

    ret = {
        'log_err_nullbasis':          log_err_arr.tolist(),
        'out_gaps':                   out_gaps_arr.tolist(),
        'desat_gaps':                 gaps_float.tolist(),
        'accepted':                   accept_mask.tolist(),
        'incorrect_correction':       incorrect.tolist(),
        'plain_incorrect_correction': plain_incorrect.tolist(),
        'postsel_discards':           postsel_discards,
        'threshold':                  threshold,
        'accept_log_err_nullbasis':   log_err_arr[accept_mask].tolist(),
        'discard_log_err_nullbasis':  log_err_arr[~accept_mask].tolist(),
        'accept_desat_gaps':          gaps_float[accept_mask].tolist(),
        'discard_desat_gaps':         gaps_float[~accept_mask].tolist(),
        'accept_out_gaps':            out_gaps_arr[accept_mask].tolist(),
        'discard_out_gaps':           out_gaps_arr[~accept_mask].tolist(),
        'accept_incorrect':           incorrect[accept_mask].tolist(),
        'discard_incorrect':          incorrect[~accept_mask].tolist(),
        'accept_summary':             accept_summary,
        'discard_summary':            discard_summary,
    }
    if decode:
        ret['our_incorrect_correction'] = our_incorrect_arr.tolist()
        ret['accept_our_incorrect']     = our_incorrect_arr[accept_mask].tolist()
        ret['discard_our_incorrect']    = our_incorrect_arr[~accept_mask].tolist()
    return ret


def check_pymatching_vs_pge_from_sample(
    samples: dict,
    estimator_use,
    pge_threshold: float,
    comp_gap_threshold: float,
    gap_type: str = 'binary',
    aggregate: str = 'min',
    over_grow_step: int = 0,
    bits_per_step: int = 1,
    asb: bool = False,
    decode: bool = False,
    prior_gap_estimate_type: str = 'cpp',
) -> dict:
    """
    Compare the complementary-gap (desaturation) reject/accept strategy against
    the PriorGapEstimator (PGE) strategy on pre-sampled shot data.

    For each strategy a threshold determines which shots are accepted:
      - Complementary gap : accept if gap >= comp_gap_threshold.
      - PGE gap           : accept if pge_gap >= pge_threshold.

    Rates are always divided by n_kept (shots surviving postselection).

    For the PGE strategy two logical-error rates are reported:
      1. Using plain PyMatching as the decoder (our gap only decides what to keep).
      2. Using PGE's own decoded prediction (only when decode=True).

    Parameters
    ----------
    samples : dict
        Output of sample_gap_shots.
    estimator_use :
        Pre-built PriorGapEstimatorUse (Python) or prior_gap_estimator_cpp
        (C++).  Must support execute() and, for 'cpp', execute_batch().
    pge_threshold : float
        Accept if PGE gap >= this value.
    comp_gap_threshold : float
        Accept if desaturation complementary gap >= this value.
    gap_type : {'binary', 'hamming', 'prior_weight', 'weight_diff'}
    aggregate : {'min', 'sum'}
    over_grow_step, bits_per_step, asb : passed to estimator_use.execute[_batch].
    decode : bool
        If True, also collect PGE's own decoded prediction and report a second
        logical-error rate.  Required when gap_type='weight_diff'.
    prior_gap_estimate_type : {'python', 'cpp'}
        'python' uses execute() in a Python for-loop.
        'cpp'    uses execute_batch() — one C++ call for all shots.

    Returns
    -------
    dict with keys:
        'n_kept', 'postsel_discards'
        'comp_accept', 'comp_reject_rate', 'comp_ler'
        'pge_gaps', 'pge_accept', 'pge_reject_rate'
        'pge_ler_pymatching'
        'pge_ler_pge'         (only when decode=True)
        'pge_flips'           (only when decode=True)
    """
    # ------------------------------------------------------------------ #
    # Extract from samples                                                 #
    # ------------------------------------------------------------------ #
    dets_kept        = samples['dets_kept']          # (n_kept, n_gap_dets)
    obs_kept         = samples['obs_kept']            # (n_kept, n_obs)
    gaps_float       = samples['gaps_float']          # (n_kept,) complementary gap
    incorrect        = samples['incorrect']           # (n_kept,) desat incorrect
    plain_incorrect  = samples['plain_incorrect']     # (n_kept,) PyMatching incorrect
    postsel_discards = samples['postsel_discards']
    n_original_dets  = samples['n_original_dets']
    n_cluster_dets   = samples['n_cluster_dets']

    n_kept = len(dets_kept)

    # ------------------------------------------------------------------ #
    # Helper: print a rate as  m/n=r                                      #
    # ------------------------------------------------------------------ #
    def _rate(m: int, n: int) -> str:
        return f"{m}/{n}={m/n:.6f}" if n > 0 else f"0/0=N/A"

    # ------------------------------------------------------------------ #
    # Complementary gap stats (pure numpy, no loops)                       #
    # ------------------------------------------------------------------ #
    comp_accept      = gaps_float >= comp_gap_threshold
    comp_n_discard   = int((~comp_accept).sum())
    comp_n_incorrect = int(incorrect[comp_accept].sum())
    comp_reject_rate = comp_n_discard / n_kept if n_kept > 0 else float('nan')
    comp_ler         = comp_n_incorrect / n_kept if n_kept > 0 else float('nan')

    # ------------------------------------------------------------------ #
    # Build cluster syndrome matrix                                        #
    # ------------------------------------------------------------------ #
    dets_cluster = np.zeros((n_kept, n_cluster_dets), dtype=np.uint8)
    dets_cluster[:, :n_original_dets] = dets_kept[:, :n_original_dets]

    # ------------------------------------------------------------------ #
    # Run PGE — one batch call (cpp) or per-shot loop (python)            #
    # ------------------------------------------------------------------ #
    if prior_gap_estimate_type == 'cpp':
        pge_gaps, _, pge_flips = estimator_use.execute_batch(
            dets_cluster,
            gap_type=gap_type, aggregate=aggregate,
            over_grow_step=over_grow_step, bits_per_step=bits_per_step,
            asb=asb, decode=decode,
        )
        pge_gaps = np.asarray(pge_gaps, dtype=np.float64)

    else:  # 'python' — per-shot loop
        pge_gaps   = np.empty(n_kept, dtype=np.float64)
        pge_flips  = [] if decode else None
        for i in range(n_kept):
            syndrome = dets_cluster[i]
            if decode:
                gap, flip, _ = estimator_use.execute(
                    syndrome, gap_type=gap_type, aggregate=aggregate,
                    over_grow_step=over_grow_step, bits_per_step=bits_per_step,
                    asb=asb, decode=True,
                )
                pge_flips.append(flip)
            else:
                gap, _ = estimator_use.execute(
                    syndrome, gap_type=gap_type, aggregate=aggregate,
                    over_grow_step=over_grow_step, bits_per_step=bits_per_step,
                    asb=asb, decode=False,
                )
            pge_gaps[i] = gap
        if decode and pge_flips:
            pge_flips = np.array(pge_flips, dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # Baseline: plain PyMatching with no discard                          #
    # ------------------------------------------------------------------ #
    n_plain_incorrect_total = int(plain_incorrect.sum())
    baseline_ler = n_plain_incorrect_total / n_kept if n_kept > 0 else float('nan')

    # ------------------------------------------------------------------ #
    # PGE stats (pure numpy)                                               #
    # ------------------------------------------------------------------ #
    pge_accept      = pge_gaps >= pge_threshold
    pge_n_discard   = int((~pge_accept).sum())
    pge_n_incorrect_pm = int(plain_incorrect[pge_accept].sum())
    pge_reject_rate    = pge_n_discard / n_kept if n_kept > 0 else float('nan')
    pge_ler_pymatching = pge_n_incorrect_pm / n_kept if n_kept > 0 else float('nan')

    pge_ler_pge     = None
    pge_incorrect   = None
    if decode and pge_flips is not None:
        pge_incorrect = np.any(
            pge_flips.astype(bool) ^ obs_kept.astype(bool), axis=1
        )
        pge_n_incorrect_pge = int(pge_incorrect[pge_accept].sum())
        pge_ler_pge = pge_n_incorrect_pge / n_kept if n_kept > 0 else float('nan')

    # How many plain-incorrect shots does each gap method discard?
    # n  = total plain-incorrect shots
    # m1 = plain-incorrect shots discarded by complementary gap
    # m2 = plain-incorrect shots discarded by PGE gap
    n  = n_plain_incorrect_total
    m1 = int(plain_incorrect[~comp_accept].sum())
    m2 = int(plain_incorrect[~pge_accept].sum())

    # ------------------------------------------------------------------ #
    # Print                                                                #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"check_pymatching_vs_pge_from_sample")
    print(f"  gap_type={gap_type!r}  aggregate={aggregate!r}  "
          f"over_grow_step={over_grow_step}  bits_per_step={bits_per_step}")
    print(f"  prior_gap_estimate_type={prior_gap_estimate_type!r}  decode={decode}")
    print(f"  total shots drawn : {n_kept + postsel_discards}")
    print(f"  postsel discards  : {postsel_discards}")
    print(f"  kept shots (n)    : {n_kept}")

    print(f"\n--- Baseline (PyMatching, no discard) ---")
    print(f"  LER : {_rate(n_plain_incorrect_total, n_kept)}")

    print(f"\n--- Complementary gap  (threshold={comp_gap_threshold}) ---")
    print(f"  reject rate : {_rate(comp_n_discard,   n_kept)}")
    print(f"  LER (desat) : {_rate(comp_n_incorrect, n_kept)}")

    print(f"\n--- PGE gap  (threshold={pge_threshold}) ---")
    print(f"  reject rate           : {_rate(pge_n_discard,        n_kept)}")
    print(f"  LER (PyMatching dec.) : {_rate(pge_n_incorrect_pm,   n_kept)}")
    if decode and pge_ler_pge is not None:
        print(f"  LER (PGE dec.)        : {_rate(pge_n_incorrect_pge, n_kept)}")

    print(f"\n--- Incorrect-shot discard  (n={n} plain-incorrect shots) ---")
    print(f"  comp gap discards incorrect : m1/n = {_rate(m1, n)}")
    print(f"  PGE gap discards incorrect  : m2/n = {_rate(m2, n)}")
    print(f"  PGE vs comp gap             : m2/m1 = {_rate(m2, m1)}")

    # ------------------------------------------------------------------ #
    # Return                                                               #
    # ------------------------------------------------------------------ #
    ret = {
        'n_kept':                    n_kept,
        'postsel_discards':          postsel_discards,
        'baseline_ler':              baseline_ler,
        'n_plain_incorrect_total':   n_plain_incorrect_total,
        'comp_accept':               comp_accept,
        'comp_reject_rate':          comp_reject_rate,
        'comp_ler':                  comp_ler,
        'm1':                        m1,
        'pge_gaps':                  pge_gaps,
        'pge_accept':                pge_accept,
        'pge_reject_rate':           pge_reject_rate,
        'pge_ler_pymatching':        pge_ler_pymatching,
        'm2':                        m2,
    }
    if decode:
        ret['pge_flips']   = pge_flips
        ret['pge_ler_pge'] = pge_ler_pge
    return ret
