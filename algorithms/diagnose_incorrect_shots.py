"""
Diagnostic script: for incorrect + zero-null-basis shots (discarded group),
examine the exact error pattern, clustering structure, and why the algorithm
cannot detect the fault.

Run from the algorithms/ directory.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path('..') / 'src'))
sys.path.insert(0, str(Path('..') / 'explorations'))

import numpy as np
import sinter
import stim

from generate_circuit_with_stage_map import CircuitGenerator
from cultiv._decoding._desaturation_sampler import CompiledDesaturationSampler
from prior_gap_estimation import PriorGapEstimator
from clustering import Clustering

# ── same parameters as the notebook ──────────────────────────────────────────
circuit_type       = 'end2end-inplace-distillation'
injection_protocol = 'unitary'
noise_model        = 'uniform-depolarizing'
noise_strength     = 1e-3
gateset            = 'css'
basis              = 'Y'
d1, d2             = 3, 15
r_in_escape        = 3
r_post_escape      = 0

cg = CircuitGenerator.from_input_params(
    basis=basis, gateset=gateset, circuit_type=circuit_type,
    noise_model=noise_model, noise_strength=noise_strength,
    injection_protocol=injection_protocol,
    r_in_escape=r_in_escape, r_post_escape=r_post_escape,
    d1=d1, d2=d2, HasNoise=True,
)
cg.generate()
raw_dem = cg.noisy_circuit.detector_error_model()

task     = sinter.Task(circuit=cg.noisy_circuit, detector_error_model=raw_dem)
compiled = CompiledDesaturationSampler.from_task(task)

estimator = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type='per_obs')
pcm       = estimator.PCM

n_logical      = pcm.n_logical_check_nodes
n_gap_dets     = compiled.gap_circuit.num_detectors
n_original_det = n_gap_dets - 1
n_cluster_dets = pcm.H.shape[0]

print(f'PCM shape           : {pcm.H.shape}')
print(f'n_logical_check_nodes: {n_logical}')
print(f'n_original detectors : {n_original_det}')
gap_threshold = 67

# ── sample shots (fixed seed for reproducibility) ────────────────────────────
SHOTS   = 50_000
SEED    = 42
rng     = np.random.default_rng(SEED)
stim_seed = int(rng.integers(0, 2**32))

sampler = compiled.gap_circuit.compile_detector_sampler(seed=stim_seed)
dets_all, obs_all = sampler.sample(SHOTS, separate_observables=True, bit_packed=False)

# post-selection
keep = np.ones(SHOTS, dtype=bool)
for d in compiled.postselected_detectors:
    keep &= ~dets_all[:, d].astype(bool)
dets_kept = dets_all[keep]
obs_kept  = obs_all[keep]
print(f'\nAfter post-sel: {len(dets_kept)} / {SHOTS} shots kept')

# desaturation decode
dets_packed  = np.packbits(dets_kept, bitorder='little', axis=1).copy()
predictions, gaps = compiled._decode_batch_overwrite_last_byte(dets_packed)
actual_flip  = obs_kept.any(axis=1)
incorrect    = predictions.astype(bool) ^ actual_flip
accept_mask  = gaps >= gap_threshold

print(f'Accepted   : {accept_mask.sum()}  |  incorrect accepted   : {(incorrect & accept_mask).sum()}')
print(f'Discarded  : {(~accept_mask).sum()}  |  incorrect discarded  : {(incorrect & ~accept_mask).sum()}')

# ── clustering on discarded+incorrect shots ───────────────────────────────────
target_mask = incorrect & ~accept_mask   # discarded AND incorrect
target_idx  = np.where(target_mask)[0]
print(f'\nTarget shots (discarded+incorrect): {len(target_idx)}')

# helper: extract the RREF correction (minimum-weight solution)
def get_correction(cl):
    """Return local correction vector (length = n_bits) from a cluster's RREF."""
    corr = np.zeros(cl.rref.n_bits, dtype=np.uint8)
    for i, pm in enumerate(cl.rref.pivot_map):
        if pm is not None and cl.rref.s_prime[i] == 1:
            corr[pm] = 1
    return corr

# helper: map local correction back to global PCM fault indices
def global_faults_in_correction(cl):
    corr = get_correction(cl)
    return [cl.cluster_fault_idx_to_pcm_fault_idx[j]
            for j in range(len(corr)) if corr[j] == 1]

# helper: faults on the cluster frontier that were NOT absorbed
def frontier_faults(cl, global_fault_membership, weights):
    """
    Faults reachable from the cluster (in dist) but not yet absorbed.
    Returns list of (virtual_weight, global_fault_idx, obs_mask) sorted by weight.
    """
    out = []
    for j, vw in cl.dist.items():
        if global_fault_membership[j] is not cl:
            obs_mask = pcm.L[:, j] if pcm.L is not None else None
            flips_logical = bool(np.any(obs_mask)) if obs_mask is not None else False
            out.append((vw, j, flips_logical))
    out.sort()
    return out

dets_cluster = np.zeros((len(dets_kept), n_cluster_dets), dtype=np.uint8)
dets_cluster[:, :n_original_det] = dets_kept[:, :n_original_det]

clustering = Clustering(pcm)

MAX_SHOW = 10  # show at most this many shots in detail

n_zero_basis_incorrect = 0
n_shown = 0

for shot_i in target_idx:
    syndrome = dets_cluster[shot_i].astype(np.uint8)
    clustering.run(syndrome)

    # collect null-basis counts
    regions   = clustering.create_degenerate_cycle_regions()
    log_basis = sum(len(v['logical_error']) for v in regions.values())

    if log_basis != 0:
        continue   # only care about zero-null-basis incorrect shots

    n_zero_basis_incorrect += 1
    if n_shown >= MAX_SHOW:
        continue
    n_shown += 1

    gap_val      = gaps[shot_i]
    actual       = bool(actual_flip[shot_i])
    pred         = bool(predictions[shot_i])
    n_det_fired  = int(syndrome[:n_original_det].sum())

    print(f'\n{"="*70}')
    print(f'Shot {shot_i}  |  gap={gap_val:.2f} dB  |  actual_flip={actual}  |  predicted={pred}')
    print(f'  Detectors fired (orig): {n_det_fired}')
    print(f'  Clusters formed       : {len(clustering.active_valid_clusters)}')

    # global observable prediction from union of all cluster corrections
    global_corr_obs = np.zeros(pcm.L.shape[0], dtype=np.uint8)

    for cl_id, cl in clustering.active_valid_clusters.items():
        corr_local   = get_correction(cl)
        corr_faults  = global_faults_in_correction(cl)
        logical_eff  = (cl.L @ corr_local % 2) if cl.L is not None else None
        global_corr_obs ^= logical_eff if logical_eff is not None else 0

        # which absorbed faults flip an observable?
        absorbed_logical = [j for j in cl.fault_nodes
                            if np.any(pcm.L[:, j])]
        # frontier faults
        front = frontier_faults(cl, clustering.global_fault_membership, clustering.weights)
        front_logical = [(vw, j) for vw, j, fl in front if fl]

        n_checks_fired = sum(1 for c in cl.check_nodes
                             if c < n_original_det and syndrome[c])

        print(f'\n  Cluster {cl_id}:')
        print(f'    check_nodes       : {len(cl.check_nodes)}  ({n_checks_fired} fired)')
        print(f'    fault_nodes       : {len(cl.fault_nodes)}')
        print(f'    null_basis (log)  : {len(regions.get(cl_id, {}).get("logical_error", []))}')
        print(f'    null_basis (stab) : {len(regions.get(cl_id, {}).get("stabilizer", []))}')
        print(f'    correction faults : {len(corr_faults)}')
        print(f'    logical effect    : {logical_eff}')
        print(f'    absorbed logical-flip faults: {len(absorbed_logical)}')

        # Show the faults in the correction and whether each flips an observable
        if corr_faults:
            print(f'    Correction fault details:')
            for j in corr_faults[:10]:
                ed     = pcm.error_data[j]
                l_row  = pcm.L[:, j]
                print(f'      fault {j:5d}  p={ed["prob"]:.2e}  '
                      f'dets={sorted(ed["detectors"])}  obs_flip={l_row.tolist()}')

        # Show closest frontier faults that flip an observable
        if front_logical:
            print(f'    Nearest unabsorbed logical-flip faults (frontier):')
            for vw, j in front_logical[:5]:
                ed    = pcm.error_data[j]
                l_row = pcm.L[:, j]
                print(f'      fault {j:5d}  virt_w={vw:.3f}  p={ed["prob"]:.2e}  '
                      f'dets={sorted(ed["detectors"])}  obs_flip={l_row.tolist()}')
        else:
            print(f'    No unabsorbed logical-flip faults in frontier')

    print(f'\n  GLOBAL correction obs effect: {global_corr_obs.tolist()}')
    print(f'  ACTUAL observable flip      : {obs_kept[shot_i].tolist()}')

print(f'\n{"="*70}')
print(f'Zero-null-basis incorrect shots found: {n_zero_basis_incorrect} / {len(target_idx)} total incorrect discarded')
print(f'(showed first {min(n_shown, MAX_SHOW)})')
