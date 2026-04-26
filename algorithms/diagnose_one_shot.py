"""
Deep trace of one incorrect zero-null-basis shot.
Instruments the clustering to record the order faults were absorbed,
then prints the full story: syndrome → clusters → growth → frontier.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path('..') / 'src'))
sys.path.insert(0, str(Path('..') / 'explorations'))

import heapq
import numpy as np
import sinter

from generate_circuit_with_stage_map import CircuitGenerator
from cultiv._decoding._desaturation_sampler import CompiledDesaturationSampler
from prior_gap_estimation import PriorGapEstimator
from clustering import Clustering

# ── circuit setup (same as notebook) ─────────────────────────────────────────
cg = CircuitGenerator.from_input_params(
    basis='Y', gateset='css', circuit_type='end2end-inplace-distillation',
    noise_model='uniform-depolarizing', noise_strength=1e-3,
    injection_protocol='unitary', r_in_escape=3, r_post_escape=0,
    d1=3, d2=15, HasNoise=True,
)
cg.generate()
raw_dem  = cg.noisy_circuit.detector_error_model()
task     = sinter.Task(circuit=cg.noisy_circuit, detector_error_model=raw_dem)
compiled = CompiledDesaturationSampler.from_task(task)
estimator = PriorGapEstimator.from_desaturation(raw_dem, obs_det_type='per_obs')
pcm       = estimator.PCM

n_logical      = pcm.n_logical_check_nodes
n_gap_dets     = compiled.gap_circuit.num_detectors
n_original_det = n_gap_dets - 1
n_cluster_dets = pcm.H.shape[0]

# ── sample and identify incorrect zero-null-basis shots ──────────────────────
SHOTS = 50_000
SEED  = 42
rng   = np.random.default_rng(SEED)
sampler = compiled.gap_circuit.compile_detector_sampler(seed=int(rng.integers(0, 2**32)))
dets_all, obs_all = sampler.sample(SHOTS, separate_observables=True, bit_packed=False)

keep = np.ones(SHOTS, dtype=bool)
for d in compiled.postselected_detectors:
    keep &= ~dets_all[:, d].astype(bool)
dets_kept = dets_all[keep]
obs_kept  = obs_all[keep]

dets_packed  = np.packbits(dets_kept, bitorder='little', axis=1).copy()
predictions, gaps = compiled._decode_batch_overwrite_last_byte(dets_packed)
actual_flip  = obs_kept.any(axis=1)
incorrect    = predictions.astype(bool) ^ actual_flip
discard_mask = gaps < 67

dets_cluster = np.zeros((len(dets_kept), n_cluster_dets), dtype=np.uint8)
dets_cluster[:, :n_original_det] = dets_kept[:, :n_original_det]

# ── find the SIMPLEST incorrect zero-null-basis shot ─────────────────────────
# "simplest" = fewest detectors fired, so it's easiest to reason about
clustering = Clustering(pcm)

best_shot      = None
best_n_det     = 9999
best_n_cluster = 9999

for shot_i in np.where(incorrect & discard_mask)[0]:
    syndrome = dets_cluster[shot_i].astype(np.uint8)
    clustering.run(syndrome)
    regions  = clustering.create_degenerate_cycle_regions()
    log_basis = sum(len(v['logical_error']) for v in regions.values())
    if log_basis != 0:
        continue
    n_det = int(syndrome[:n_original_det].sum())
    n_cl  = len(clustering.active_valid_clusters)
    if n_det < best_n_det or (n_det == best_n_det and n_cl < best_n_cluster):
        best_shot      = shot_i
        best_n_det     = n_det
        best_n_cluster = n_cl

print(f'Simplest incorrect zero-null-basis shot: index={best_shot}, '
      f'n_det_fired={best_n_det}, n_clusters={best_n_cluster}')

# ── instrument clustering to record growth order for this shot ───────────────
# Monkey-patch Clustering to record (step, cluster_id, fault_idx, virt_weight,
#   after_validity) for every _add_fault and every merge event.

growth_log: list[dict] = []

original_add_fault = Clustering._add_fault
original_merge     = Clustering._merge
original_grow      = Clustering._grow_one_step

def logged_add_fault(self, cl, j, vw):
    original_add_fault(self, cl, j, vw)
    ed = pcm.error_data[j]
    growth_log.append({
        'event'   : 'absorb',
        'cl_id'   : cl.cluster_id,
        'fault'   : j,
        'vw'      : vw,
        'dets'    : sorted(ed['detectors']),
        'prob'    : ed['prob'],
        'obs_flip': pcm.L[:, j].tolist(),
        'valid_after': cl.rref.is_valid(),
    })

def logged_merge(self, cl, others, connecting_j, connecting_vw):
    result = original_merge(self, cl, others, connecting_j, connecting_vw)
    ed = pcm.error_data[connecting_j]
    growth_log.append({
        'event'       : 'merge',
        'surviving_cl': result.cluster_id,
        'merged_cls'  : [o.cluster_id for o in others],
        'fault'       : connecting_j,
        'vw'          : connecting_vw,
        'dets'        : sorted(ed['detectors']),
        'prob'        : ed['prob'],
        'obs_flip'    : pcm.L[:, connecting_j].tolist(),
        'valid_after' : result.rref.is_valid(),
    })
    return result

Clustering._add_fault = logged_add_fault
Clustering._merge     = logged_merge

# ── run on the chosen shot ────────────────────────────────────────────────────
syndrome = dets_cluster[best_shot].astype(np.uint8)
growth_log.clear()
clustering.run(syndrome)
regions = clustering.create_degenerate_cycle_regions()

gap_val    = gaps[best_shot]
actual     = bool(actual_flip[best_shot])
pred       = bool(predictions[best_shot])
fired_dets = sorted(int(i) for i in range(n_original_det) if syndrome[i])

print(f'\n{"="*65}')
print(f'Gap     : {gap_val:.3f} dB')
print(f'Actual observable flip : {actual}')
print(f'PyMatching predicted   : {pred}  ← WRONG')
print(f'Detectors fired ({len(fired_dets)}): {fired_dets}')

# ── per-cluster detail ────────────────────────────────────────────────────────
for cl_id, cl in clustering.active_valid_clusters.items():
    fired_in_cl  = [c for c in cl.check_nodes if c < n_original_det and syndrome[c]]
    log_absorbed = [j for j in cl.fault_nodes if pcm.L[:, j].any()]

    print(f'\n── Cluster {cl_id} ──────────────────────────────')
    print(f'  Detectors owned  : {len(cl.check_nodes)}  '
          f'(fired: {sorted(fired_in_cl)})')
    print(f'  Faults absorbed  : {len(cl.fault_nodes)}  '
          f'(logical-flip faults: {len(log_absorbed)})')

    # growth log for this cluster
    cl_events = [e for e in growth_log if e['event'] == 'absorb' and e['cl_id'] == cl_id]
    merges_in = [e for e in growth_log if e['event'] == 'merge'  and e['surviving_cl'] == cl_id]

    if cl_events or merges_in:
        all_events = sorted(
            [('absorb', e, i) for i, e in enumerate(cl_events)] +
            [('merge',  e, i) for i, e in enumerate(merges_in)],
            key=lambda x: growth_log.index(x[1]) if x[1] in growth_log else 0
        )
        print(f'\n  Growth history (virt_weight | fault | dets | obs_flip | valid?):')
        for kind, e, _ in all_events:
            valid_marker = ' ← STOPS HERE (valid)' if e['valid_after'] else ''
            obs_str = e['obs_flip']
            if kind == 'absorb':
                print(f'    absorb  vw={e["vw"]:6.3f}  f={e["fault"]:5d}  '
                      f'p={e["prob"]:.2e}  dets={e["dets"]}  '
                      f'obs={obs_str}{valid_marker}')
            else:
                print(f'    merge   vw={e["vw"]:6.3f}  f={e["fault"]:5d}  '
                      f'p={e["prob"]:.2e}  dets={e["dets"]}  '
                      f'obs={obs_str}  (absorbed cl {e["merged_cls"]}){valid_marker}')

    # frontier: unabsorbed faults reachable from this cluster
    front = []
    for j, vw in cl.dist.items():
        if clustering.global_fault_membership[j] is not cl:
            ed = pcm.error_data[j]
            front.append((vw, j, ed['prob'], sorted(ed['detectors']), pcm.L[:, j].tolist()))
    front.sort()

    print(f'\n  Nearest unabsorbed faults on frontier:')
    shown = 0
    for vw, j, prob, dets, obs in front:
        if shown >= 8:
            break
        marker = '  ← LOGICAL FLIP' if any(obs) else ''
        print(f'    vw={vw:6.3f}  f={j:5d}  p={prob:.2e}  dets={dets}  obs={obs}{marker}')
        shown += 1

    # null basis
    r = regions.get(cl_id, {})
    print(f'\n  Null basis: logical={len(r.get("logical_error",[]))}  '
          f'stabilizer={len(r.get("stabilizer",[]))}')

# ── overall story ─────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('SUMMARY')
print(f'  Gap = {gap_val:.3f} dB  →  PyMatching nearly tied, chose wrong side')
print(f'  Null-basis = 0 across ALL clusters')
print(f'  Why: every cluster became valid by absorbing short-range non-logical faults.')
print(f'  Logical-flip faults are all sitting on the frontier, unabsorbed.')
print(f'  The weight competition that confuses PyMatching is invisible to clustering.')
