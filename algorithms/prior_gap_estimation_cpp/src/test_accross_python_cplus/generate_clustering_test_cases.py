#!/home/bobxu_0712/miniforge3/envs/basic_quantum/bin/python3
"""
Generate clustering cross-language test cases.

Runs the Python Clustering class on small graphs and writes inputs +
expected outputs to clustering_test_cases.txt for the C++ validator.

Includes all test cases from algorithms/test_clustering.py that exercise
run() behaviour, ported to the text-file format for C++ replay.

Usage (from src/):
    python3 test_accross_python_cplus/generate_clustering_test_cases.py
"""

import os
import sys
import numpy as np

_here    = os.path.dirname(os.path.abspath(__file__))
_alg_dir = os.path.abspath(os.path.join(_here, '..', '..', '..'))
sys.path.insert(0, _alg_dir)

import stim
from clustering import Clustering
from parity_matrix_construct import ParityCheckMatrices


# ---------------------------------------------------------------------------
# PCM stubs
# ---------------------------------------------------------------------------

class StubPCM:
    """Build PCM from explicit fault_to_checks lists (ordered lists for detectors)."""
    def __init__(self, n_det, fault_to_checks, probs):
        n_fault = len(fault_to_checks)
        self.H = np.zeros((n_det, n_fault), dtype=np.uint8)
        for j, checks in enumerate(fault_to_checks):
            for i in checks:
                self.H[i, j] = 1
        self.error_data = [
            {'detectors': list(fault_to_checks[j]), 'prob': float(probs[j])}
            for j in range(n_fault)
        ]
        self.L = None


class PCMFromH:
    """
    Build PCM from an H matrix — identical to _PCM in test_clustering.py.
    Uses frozenset for 'detectors' (matching the original test fixtures exactly).
    """
    def __init__(self, H, probs):
        self.H = np.array(H, dtype=np.uint8)
        self.error_data = [
            {'detectors': frozenset(int(i) for i in np.where(self.H[:, j])[0]),
             'prob': float(probs[j])}
            for j in range(self.H.shape[1])
        ]
        self.L = None


def syndrome_from_faults(H, fired):
    """H @ e mod 2  for the given fired fault indices — always a feasible syndrome."""
    s = np.zeros(H.shape[0], dtype=np.uint8)
    for j in fired:
        s ^= H[:, j]
    return s


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_adj(f, keyword, adj_list):
    """Write adjacency as 'KEYWORD count idx0 idx1 ...' (count-prefixed)."""
    for neighbors in adj_list:
        neighbors = list(neighbors)   # frozenset-safe
        f.write(f'{keyword} {len(neighbors)} {" ".join(str(x) for x in neighbors)}\n')


def write_test(f, name, pcm, syndrome_list):
    """Run the Python Clustering engine and write one TEST block."""
    syndrome = np.array(syndrome_list, dtype=np.uint8)
    engine   = Clustering(pcm)
    engine.run(syndrome)

    n_det   = int(pcm.H.shape[0])
    n_fault = int(pcm.H.shape[1])

    # Use engine's adjacency directly — this is exactly what Python used,
    # regardless of whether 'detectors' was a frozenset or a list.
    check_to_faults = engine.check_to_faults
    fault_to_checks = engine.fault_to_checks
    weights         = engine.weights

    f.write(f'TEST {name}\n')
    f.write(f'NDET {n_det}\n')
    f.write(f'NFAULT {n_fault}\n')
    _write_adj(f, 'CTF', check_to_faults)
    _write_adj(f, 'FTC', fault_to_checks)
    weights_str = ' '.join(f'{w:.17g}' for w in weights)
    f.write(f'WEIGHTS {weights_str}\n')
    f.write(f'SYNDROME {" ".join(str(s) for s in syndrome_list)}\n')

    avc = engine.active_valid_clusters
    f.write(f'NACTIVE_VALID {len(avc)}\n')

    for cl_id in sorted(avc.keys()):
        cl   = avc[cl_id]
        rref = cl.rref
        fault_ord = cl.cluster_fault_idx_to_pcm_fault_idx
        check_ord = cl.cluster_check_idx_to_pcm_check_idx
        pm  = [int(p) if p is not None else -1 for p in rref.pivot_map]
        sp  = [int(x) for x in rref.s_prime]

        f.write(f'CLUSTER {cl_id}\n')
        f.write(f'FAULT_NODES {len(cl.fault_nodes)} {" ".join(str(x) for x in sorted(cl.fault_nodes))}\n')
        f.write(f'CHECK_NODES {len(cl.check_nodes)} {" ".join(str(x) for x in sorted(cl.check_nodes))}\n')
        f.write(f'FAULT_ORDER {len(fault_ord)} {" ".join(str(x) for x in fault_ord)}\n')
        f.write(f'CHECK_ORDER {len(check_ord)} {" ".join(str(x) for x in check_ord)}\n')
        f.write(f'NCHECKS {rref.n_checks}\n')
        f.write(f'NBITS {rref.n_bits}\n')
        f.write(f'PIVOT_MAP {" ".join(str(x) for x in pm)}\n')
        f.write(f'SPRIME {" ".join(str(x) for x in sp)}\n')
        f.write(f'ZCOUNT {len(rref.Z)}\n')
        for z in rref.Z:
            f.write(f'Z {" ".join(str(int(x)) for x in z)}\n')
        f.write(f'ISVALID {1 if rref.is_valid() else 0}\n')
        f.write(f'ENDCLUSTER\n')

    f.write(f'ENDTEST\n\n')


# ---------------------------------------------------------------------------
# Graph family helpers
# ---------------------------------------------------------------------------

def gen_chain(f, n_faults, syndrome_list, name=None):
    """Open chain: n_faults faults, n_faults-1 checks."""
    n_checks = n_faults - 1
    ftc = []
    for j in range(n_faults):
        checks = []
        if j > 0:        checks.append(j - 1)
        if j < n_checks: checks.append(j)
        ftc.append(checks)
    tag = name or f'n{n_faults}_s{"".join(str(s) for s in syndrome_list)}'
    write_test(f, f'chain_{tag}', StubPCM(n_checks, ftc, [0.1]*n_faults), syndrome_list)


def gen_cycle(f, n, syndrome_list, name=None):
    """Cycle (ring code): n faults, n checks. Even-weight syndromes only."""
    assert sum(syndrome_list) % 2 == 0, f'Infeasible cycle syndrome: {syndrome_list}'
    ftc = [[j, (j - 1) % n] for j in range(n)]
    tag = name or f'n{n}_s{"".join(str(s) for s in syndrome_list)}'
    write_test(f, f'cycle_{tag}', StubPCM(n, ftc, [0.1]*n), syndrome_list)


def gen_from_H(f, name, H, probs, syndrome_list):
    """Generate from H matrix + probs — identical to _PCM in test_clustering.py."""
    write_test(f, name, PCMFromH(H, probs), syndrome_list)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    out_path = os.path.join(_here, 'clustering_test_cases.txt')

    with open(out_path, 'w') as f:
        f.write('# Clustering cross-language test cases\n')
        f.write('# Generated by generate_clustering_test_cases.py\n\n')

        # ================================================================
        # Section 1 — original systematic test suite
        # ================================================================

        # ---- Chain (repetition code) ----
        gen_chain(f, 3, [1, 1])
        gen_chain(f, 3, [1, 0])
        gen_chain(f, 3, [0, 1])
        gen_chain(f, 3, [0, 0])
        gen_chain(f, 4, [1, 0, 1])
        gen_chain(f, 4, [1, 1, 0])
        gen_chain(f, 4, [1, 1, 1])
        gen_chain(f, 5, [1, 0, 0, 1])
        gen_chain(f, 5, [1, 1, 1, 1])

        # ---- Cycle (ring code, even-weight syndromes only) ----
        gen_cycle(f, 3, [0, 0, 0])
        gen_cycle(f, 3, [1, 1, 0])
        gen_cycle(f, 3, [1, 0, 1])
        gen_cycle(f, 3, [0, 1, 1])
        gen_cycle(f, 4, [1, 0, 1, 0])
        gen_cycle(f, 4, [1, 1, 0, 0])
        gen_cycle(f, 4, [1, 1, 1, 1])

        # ---- Custom / original ----
        write_test(f, 'disconnected_two_chains',
            StubPCM(4, [[0],[0,1],[1],[2],[2,3],[3]], [0.1]*6), [1,1,1,1])
        write_test(f, 'star_1check_4faults',
            StubPCM(1, [[0],[0],[0],[0]], [0.1,0.2,0.3,0.05]), [1])
        write_test(f, 'asymmetric_weights',
            StubPCM(3, [[0],[0,1],[1,2],[2]], [0.3,0.01,0.4,0.1]), [1,0,1])

        # ---- Random (feasible syndromes from H @ error % 2) ----
        rng = np.random.default_rng(42)
        for trial in range(10):
            n_det   = int(rng.integers(3, 7))
            n_fault = int(rng.integers(n_det, n_det + 4))
            ftc = []
            for _ in range(n_fault):
                k      = int(rng.integers(1, 3))
                checks = list(map(int, rng.choice(n_det, size=k, replace=False)))
                ftc.append(checks)
            probs   = list(rng.uniform(0.05, 0.45, size=n_fault))
            pcm_tmp = StubPCM(n_det, ftc, probs)
            error   = rng.integers(0, 2, size=n_fault).astype(np.uint8)
            syndrome = list(map(int, (pcm_tmp.H @ error) % 2))
            write_test(f, f'random_{trial:02d}_d{n_det}_f{n_fault}',
                       StubPCM(n_det, ftc, probs), syndrome)

        # ================================================================
        # Section 2 — ported from algorithms/test_clustering.py
        # ================================================================

        # ---- Layer 1a: _init_each_cluster ----

        # test_check_nodes_and_enclosed_syndromes / test_rref_empty_at_init
        # rep(5): 5 faults, 4 checks; seed only at check 2
        H_rep5, p5 = (lambda n: (np.array([[1 if abs(i-j)<=0 and (i==j or i==j+1) else 0
                for j in range(n-1)] for i in range(n-1)], dtype=np.uint8), [0.1]*n))(5)
        # Build rep(5) H properly
        H_rep5 = np.zeros((4, 5), dtype=np.uint8)
        for i in range(4): H_rep5[i, i] = H_rep5[i, i+1] = 1
        gen_from_H(f, 'layer1a_rep5_seed2', H_rep5, [0.1]*5, [0,0,1,0])

        # test_multiple_seeds_independent: seeds at check 0 and check 3
        gen_from_H(f, 'layer1a_rep5_seeds0_3', H_rep5, [0.1]*5, [1,0,0,1])

        # ---- Layer 1b: _add_fault ----

        # test_seed_enters_rref_on_first_fault: rep(5), seed at check 2
        # (same as above — the full run output captures the correct state)
        gen_from_H(f, 'layer1b_rep5_seed2_grow', H_rep5, [0.1]*5, [0,0,1,0])

        # test_rref_syndrome_set_correctly: rep(5), syndrome[2]=1, syndrome[1]=0
        gen_from_H(f, 'layer1b_rep5_rref_syndrome', H_rep5, [0.1]*5, [0,0,1,0])

        # ---- Layer 1c: _grow_one_step ----

        # test_minimum_weight_fault_chosen:
        # f0: det0–det1 (p=0.4, weight≈0.41); f1: det0–det2 (p=0.1, weight≈2.20)
        # seeds at det0 and det1; f0 should be chosen first
        H_min = np.zeros((3, 2), dtype=np.uint8)
        H_min[0, 0] = H_min[1, 0] = 1   # f0: det0–det1
        H_min[0, 1] = H_min[2, 1] = 1   # f1: det0–det2
        gen_from_H(f, 'layer1c_min_weight_chosen', H_min, [0.4, 0.1], [1,1,0])

        # test_free_fault_absorbed: rep(5), seeds at det0 and det3
        gen_from_H(f, 'layer1c_free_fault_absorbed', H_rep5, [0.1]*5, [1,0,0,1])

        # test_collision_triggers_merge: one fault connecting two seeds
        H_col = np.zeros((2, 1), dtype=np.uint8)
        H_col[0, 0] = H_col[1, 0] = 1
        gen_from_H(f, 'layer1c_collision_merge', H_col, [0.3], [1,1])

        # test_stale_entry_skipped: rep(5), syndrome [1,0,0,1]
        gen_from_H(f, 'layer1c_stale_entry', H_rep5, [0.1]*5, [1,0,0,1])

        # ---- Layer 1d: _merge ----

        # test_larger_survives + test_unified_check_map_complete:
        # chain 4 checks, 3 faults: det0-f0-det1-f1-det2-f2-det3
        H_ch4 = np.zeros((4, 3), dtype=np.uint8)
        H_ch4[0,0] = H_ch4[1,0] = 1
        H_ch4[1,1] = H_ch4[2,1] = 1
        H_ch4[2,2] = H_ch4[3,2] = 1
        gen_from_H(f, 'layer1d_larger_survives', H_ch4, [0.4,0.4,0.4], [1,0,0,1])

        # test_connecting_j_in_fault_nodes: one fault connecting two seeds
        gen_from_H(f, 'layer1d_connecting_j', H_col, [0.3], [1,1])

        # test_three_way_merge: 3-body hyperedge
        H_3way = np.zeros((3, 1), dtype=np.uint8)
        H_3way[0,0] = H_3way[1,0] = H_3way[2,0] = 1
        gen_from_H(f, 'layer1d_three_way_merge', H_3way, [0.1], [1,1,1])

        # test_connecting_fault_with_unclaimed_check:
        # f0: det0–det1 (high prob p=0.45), f1: det0–det2–det1 (lower p=0.1)
        H_unc = np.zeros((3, 2), dtype=np.uint8)
        H_unc[0, 0] = H_unc[1, 0] = 1            # f0: det0–det1
        H_unc[0, 1] = H_unc[2, 1] = H_unc[1, 1] = 1  # f1: det0–det2–det1
        s_unc = list(map(int, syndrome_from_faults(H_unc, [1])))  # [1,1,1]
        gen_from_H(f, 'layer1d_unclaimed_check', H_unc, [0.45, 0.1], s_unc)

        # test_heap_union_allows_further_growth:
        # chain 5 dets, 4 faults; seeds at det0 and det4
        H_ch5 = np.zeros((5, 4), dtype=np.uint8)
        H_ch5[0,0] = H_ch5[1,0] = 1
        H_ch5[1,1] = H_ch5[2,1] = 1
        H_ch5[2,2] = H_ch5[3,2] = 1
        H_ch5[3,3] = H_ch5[4,3] = 1
        gen_from_H(f, 'layer1d_heap_union_growth', H_ch5, [0.4]*4, [1,0,0,0,1])

        # ---- Layer 3: exhaustive correctness ----

        # rep code n=3..6, all 2^(n-1) syndromes
        for n in range(3, 7):
            H_rep = np.zeros((n-1, n), dtype=np.uint8)
            for i in range(n-1): H_rep[i, i] = H_rep[i, i+1] = 1
            m = n - 1
            for bits in range(2**m):
                s = [(bits >> i) & 1 for i in range(m)]
                gen_from_H(f, f'exhaustive_rep_n{n}_b{bits:0{m}b}',
                           H_rep, [0.1]*n, s)

        # ring code n=3..5, all achievable syndromes (from fault subsets; deduplicated)
        for n in range(3, 6):
            H_ring = np.zeros((n, n), dtype=np.uint8)
            for i in range(n): H_ring[i, i] = H_ring[i, (i+1)%n] = 1
            seen = set()
            for mask in range(2**n):
                fired = [j for j in range(n) if (mask >> j) & 1]
                s = tuple(map(int, syndrome_from_faults(H_ring, fired)))
                if s in seen: continue
                seen.add(s)
                gen_from_H(f, f'exhaustive_ring_n{n}_s{"".join(str(x) for x in s)}',
                           H_ring, [0.1]*n, list(s))

        # ---- Layer 4: Dijkstra ordering ----

        # test_cheaper_direct_path_wins: two parallel faults, cheaper (higher-p) wins
        H_dij1 = np.zeros((2, 2), dtype=np.uint8)
        H_dij1[0,0] = H_dij1[1,0] = 1   # f0: p=0.4 (cheaper)
        H_dij1[0,1] = H_dij1[1,1] = 1   # f1: p=0.05
        gen_from_H(f, 'layer4_cheaper_direct_wins', H_dij1, [0.4, 0.05], [1,1])

        # test_virtual_weight_accumulation: long cheap path vs short expensive path
        # det0–f0–det1–f1–det2 (each p=0.01) vs det0–f2–det2 (p=0.4)
        H_dij2 = np.zeros((3, 3), dtype=np.uint8)
        H_dij2[0,0] = H_dij2[1,0] = 1   # f0: det0–det1, p=0.01
        H_dij2[1,1] = H_dij2[2,1] = 1   # f1: det1–det2, p=0.01
        H_dij2[0,2] = H_dij2[2,2] = 1   # f2: det0–det2, p=0.4
        gen_from_H(f, 'layer4_vw_accumulation', H_dij2, [0.01, 0.01, 0.4], [1,0,1])

        # ---- Layer 5: edge cases ----

        # test_empty_syndrome
        gen_from_H(f, 'layer5_empty_syndrome', H_rep5, [0.1]*5, [0,0,0,0])

        # test_single_boundary_fault: one fault touching one detector
        H_bdy = np.zeros((1, 1), dtype=np.uint8); H_bdy[0, 0] = 1
        gen_from_H(f, 'layer5_single_boundary_fault', H_bdy, [0.3], [1])

        # test_null_space_in_ring_code: ring(4), fire f0+f2 → [1,0,1,0]
        H_ring4 = np.zeros((4, 4), dtype=np.uint8)
        for i in range(4): H_ring4[i, i] = H_ring4[i, (i+1)%4] = 1
        s_null = list(map(int, syndrome_from_faults(H_ring4, [0, 2])))
        gen_from_H(f, 'layer5_null_space_ring4', H_ring4, [0.1]*4, s_null)

        # test_independent_components_no_merge: two disconnected components
        H_ind = np.zeros((4, 2), dtype=np.uint8)
        H_ind[0,0] = H_ind[1,0] = 1   # component A
        H_ind[2,1] = H_ind[3,1] = 1   # component B
        gen_from_H(f, 'layer5_independent_components', H_ind, [0.3, 0.3], [1,1,1,1])

        # test_run_resets_state: rep(5) first run s1, second run s2
        s1 = [1, 0, 0, 1]; s2 = [1, 1, 0, 0]
        gen_from_H(f, 'layer5_reset_first_run',  H_rep5, [0.1]*5, s1)
        gen_from_H(f, 'layer5_reset_second_run', H_rep5, [0.1]*5, s2)

        # test_large_chain: chain n=20 (n_det=20, n_fault=19)
        n_lg = 20
        H_lg = np.zeros((n_lg, n_lg - 1), dtype=np.uint8)
        for j in range(n_lg - 1): H_lg[j, j] = H_lg[j+1, j] = 1
        s_lg = [0] * n_lg; s_lg[0] = s_lg[n_lg - 1] = 1
        gen_from_H(f, 'layer5_large_chain_n20', H_lg, [0.1]*(n_lg-1), s_lg)

        # test_connecting_fault_pushes_unseen_candidate: 6 dets, 5 faults
        H_unseen = np.zeros((6, 5), dtype=np.uint8)
        H_unseen[0,0] = H_unseen[1,0] = 1            # f0: det0–det1 (expensive)
        H_unseen[0,1] = H_unseen[2,1] = 1            # f1: det0–det2 (cheap)
        H_unseen[1,2] = H_unseen[3,2] = 1            # f2: det1–det3 (cheap)
        H_unseen[2,3] = H_unseen[3,3] = H_unseen[4,3] = 1  # f3: det2–det3–det4
        H_unseen[4,4] = H_unseen[5,4] = 1            # f4: det4–det5 (unseen until f3)
        s_unseen = list(map(int, syndrome_from_faults(H_unseen, [0])))  # [1,1,0,0,0,0]
        gen_from_H(f, 'layer5_unseen_candidate',
                   H_unseen, [0.01, 0.4, 0.4, 0.2, 0.1], s_unseen)

        # test_star_graph: hub det0, spokes det1..det4
        n_sp = 4
        H_star = np.zeros((n_sp+1, n_sp), dtype=np.uint8)
        for j in range(n_sp): H_star[0,j] = H_star[j+1,j] = 1
        s_star = list(map(int, syndrome_from_faults(H_star, list(range(n_sp)))))
        gen_from_H(f, 'layer5_star_graph', H_star, [0.2]*n_sp, s_star)

        # test_redundant_faults_null_space: 2 faults both touching dets 0,1
        H_red = np.zeros((2, 2), dtype=np.uint8)
        H_red[0,0] = H_red[1,0] = 1
        H_red[0,1] = H_red[1,1] = 1
        gen_from_H(f, 'layer5_redundant_faults', H_red, [0.3, 0.2], [1,1])

        # ================================================================
        # Section 3 — Layer 6: surface code (d=3, 1 round, rotated memory-Z)
        # ================================================================

        def make_surface_pcm(distance=3, rounds=1, p=0.01):
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=distance,
                rounds=rounds,
                after_clifford_depolarization=p,
            )
            dem = circuit.detector_error_model(flatten_loops=True)
            return ParityCheckMatrices.from_DEM(dem, decompose=False)

        def surface_syndrome_from_faults(pcm, fired):
            s = np.zeros(pcm.H.shape[0], dtype=np.uint8)
            for j in fired:
                s ^= pcm.H[:, j]
            return s

        surf_pcm = make_surface_pcm(distance=3, rounds=1, p=0.01)
        n_surf_fault = surf_pcm.H.shape[1]

        # test_empty_syndrome
        write_test(f, 'layer6_surface_empty_syndrome', surf_pcm,
                   list(np.zeros(surf_pcm.H.shape[0], dtype=np.uint8)))

        # test_single_fault_each — fire each fault individually
        for j in range(n_surf_fault):
            s = list(map(int, surface_syndrome_from_faults(surf_pcm, [j])))
            write_test(f, f'layer6_surface_single_fault_{j}', surf_pcm, s)

        # test_two_fault_combinations — fire every pair of distinct faults
        for j0 in range(n_surf_fault):
            for j1 in range(j0 + 1, n_surf_fault):
                s = list(map(int, surface_syndrome_from_faults(surf_pcm, [j0, j1])))
                write_test(f, f'layer6_surface_pair_{j0}_{j1}', surf_pcm, s)

        # test_random_fault_subsets — representative fixed subsets
        subsets = [
            [],
            [0],
            list(range(n_surf_fault)),
            list(range(n_surf_fault // 2)),
            list(range(n_surf_fault // 2, n_surf_fault)),
        ]
        for idx, fired in enumerate(subsets):
            s = list(map(int, surface_syndrome_from_faults(surf_pcm, fired)))
            write_test(f, f'layer6_surface_subset_{idx}', surf_pcm, s)

        # extended random fault subsets (seeded, diverse fault counts)
        rng_surf = np.random.default_rng(7)
        for trial in range(20):
            k      = int(rng_surf.integers(0, n_surf_fault + 1))
            fired  = sorted(rng_surf.choice(n_surf_fault, size=k, replace=False).tolist())
            s      = list(map(int, surface_syndrome_from_faults(surf_pcm, fired)))
            write_test(f, f'layer6_surface_random_{trial:02d}', surf_pcm, s)

    n_tests = sum(1 for line in open(out_path) if line.startswith('TEST '))
    print(f'Wrote {out_path}')
    print(f'Total test cases: {n_tests}')
