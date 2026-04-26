#include "clustering_overgrow_batch.hpp"
#include "gf2matrix.hpp"    // GF2Matrix::pack_vec — used in create_degenerate_cycle_regions

#include <algorithm>
#include <cassert>
#include <limits>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ClusteringOvgBatch::ClusteringOvgBatch(
    int n_det, int n_fault,
    std::vector<std::vector<int>> check_to_faults,
    std::vector<std::vector<int>> fault_to_checks,
    std::vector<double>           weights,
    int n_logical,
    std::vector<std::vector<uint8_t>> L)
    : n_det_(n_det),
      n_fault_(n_fault),
      check_to_faults_(std::move(check_to_faults)),
      fault_to_checks_(std::move(fault_to_checks)),
      weights_(std::move(weights)),
      n_logical_(n_logical)
{
    // Precompute L_col_packed_[j] = uint64_t with bit i = L[i][j].
    // This encodes all n_logical observable bits for fault j in one word,
    // enabling O(1) observable-flip lookup during create_degenerate_cycle_regions.
    L_col_packed_.assign(n_fault_, 0);
    assert(n_logical == 0 || (int)L.size() == n_logical);
    for (int i = 0; i < n_logical; ++i) {
        assert((int)L[i].size() == n_fault_);
        const uint64_t imask = 1ULL << i;
        for (int j = 0; j < n_fault_; ++j)
            if (L[i][j] & 1)
                L_col_packed_[j] |= imask;
    }
    // L is consumed; the raw matrix is not retained.
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

void ClusteringOvgBatch::run(const std::vector<uint8_t>& syndrome,
                              int over_grow_step, int bits_per_step)
{
    assert(bits_per_step >= 1 && "bits_per_step must be >= 1");
    over_grow_step_ = over_grow_step;
    bits_per_step_  = bits_per_step;
    active_valid_clusters.clear();
    clusters_initialization(syndrome);

    std::vector<ClusterStateOGB*> invalid;
    for (auto& up : clusters_)
        if (!up->valid) invalid.push_back(up.get());

    while (!invalid.empty()) {
        for (ClusterStateOGB* cl : invalid)
            if (cl->active && !cl->valid)
                _grow_one_step(cl);
        invalid.clear();
        for (auto& up : clusters_)
            if (up->active && !up->valid)
                invalid.push_back(up.get());
    }

    _get_active_valid_clusters();
}

// ---------------------------------------------------------------------------
// clusters_initialization
// ---------------------------------------------------------------------------

void ClusteringOvgBatch::clusters_initialization(const std::vector<uint8_t>& syndrome)
{
    syndrome_ = syndrome;
    global_check_membership_.assign(n_det_,   nullptr);
    global_fault_membership_.assign(n_fault_,  nullptr);
    clusters_.clear();

    for (int i = 0; i < n_det_; ++i)
        if (syndrome[i])
            _init_each_cluster(i);
}

// ---------------------------------------------------------------------------
// _init_each_cluster
// ---------------------------------------------------------------------------

void ClusteringOvgBatch::_init_each_cluster(int seed)
{
    clusters_.push_back(std::make_unique<ClusterStateOGB>(seed));
    ClusterStateOGB* cl = clusters_.back().get();

    cl->check_nodes.insert(seed);
    cl->boundary_check_nodes.insert(seed);
    cl->enclosed_syndromes.insert(seed);
    global_check_membership_[seed] = cl;

    const double inf = std::numeric_limits<double>::infinity();
    for (int j : check_to_faults_[seed]) {
        double w  = weights_[j];
        auto it   = cl->dist.find(j);
        double bst = (it != cl->dist.end()) ? it->second : inf;
        if (w < bst) {
            cl->dist[j] = w;
            cl->heap.push({w, j});
        }
    }
}

// ---------------------------------------------------------------------------
// _add_fault_batch
//
// Absorb k free faults into cl in one add_columns call.
//
// Steps
// -----
// 1. Union of new RREF check rows across all k faults (discovery order).
// 2. Assign local indices; update check ownership.
// 3. Build one column vector per fault against the unified check-index map.
// 4. Call add_columns(columns, s_extra).
// 5. Register faults; push Dijkstra candidates.
// ---------------------------------------------------------------------------

void ClusteringOvgBatch::_add_fault_batch(
    ClusterStateOGB* cl,
    const std::vector<std::pair<double,int>>& faults)
{
    if (faults.empty()) return;
    const double inf = std::numeric_limits<double>::infinity();

    // Logical bridge check nodes are the last n_logical_ detector indices.
    // They participate in ownership/collision tracking but NOT in the RREF.
    // Excluding them matches Python's _add_fault behaviour: the RREF null
    // space then contains logical-error vectors (L @ z != 0), whereas
    // including them would impose L @ z = 0 constraints and eliminate those
    // vectors entirely, making create_degenerate_cycle_regions always return
    // zero logical-error null-basis vectors.
    const int logical_start = n_det_ - n_logical_;

    // ------------------------------------------------------------------
    // Step 1: Union of new RREF check rows across all faults (regular only).
    // Logical bridge nodes (c >= logical_start) are skipped for the RREF.
    // ------------------------------------------------------------------
    std::vector<int> all_new;
    std::unordered_set<int> seen_new;

    for (auto& [vw, j] : faults) {
        for (int c : fault_to_checks_[j]) {
            if (n_logical_ > 0 && c >= logical_start) continue;   // logical — no RREF row
            if (cl->pcm_check_idx_to_cluster_check_idx.find(c)
                    == cl->pcm_check_idx_to_cluster_check_idx.end()
                && seen_new.find(c) == seen_new.end()) {
                all_new.push_back(c);
                seen_new.insert(c);
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 2: Assign local RREF indices for new regular checks; update ownership.
    // ------------------------------------------------------------------
    const int n_existing = cl->rref.n_checks;
    for (int idx = 0; idx < (int)all_new.size(); ++idx) {
        int c     = all_new[idx];
        int local = n_existing + idx;
        cl->pcm_check_idx_to_cluster_check_idx[c] = local;
        cl->cluster_check_idx_to_pcm_check_idx.push_back(c);
        if (!cl->check_nodes.count(c)) {
            cl->check_nodes.insert(c);
            cl->boundary_check_nodes.insert(c);
            global_check_membership_[c] = cl;
        }
    }

    // Claim ownership of logical bridge nodes (no RREF row).
    if (n_logical_ > 0) {
        for (auto& [vw, j] : faults) {
            for (int c : fault_to_checks_[j]) {
                if (c >= logical_start && !cl->check_nodes.count(c)) {
                    cl->check_nodes.insert(c);
                    global_check_membership_[c] = cl;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 3: Build column vectors (regular checks only).
    // ------------------------------------------------------------------
    const int n_full = n_existing + (int)all_new.size();
    std::vector<std::vector<uint8_t>> columns(faults.size(),
                                              std::vector<uint8_t>(n_full, 0));
    for (int fi = 0; fi < (int)faults.size(); ++fi) {
        int j = faults[fi].second;
        for (int c : fault_to_checks_[j]) {
            if (n_logical_ > 0 && c >= logical_start) continue;   // skip logical
            columns[fi][cl->pcm_check_idx_to_cluster_check_idx.at(c)] = 1;
        }
    }

    // ------------------------------------------------------------------
    // Step 4: Build s_extra for new check rows; call add_columns.
    // ------------------------------------------------------------------
    std::vector<uint8_t> s_extra;
    const std::vector<uint8_t>* s_ptr = nullptr;
    if (!all_new.empty()) {
        s_extra.resize(all_new.size());
        for (int idx = 0; idx < (int)all_new.size(); ++idx)
            s_extra[idx] = syndrome_[all_new[idx]];
        s_ptr = &s_extra;
    }
    cl->rref.add_columns(columns, s_ptr);

    // ------------------------------------------------------------------
    // Step 5: Register faults; push Dijkstra candidates.
    // ------------------------------------------------------------------
    for (auto& [vw, j] : faults) {
        cl->fault_nodes.insert(j);
        cl->cluster_fault_idx_to_pcm_fault_idx.push_back(j);
        global_fault_membership_[j] = cl;
    }

    for (auto& [vw, j] : faults) {
        for (int c : fault_to_checks_[j]) {
            for (int k : check_to_faults_[c]) {
                if (global_fault_membership_[k] != cl) {
                    double new_vw = vw + weights_[k];
                    auto it = cl->dist.find(k);
                    double best = (it != cl->dist.end()) ? it->second : inf;
                    if (new_vw < best) {
                        cl->dist[k] = new_vw;
                        cl->heap.push({new_vw, k});
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// _merge_batch
//
// Merge cl and all clusters in `others` using k connecting faults at once.
//
// Steps
// -----
// 1. Determine surviving cluster (largest by fault count).
// 2. Block-diagonal RREF merge of all involved clusters (no connecting edges).
// 3. Add all connecting faults via one add_columns call.
// 4. Update surviving cluster's state; absorb smaller clusters.
// 5. Register connecting faults; push Dijkstra candidates.
// ---------------------------------------------------------------------------

ClusterStateOGB* ClusteringOvgBatch::_merge_batch(
    ClusterStateOGB* cl,
    std::unordered_set<ClusterStateOGB*>& others,
    const std::vector<int>&    connecting_js,
    const std::vector<double>& connecting_vws)
{
    const double inf = std::numeric_limits<double>::infinity();

    std::vector<ClusterStateOGB*> active_others;
    for (ClusterStateOGB* o : others)
        if (o->active) active_others.push_back(o);

    // ------------------------------------------------------------------
    // Determine surviving cluster (largest by fault count).
    // ------------------------------------------------------------------
    std::vector<ClusterStateOGB*> all_involved;
    all_involved.push_back(cl);
    all_involved.insert(all_involved.end(),
                        active_others.begin(), active_others.end());
    std::sort(all_involved.begin(), all_involved.end(),
              [](ClusterStateOGB* a, ClusterStateOGB* b) {
                  return a->fault_nodes.size() > b->fault_nodes.size();
              });
    ClusterStateOGB* larger = all_involved[0];

    // ------------------------------------------------------------------
    // Step 1: Block-diagonal RREF merge of all clusters.
    //
    // Build unified check map from all involved clusters.
    // Move larger's RREF into merged_rref; merge each smaller in turn
    // using IncrementalRREF::merge with empty connecting edges.
    // ------------------------------------------------------------------
    std::vector<int>            unified_list = larger->cluster_check_idx_to_pcm_check_idx;
    std::unordered_map<int,int> unified_map  = larger->pcm_check_idx_to_cluster_check_idx;
    int offset = larger->rref.n_checks;

    IncrementalRREF merged_rref = std::move(larger->rref);

    for (int i = 1; i < (int)all_involved.size(); ++i) {
        ClusterStateOGB* other = all_involved[i];
        for (auto& [c, local] : other->pcm_check_idx_to_cluster_check_idx)
            unified_map[c] = offset + local;
        for (int c : other->cluster_check_idx_to_pcm_check_idx)
            unified_list.push_back(c);
        merged_rref = IncrementalRREF::merge(merged_rref, other->rref, {});
        offset += other->rref.n_checks;
    }

    // ------------------------------------------------------------------
    // Step 2: Collect new RREF check rows across all connecting faults.
    // Logical bridge nodes are skipped — ownership only, no RREF row.
    // ------------------------------------------------------------------
    const int logical_start = n_det_ - n_logical_;

    std::vector<int>        all_new_checks;
    std::unordered_set<int> seen_new;

    for (int j : connecting_js) {
        for (int c : fault_to_checks_[j]) {
            if (n_logical_ > 0 && c >= logical_start) continue;   // logical — no RREF row
            if (!unified_map.count(c) && !seen_new.count(c)) {
                all_new_checks.push_back(c);
                seen_new.insert(c);
            }
        }
    }
    for (int idx = 0; idx < (int)all_new_checks.size(); ++idx) {
        unified_map[all_new_checks[idx]] = offset + idx;
        unified_list.push_back(all_new_checks[idx]);
    }

    // Claim ownership of logical bridge nodes from connecting faults.
    if (n_logical_ > 0) {
        for (int j : connecting_js) {
            for (int c : fault_to_checks_[j]) {
                if (c >= logical_start) {
                    larger->check_nodes.insert(c);
                    global_check_membership_[c] = larger;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 3: Build column vectors (regular checks only); call add_columns.
    // ------------------------------------------------------------------
    const int n_total = offset + (int)all_new_checks.size();
    std::vector<std::vector<uint8_t>> conn_cols(
        connecting_js.size(), std::vector<uint8_t>(n_total, 0));
    for (int ci = 0; ci < (int)connecting_js.size(); ++ci)
        for (int c : fault_to_checks_[connecting_js[ci]]) {
            if (n_logical_ > 0 && c >= logical_start) continue;   // skip logical
            conn_cols[ci][unified_map.at(c)] = 1;
        }

    std::vector<uint8_t> s_extra;
    const std::vector<uint8_t>* s_ptr = nullptr;
    if (!all_new_checks.empty()) {
        s_extra.resize(all_new_checks.size());
        for (int idx = 0; idx < (int)all_new_checks.size(); ++idx)
            s_extra[idx] = syndrome_[all_new_checks[idx]];
        s_ptr = &s_extra;
    }
    merged_rref.add_columns(conn_cols, s_ptr);

    // ------------------------------------------------------------------
    // Step 4: Update surviving cluster's state.
    // ------------------------------------------------------------------
    larger->rref                              = std::move(merged_rref);
    larger->cluster_check_idx_to_pcm_check_idx = std::move(unified_list);
    larger->pcm_check_idx_to_cluster_check_idx = std::move(unified_map);

    for (int i = 1; i < (int)all_involved.size(); ++i) {
        ClusterStateOGB* other = all_involved[i];
        for (int j2 : other->cluster_fault_idx_to_pcm_fault_idx) {
            larger->fault_nodes.insert(j2);
            larger->cluster_fault_idx_to_pcm_fault_idx.push_back(j2);
            global_fault_membership_[j2] = larger;
        }
        for (int c : other->check_nodes) {
            larger->check_nodes.insert(c);
            global_check_membership_[c] = larger;
        }
        for (int c : other->boundary_check_nodes)
            larger->boundary_check_nodes.insert(c);
        for (int s : other->enclosed_syndromes)
            larger->enclosed_syndromes.insert(s);
        larger->heap.absorb(other->heap);
        for (auto& [k, d] : other->dist) {
            auto it = larger->dist.find(k);
            if (it == larger->dist.end() || d < it->second)
                larger->dist[k] = d;
        }
        other->active = false;
    }

    // ------------------------------------------------------------------
    // Step 5: Register connecting faults; update new-check ownership;
    // push Dijkstra candidates reachable through connecting faults.
    // ------------------------------------------------------------------
    for (int j : connecting_js) {
        larger->fault_nodes.insert(j);
        larger->cluster_fault_idx_to_pcm_fault_idx.push_back(j);
        global_fault_membership_[j] = larger;
    }
    for (int c : all_new_checks) {
        larger->check_nodes.insert(c);
        larger->boundary_check_nodes.insert(c);
        global_check_membership_[c] = larger;
    }
    for (int ci = 0; ci < (int)connecting_js.size(); ++ci) {
        int    j  = connecting_js[ci];
        double vw = connecting_vws[ci];
        for (int c : fault_to_checks_[j]) {
            for (int k : check_to_faults_[c]) {
                if (global_fault_membership_[k] != larger) {
                    double new_vw = vw + weights_[k];
                    auto it = larger->dist.find(k);
                    double best = (it != larger->dist.end()) ? it->second : inf;
                    if (new_vw < best) {
                        larger->dist[k] = new_vw;
                        larger->heap.push({new_vw, k});
                    }
                }
            }
        }
    }

    return larger;
}

// ---------------------------------------------------------------------------
// _grow_one_step
//
// Pop up to bits_per_step_ valid fault candidates; classify each as free
// (no collision) or collision (some check owned by another cluster).
// Merge collisions first (so free faults go into the merged cluster),
// then batch-add free faults.  Apply the over-grow countdown last.
// ---------------------------------------------------------------------------

ClusterStateOGB* ClusteringOvgBatch::_grow_one_step(ClusterStateOGB* cl)
{
    std::vector<std::pair<double,int>>   free_faults;
    std::vector<std::pair<double,int>>   collision_faults;
    std::unordered_set<ClusterStateOGB*> merge_set;

    int actions = 0;
    while (!cl->heap.empty() && actions < bits_per_step_) {
        auto [vw, j] = cl->heap.top();
        cl->heap.pop();

        // Skip stale heap entries (lazy deletion).
        auto dit = cl->dist.find(j);
        if (dit == cl->dist.end() || vw > dit->second) continue;
        // Skip faults already absorbed into this cluster.
        if (global_fault_membership_[j] == cl) continue;

        // Classify: free if all checks are in cl or unclaimed;
        // collision if any check belongs to another cluster.
        std::unordered_set<ClusterStateOGB*> colliding;
        for (int c : fault_to_checks_[j]) {
            ClusterStateOGB* owner = global_check_membership_[c];
            if (owner && owner != cl) colliding.insert(owner);
        }

        if (!colliding.empty()) {
            collision_faults.push_back({vw, j});
            merge_set.insert(colliding.begin(), colliding.end());
        } else {
            free_faults.push_back({vw, j});
        }
        ++actions;
    }

    // Process: merge first so free faults can be added to the merged cluster.
    if (!merge_set.empty()) {
        std::vector<int>    cjs;
        std::vector<double> cvws;
        for (auto& [vw, j] : collision_faults) { cjs.push_back(j); cvws.push_back(vw); }
        cl = _merge_batch(cl, merge_set, cjs, cvws);
    }
    if (!free_faults.empty())
        _add_fault_batch(cl, free_faults);

    // ------------------------------------------------------------------
    // Over-grow countdown (mirrors clustering_overgrow.py exactly).
    //
    // is_neutral: syndrome in image(H_cluster) — checked once after ALL
    // faults in this step are processed.  This single check per step
    // (not per fault) is the semantic difference from bits_per_step=1.
    // ------------------------------------------------------------------
    bool is_neutral = cl->rref.is_valid();
    if (!is_neutral) {
        cl->overgrow_budget = -1;
    } else if (cl->overgrow_budget < 0) {
        cl->overgrow_budget = over_grow_step_;  // start countdown
    } else if (cl->overgrow_budget > 0) {
        --cl->overgrow_budget;                  // count down
    }
    cl->valid = (cl->overgrow_budget == 0);
    return cl;
}

// ---------------------------------------------------------------------------
// _get_active_valid_clusters
//
// Identify active valid clusters and build L_col_packed_local for each.
//
// L_col_packed_local[j_local] = L_col_packed_[cluster_fault[j_local]].
// This per-cluster copy eliminates the double indirection in
// create_degenerate_cycle_regions and fits in L1 cache (n_faults_cl × 8 B).
// Built here rather than during grow/merge to avoid wasted work for
// clusters that are later absorbed.
// ---------------------------------------------------------------------------

void ClusteringOvgBatch::_get_active_valid_clusters()
{
    active_valid_clusters.clear();
    for (auto& up : clusters_) {
        ClusterStateOGB* cl = up.get();
        if (!cl->active || !cl->valid) continue;

        if (n_logical_ > 0) {
            const int n_fl = static_cast<int>(cl->cluster_fault_idx_to_pcm_fault_idx.size());
            cl->L_col_packed_local.resize(n_fl);
            for (int j_local = 0; j_local < n_fl; ++j_local)
                cl->L_col_packed_local[j_local] =
                    L_col_packed_[cl->cluster_fault_idx_to_pcm_fault_idx[j_local]];
        }

        active_valid_clusters[cl->cluster_id] = cl;
    }
}

// ---------------------------------------------------------------------------
// create_degenerate_cycle_regions
//
// For each null-space basis vector z in every active valid cluster:
//   flip_acc = XOR{ L_col_packed_local[j] : j where z[j] == 1 }
//
// The branchless mask trick replaces the branch `if (z[j])` with
//   mask = -(uint64_t)(z[j] & 1)   →   all-ones if z[j]=1, zero if z[j]=0
//   L_col_packed_local[j] & mask   →   full value or zero
//
// This allows the compiler to auto-vectorise the inner loop with SIMD,
// since it is a uniform reduction with no data-dependent control flow.
//
// flip_acc == 0 → stabilizer; flip_acc != 0 → logical error,
//   with bit k set in flip_acc iff observable k is flipped by z.
//
// Requires n_logical_ <= 64.
// ---------------------------------------------------------------------------

std::unordered_map<int, ClusterCycleInfo>
ClusteringOvgBatch::create_degenerate_cycle_regions() const
{
    assert(n_logical_ <= 64 &&
           "create_degenerate_cycle_regions: n_logical must be <= 64");

    std::unordered_map<int, ClusterCycleInfo> result;

    for (auto& [cl_id, cl] : active_valid_clusters) {
        ClusterCycleInfo info;

        for (const auto& z : cl->rref.Z) {
            if (n_logical_ == 0) {
                info.stabilizers.push_back(z);
                continue;
            }

            // Branchless scan: accumulate observable flip bits for this z.
            uint64_t flip_acc = 0;
            const int n_fl = static_cast<int>(z.size());
            for (int j = 0; j < n_fl; ++j)
                flip_acc ^= cl->L_col_packed_local[j]
                            & -(uint64_t)(z[j] & 1);   // mask: all-1s or 0

            if (flip_acc == 0) {
                info.stabilizers.push_back(z);
            } else {
                info.logical_errors.push_back(
                    {z, static_cast<int64_t>(flip_acc)});
            }
        }

        result[cl_id] = std::move(info);
    }

    return result;
}

// ---------------------------------------------------------------------------
// run_and_create_degenerate_cycle_regions
// ---------------------------------------------------------------------------

std::unordered_map<int, ClusterCycleInfo>
ClusteringOvgBatch::run_and_create_degenerate_cycle_regions(
    const std::vector<uint8_t>& syndrome,
    int over_grow_step,
    int bits_per_step)
{
    run(syndrome, over_grow_step, bits_per_step);
    return create_degenerate_cycle_regions();
}
