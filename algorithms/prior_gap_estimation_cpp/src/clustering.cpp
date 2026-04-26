#include "clustering.hpp"

#include <algorithm>
#include <cassert>
#include <limits>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Clustering::Clustering(int n_det, int n_fault,
                       std::vector<std::vector<int>> check_to_faults,
                       std::vector<std::vector<int>> fault_to_checks,
                       std::vector<double>           weights)
    : n_det_(n_det),
      n_fault_(n_fault),
      check_to_faults_(std::move(check_to_faults)),
      fault_to_checks_(std::move(fault_to_checks)),
      weights_(std::move(weights))
{}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

void Clustering::run(const std::vector<uint8_t>& syndrome)
{
    active_valid_clusters.clear();
    clusters_initialization(syndrome);

    // Collect initially invalid clusters.
    std::vector<ClusterState*> invalid;
    for (auto& up : clusters_)
        if (!up->valid)
            invalid.push_back(up.get());

    while (!invalid.empty()) {
        for (ClusterState* cl : invalid) {
            if (cl->active && !cl->valid)
                _grow_one_step(cl);
        }
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

void Clustering::clusters_initialization(const std::vector<uint8_t>& syndrome)
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

void Clustering::_init_each_cluster(int seed)
{
    clusters_.push_back(std::make_unique<ClusterState>(seed));
    ClusterState* cl = clusters_.back().get();

    cl->check_nodes.insert(seed);
    cl->boundary_check_nodes.insert(seed);
    cl->enclosed_syndromes.insert(seed);
    global_check_membership_[seed] = cl;

    // Seed the Dijkstra heap with faults directly adjacent to the seed.
    const double inf = std::numeric_limits<double>::infinity();
    for (int j : check_to_faults_[seed]) {
        double w = weights_[j];
        auto it = cl->dist.find(j);
        double best = (it != cl->dist.end()) ? it->second : inf;
        if (w < best) {
            cl->dist[j] = w;
            cl->heap.push({w, j});
        }
    }
}

// ---------------------------------------------------------------------------
// _add_fault
// ---------------------------------------------------------------------------

void Clustering::_add_fault(ClusterState* cl, int j, double vw)
{
    const double inf = std::numeric_limits<double>::infinity();

    // Checks that need new RREF rows (not yet in local index map).
    std::vector<int> new_for_rref;
    for (int c : fault_to_checks_[j])
        if (cl->pcm_check_idx_to_cluster_check_idx.find(c)
                == cl->pcm_check_idx_to_cluster_check_idx.end())
            new_for_rref.push_back(c);

    // Assign consecutive local RREF indices.
    const int n_existing = cl->rref.n_checks;
    for (int idx = 0; idx < (int)new_for_rref.size(); ++idx) {
        int c     = new_for_rref[idx];
        int local = n_existing + idx;
        cl->pcm_check_idx_to_cluster_check_idx[c] = local;
        cl->cluster_check_idx_to_pcm_check_idx.push_back(c);
        if (cl->check_nodes.find(c) == cl->check_nodes.end()) {
            cl->check_nodes.insert(c);
            cl->boundary_check_nodes.insert(c);
            global_check_membership_[c] = cl;
        }
    }

    // Build h_j in local RREF index space.
    const int h_size = n_existing + (int)new_for_rref.size();
    std::vector<uint8_t> h_j(h_size, 0);
    for (int c : fault_to_checks_[j])
        h_j[cl->pcm_check_idx_to_cluster_check_idx.at(c)] = 1;

    // Call add_column — with or without s_extra.
    if (new_for_rref.empty()) {
        cl->rref.add_column(h_j);
    } else {
        std::vector<uint8_t> s_extra((int)new_for_rref.size());
        for (int idx = 0; idx < (int)new_for_rref.size(); ++idx)
            s_extra[idx] = syndrome_[new_for_rref[idx]];
        cl->rref.add_column(h_j, s_extra);
    }

    // Register fault in cluster.
    cl->fault_nodes.insert(j);
    cl->cluster_fault_idx_to_pcm_fault_idx.push_back(j);
    global_fault_membership_[j] = cl;

    // Push new Dijkstra candidates reachable through fault j.
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

// ---------------------------------------------------------------------------
// _merge
// ---------------------------------------------------------------------------

ClusterState* Clustering::_merge(ClusterState* cl,
                                  std::unordered_set<ClusterState*>& others,
                                  int connecting_j, double connecting_vw)
{
    const double inf = std::numeric_limits<double>::infinity();

    // Collect active others.
    std::vector<ClusterState*> active_others;
    for (ClusterState* o : others)
        if (o->active) active_others.push_back(o);

    // Determine surviving cluster: largest by fault count.
    std::vector<ClusterState*> all_involved;
    all_involved.push_back(cl);
    all_involved.insert(all_involved.end(), active_others.begin(), active_others.end());
    std::sort(all_involved.begin(), all_involved.end(),
              [](ClusterState* a, ClusterState* b) {
                  return a->fault_nodes.size() > b->fault_nodes.size();
              });
    ClusterState* larger = all_involved[0];

    // -----------------------------------------------------------------
    // Step 1: block-diagonal RREF merge of all clusters.
    // The surviving cluster's RREF is extracted (moved) first so we avoid
    // a copy; it is moved back at the end.
    // -----------------------------------------------------------------
    std::vector<int>            unified_check_list = larger->cluster_check_idx_to_pcm_check_idx;
    std::unordered_map<int,int> unified_check_map  = larger->pcm_check_idx_to_cluster_check_idx;
    int offset = larger->rref.n_checks;

    IncrementalRREF merged_rref = std::move(larger->rref);

    for (int i = 1; i < (int)all_involved.size(); ++i) {
        ClusterState* other = all_involved[i];

        // Extend unified map: other's local indices shift by current offset.
        for (auto& [c, local] : other->pcm_check_idx_to_cluster_check_idx)
            unified_check_map[c] = offset + local;
        for (int c : other->cluster_check_idx_to_pcm_check_idx)
            unified_check_list.push_back(c);

        // Block-diagonal merge (disjoint check sets guaranteed by invariant).
        merged_rref = IncrementalRREF::merge(merged_rref, other->rref, {});
        offset += other->rref.n_checks;
    }

    // -----------------------------------------------------------------
    // Step 2: add connecting_j as the single connecting edge.
    // Any checks of connecting_j not yet in the unified map become new rows.
    // -----------------------------------------------------------------
    std::vector<int> new_checks_j;
    for (int c : fault_to_checks_[connecting_j])
        if (unified_check_map.find(c) == unified_check_map.end())
            new_checks_j.push_back(c);

    for (int idx = 0; idx < (int)new_checks_j.size(); ++idx) {
        int c = new_checks_j[idx];
        unified_check_map[c] = offset + idx;
        unified_check_list.push_back(c);
    }

    const int h_size = offset + (int)new_checks_j.size();
    std::vector<uint8_t> h_j(h_size, 0);
    for (int c : fault_to_checks_[connecting_j])
        h_j[unified_check_map.at(c)] = 1;

    if (new_checks_j.empty()) {
        merged_rref.add_column(h_j);
    } else {
        std::vector<uint8_t> s_extra((int)new_checks_j.size());
        for (int idx = 0; idx < (int)new_checks_j.size(); ++idx)
            s_extra[idx] = syndrome_[new_checks_j[idx]];
        merged_rref.add_column(h_j, s_extra);
    }

    // -----------------------------------------------------------------
    // Step 3: update surviving cluster's state.
    // -----------------------------------------------------------------
    larger->rref                              = std::move(merged_rref);
    larger->cluster_check_idx_to_pcm_check_idx = std::move(unified_check_list);
    larger->pcm_check_idx_to_cluster_check_idx = std::move(unified_check_map);

    // Absorb each smaller cluster into larger.
    for (int i = 1; i < (int)all_involved.size(); ++i) {
        ClusterState* other = all_involved[i];

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

        // Lazy heap merge: bulk absorb — O(|larger|+|other|) via make_heap,
        // better than the O(|other| log |larger|) pop-loop of priority_queue.
        larger->heap.absorb(other->heap);
        // Merge dist maps: keep the minimum distance for each fault.
        for (auto& [k, d] : other->dist) {
            auto it = larger->dist.find(k);
            if (it == larger->dist.end() || d < it->second)
                larger->dist[k] = d;
        }

        other->active = false;
    }

    // Register connecting_j in the surviving cluster.
    larger->fault_nodes.insert(connecting_j);
    larger->cluster_fault_idx_to_pcm_fault_idx.push_back(connecting_j);
    global_fault_membership_[connecting_j] = larger;

    // Update check ownership for any new checks from connecting_j.
    for (int c : new_checks_j) {
        larger->check_nodes.insert(c);
        larger->boundary_check_nodes.insert(c);
        global_check_membership_[c] = larger;
    }

    // Push new Dijkstra candidates reachable through connecting_j.
    for (int c : fault_to_checks_[connecting_j]) {
        for (int k : check_to_faults_[c]) {
            if (global_fault_membership_[k] != larger) {
                double new_vw = connecting_vw + weights_[k];
                auto it = larger->dist.find(k);
                double best = (it != larger->dist.end()) ? it->second : inf;
                if (new_vw < best) {
                    larger->dist[k] = new_vw;
                    larger->heap.push({new_vw, k});
                }
            }
        }
    }

    return larger;
}

// ---------------------------------------------------------------------------
// _grow_one_step
// ---------------------------------------------------------------------------

ClusterState* Clustering::_grow_one_step(ClusterState* cl)
{
    const double inf = std::numeric_limits<double>::infinity();

    std::unordered_set<ClusterState*> merge_list;
    int    connecting_j  = -1;
    double connecting_vw = inf;

    while (!cl->heap.empty()) {
        auto [vw, j] = cl->heap.top();
        cl->heap.pop();

        // Skip stale heap entries (lazy deletion).
        auto dit = cl->dist.find(j);
        if (dit == cl->dist.end() || vw > dit->second)
            continue;

        // Skip faults already absorbed into this cluster.
        if (global_fault_membership_[j] == cl)
            continue;

        // Check for collisions: does j touch a check owned by another cluster?
        std::unordered_set<ClusterState*> colliding;
        for (int c : fault_to_checks_[j]) {
            ClusterState* owner = global_check_membership_[c];
            if (owner != nullptr && owner != cl)
                colliding.insert(owner);
        }

        if (!colliding.empty()) {
            // Collision: j is the connecting edge — do not add to any cluster yet.
            merge_list    = std::move(colliding);
            connecting_j  = j;
            connecting_vw = vw;
        } else {
            // Free: all checks are in cl or unclaimed.
            _add_fault(cl, j, vw);
        }
        break;  // exactly one action per step
    }

    if (!merge_list.empty())
        cl = _merge(cl, merge_list, connecting_j, connecting_vw);

    cl->valid = cl->rref.is_valid();
    return cl;
}

// ---------------------------------------------------------------------------
// _get_active_valid_clusters
// ---------------------------------------------------------------------------

void Clustering::_get_active_valid_clusters()
{
    active_valid_clusters.clear();
    for (auto& up : clusters_) {
        ClusterState* cl = up.get();
        if (cl->active && cl->valid)
            active_valid_clusters[cl->cluster_id] = cl;
    }
}
