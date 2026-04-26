#include "prior_gap_estimator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
//
// weights is copied into engine_ (engine takes by value) and then moved into
// weights_ so both hold independent copies without an extra allocation.
// L is moved into engine_ since we do not need it after construction.
// ---------------------------------------------------------------------------

PriorGapEstimator::PriorGapEstimator(
    int n_det, int n_fault,
    std::vector<std::vector<int>>    check_to_faults,
    std::vector<std::vector<int>>    fault_to_checks,
    std::vector<double>              weights,
    int n_logical,
    std::vector<std::vector<uint8_t>> L)
    : engine_(n_det, n_fault,
              std::move(check_to_faults),
              std::move(fault_to_checks),
              weights,           // lvalue: copied into engine's by-value param
              n_logical,
              std::move(L)),     // moved; we do not need L after this
      weights_(std::move(weights)),
      n_logical_(n_logical),
      n_fault_(n_fault)
{}

// ---------------------------------------------------------------------------
// _get_correction  (static)
//
// Returns the RREF pivot solution e1 in local fault index space (length
// rref.n_bits).  pivot_map[i] == -1 means row i has no pivot; otherwise
// fault index pivot_map[i] is in the correction iff s_prime[i] == 1.
// Mirrors Python PriorGapEstimatorUse._get_correction(cl).
// ---------------------------------------------------------------------------

std::vector<uint8_t> PriorGapEstimator::_get_correction(const ClusterStateOGB& cl)
{
    const IncrementalRREF& rref = cl.rref;
    std::vector<uint8_t> e1(rref.n_bits, 0);
    for (int i = 0; i < rref.n_checks; ++i) {
        int pm = rref.pivot_map[i];
        if (pm >= 0 && rref.s_prime[i] == 1)
            e1[pm] = 1;
    }
    return e1;
}

// ---------------------------------------------------------------------------
// _gap_binary
//
// Returns -inf if any cluster has at least one logical-error null-basis
// vector, +inf otherwise.  Exits early on the first logical error found.
// ---------------------------------------------------------------------------

double PriorGapEstimator::_gap_binary(
    const std::unordered_map<int, ClusterCycleInfo>& regions)
{
    for (const auto& [cl_id, info] : regions)
        if (!info.logical_errors.empty())
            return -std::numeric_limits<double>::infinity();
    return std::numeric_limits<double>::infinity();
}

// ---------------------------------------------------------------------------
// _gap_hamming
//
// Per cluster: minimum Hamming weight over logical-error null-basis vectors.
// Aggregated across clusters by min or sum.
// ---------------------------------------------------------------------------

double PriorGapEstimator::_gap_hamming(
    const std::unordered_map<int, ClusterCycleInfo>& regions,
    int aggregate)
{
    std::vector<double> cluster_gaps;

    for (const auto& [cl_id, info] : regions) {
        if (info.logical_errors.empty()) continue;

        int min_hw = std::numeric_limits<int>::max();
        for (const auto& [z, flip_int] : info.logical_errors) {
            int hw = 0;
            for (uint8_t v : z) hw += v;
            if (hw < min_hw) min_hw = hw;
        }
        cluster_gaps.push_back(static_cast<double>(min_hw));
    }

    if (cluster_gaps.empty()) return std::numeric_limits<double>::infinity();
    if (aggregate == AGG_MIN)
        return *std::min_element(cluster_gaps.begin(), cluster_gaps.end());
    return std::accumulate(cluster_gaps.begin(), cluster_gaps.end(), 0.0);
}

// ---------------------------------------------------------------------------
// _gap_prior
//
// Per cluster: minimum prior weight over logical-error null-basis vectors.
// Prior weight of z = sum of weights_[fault_map[j]] for j where z[j] == 1.
// ---------------------------------------------------------------------------

double PriorGapEstimator::_gap_prior(
    const std::unordered_map<int, ClusterCycleInfo>& regions,
    int aggregate)
{
    std::vector<double> cluster_gaps;

    for (const auto& [cl_id, info] : regions) {
        if (info.logical_errors.empty()) continue;

        const ClusterStateOGB* cl = engine_.active_valid_clusters.at(cl_id);
        const auto& fault_map = cl->cluster_fault_idx_to_pcm_fault_idx;
        const int   n_local   = static_cast<int>(fault_map.size());

        double min_pw = std::numeric_limits<double>::infinity();
        for (const auto& [z, flip_int] : info.logical_errors) {
            double pw = 0.0;
            for (int j = 0; j < n_local; ++j)
                pw += weights_[fault_map[j]] * z[j];
            if (pw < min_pw) min_pw = pw;
        }
        cluster_gaps.push_back(min_pw);
    }

    if (cluster_gaps.empty()) return std::numeric_limits<double>::infinity();
    if (aggregate == AGG_MIN)
        return *std::min_element(cluster_gaps.begin(), cluster_gaps.end());
    return std::accumulate(cluster_gaps.begin(), cluster_gaps.end(), 0.0);
}

// ---------------------------------------------------------------------------
// _gap_weight_diff
//
// Per cluster: minimum signed weight difference over logical-error null-basis
// vectors.  For each z:
//   diff = weight(e1 ⊕ z) - weight(e1)
//        = Σ_j  w[j] * (1 - 2*e1[j]) * z[j]
//
// If asb=true, uses |diff| instead of the signed value.
// ---------------------------------------------------------------------------

double PriorGapEstimator::_gap_weight_diff(
    const std::unordered_map<int, ClusterCycleInfo>& regions,
    int aggregate, bool asb)
{
    std::vector<double> cluster_gaps;

    for (const auto& [cl_id, info] : regions) {
        if (info.logical_errors.empty()) continue;

        const ClusterStateOGB* cl = engine_.active_valid_clusters.at(cl_id);
        const auto& fault_map = cl->cluster_fault_idx_to_pcm_fault_idx;
        const int   n_local   = static_cast<int>(fault_map.size());

        std::vector<uint8_t> e1 = _get_correction(*cl);

        double min_diff = std::numeric_limits<double>::infinity();
        for (const auto& [z, flip_int] : info.logical_errors) {
            double diff = 0.0;
            for (int j = 0; j < n_local; ++j)
                diff += weights_[fault_map[j]] * (1.0 - 2.0 * e1[j]) * z[j];
            if (asb) diff = std::abs(diff);
            if (diff < min_diff) min_diff = diff;
        }
        cluster_gaps.push_back(min_diff);
    }

    if (cluster_gaps.empty()) return std::numeric_limits<double>::infinity();
    if (aggregate == AGG_MIN)
        return *std::min_element(cluster_gaps.begin(), cluster_gaps.end());
    return std::accumulate(cluster_gaps.begin(), cluster_gaps.end(), 0.0);
}

// ---------------------------------------------------------------------------
// _compute_gap
// ---------------------------------------------------------------------------

double PriorGapEstimator::_compute_gap(
    const std::unordered_map<int, ClusterCycleInfo>& regions,
    int gap_type, int aggregate, bool asb)
{
    switch (gap_type) {
        case GAP_BINARY:      return _gap_binary(regions);
        case GAP_HAMMING:     return _gap_hamming(regions, aggregate);
        case GAP_PRIOR:       return _gap_prior(regions, aggregate);
        case GAP_WEIGHT_DIFF: return _gap_weight_diff(regions, aggregate, asb);
        default:
            throw std::invalid_argument(
                "gap_type must be 0 (binary), 1 (hamming), "
                "2 (prior_weight), or 3 (weight_diff)");
    }
}

// ---------------------------------------------------------------------------
// get_cluster_solutions
//
// For every active valid cluster after run(): extract the RREF pivot
// correction e1, compute the logical flip via L_col_packed_local (the packed
// representation precomputed in _get_active_valid_clusters), and accumulate
// the global correction and overall logical flip.
//
// Must be called after execute() — reads the engine state left by the last run().
// ---------------------------------------------------------------------------

ClusterSolutions PriorGapEstimator::get_cluster_solutions()
{
    ClusterSolutions sol;
    sol.overall_solution.assign(n_fault_, 0);
    sol.overall_logical_flip.assign(n_logical_, 0);

    for (auto& [cl_id, cl_ptr] : engine_.active_valid_clusters) {
        const ClusterStateOGB& cl = *cl_ptr;
        std::vector<uint8_t>   e1 = _get_correction(cl);
        const int  n_local  = static_cast<int>(e1.size());
        const auto& fault_map = cl.cluster_fault_idx_to_pcm_fault_idx;

        // L_col_packed_local is only populated when n_logical_ > 0
        // (mirrors the guard in _get_active_valid_clusters).
        std::vector<uint8_t> flip(n_logical_, 0);
        if (n_logical_ > 0) {
            uint64_t flip_acc = 0;
            for (int j = 0; j < n_local; ++j)
                flip_acc ^= cl.L_col_packed_local[j] & -(uint64_t)(e1[j] & 1);
            for (int k = 0; k < n_logical_; ++k)
                flip[k] = static_cast<uint8_t>((flip_acc >> k) & 1);
        }

        sol.local_solutions[cl_id]          = e1;
        sol.logical_flip_per_cluster[cl_id] = flip;

        for (int j = 0; j < n_local; ++j)
            if (e1[j]) sol.overall_solution[fault_map[j]] = 1;

        for (int k = 0; k < n_logical_; ++k)
            sol.overall_logical_flip[k] ^= flip[k];
    }
    return sol;
}

// ---------------------------------------------------------------------------
// execute  (single shot)
// ---------------------------------------------------------------------------

ExecuteResult PriorGapEstimator::execute(
    const std::vector<uint8_t>& syndrome,
    int  gap_type,
    int  aggregate,
    int  over_grow_step,
    int  bits_per_step,
    bool asb,
    bool decode)
{
    auto regions = engine_.run_and_create_degenerate_cycle_regions(
        syndrome, over_grow_step, bits_per_step);

    int nonzero_count = 0;
    for (const auto& [cl_id, info] : regions)
        nonzero_count += static_cast<int>(info.logical_errors.size());

    ExecuteResult result;
    result.gap           = _compute_gap(regions, gap_type, aggregate, asb);
    result.nonzero_count = nonzero_count;

    if (decode) {
        ClusterSolutions sol       = get_cluster_solutions();
        result.overall_logical_flip = std::move(sol.overall_logical_flip);
    }
    return result;
}

// ---------------------------------------------------------------------------
// execute_batch
//
// data is row-major (n_shots × n_dets) uint8.
// A single syndrome buffer is pre-allocated and reused across shots to avoid
// per-shot heap allocations.
// ---------------------------------------------------------------------------

BatchResult PriorGapEstimator::execute_batch(
    const uint8_t* data,
    int  n_shots,
    int  n_dets,
    int  gap_type,
    int  aggregate,
    int  over_grow_step,
    int  bits_per_step,
    bool asb,
    bool decode)
{
    BatchResult batch;
    batch.gaps.resize(n_shots);
    batch.nonzero_counts.resize(n_shots);
    if (decode)
        batch.flips.assign(static_cast<size_t>(n_shots) * n_logical_, 0);

    std::vector<uint8_t> syn(n_dets);   // reused buffer; one allocation total
    for (int i = 0; i < n_shots; ++i) {
        std::copy(data + i * n_dets, data + (i + 1) * n_dets, syn.begin());

        ExecuteResult res = execute(syn, gap_type, aggregate,
                                    over_grow_step, bits_per_step, asb, decode);
        batch.gaps[i]          = res.gap;
        batch.nonzero_counts[i] = static_cast<int32_t>(res.nonzero_count);

        if (decode && !res.overall_logical_flip.empty()) {
            const int base = i * n_logical_;
            for (int k = 0; k < n_logical_; ++k)
                batch.flips[base + k] = res.overall_logical_flip[k];
        }
    }
    return batch;
}
