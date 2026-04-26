#pragma once

#include "clustering_overgrow_batch.hpp"

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Gap-type and aggregate constants.
// The binding layer converts Python strings to these ints so the hot path
// has no string comparisons.
// ---------------------------------------------------------------------------
static constexpr int GAP_BINARY      = 0;
static constexpr int GAP_HAMMING     = 1;
static constexpr int GAP_PRIOR       = 2;
static constexpr int GAP_WEIGHT_DIFF = 3;

static constexpr int AGG_MIN = 0;
static constexpr int AGG_SUM = 1;

// ---------------------------------------------------------------------------
// ClusterSolutions  — internal return type for get_cluster_solutions().
// Not exposed to Python.
// ---------------------------------------------------------------------------
struct ClusterSolutions {
    std::unordered_map<int, std::vector<uint8_t>> local_solutions;
    std::unordered_map<int, std::vector<uint8_t>> logical_flip_per_cluster;
    std::vector<uint8_t> overall_solution;       // length n_fault_
    std::vector<uint8_t> overall_logical_flip;   // length n_logical_
};

// ---------------------------------------------------------------------------
// ExecuteResult  — single-shot return bundle.
// overall_logical_flip is empty when decode=false.
// ---------------------------------------------------------------------------
struct ExecuteResult {
    double               gap;
    int                  nonzero_count;
    std::vector<uint8_t> overall_logical_flip;
};

// ---------------------------------------------------------------------------
// BatchResult  — batch return bundle.
// flips is flat row-major (n_shots * n_logical_); empty when decode=false.
// ---------------------------------------------------------------------------
struct BatchResult {
    std::vector<double>   gaps;
    std::vector<int32_t>  nonzero_counts;
    std::vector<uint8_t>  flips;
};

// ---------------------------------------------------------------------------
// PriorGapEstimator
//
// Wraps ClusteringOvgBatch and adds gap-computation logic.
// Mirrors Python PriorGapEstimatorUse.
//
// Constructor inputs (identical to ClusteringOvgBatch):
//   n_det, n_fault              : PCM dimensions
//   check_to_faults[i]          : fault indices adjacent to check i
//   fault_to_checks[j]          : check indices adjacent to fault j
//   weights[j]                  : log((1-p_j)/p_j)
//   n_logical                   : number of logical observables
//   L                           : logical matrix, shape (n_logical, n_fault), row-major
// ---------------------------------------------------------------------------
class PriorGapEstimator {
public:
    PriorGapEstimator(
        int n_det, int n_fault,
        std::vector<std::vector<int>>    check_to_faults,
        std::vector<std::vector<int>>    fault_to_checks,
        std::vector<double>              weights,
        int n_logical = 0,
        std::vector<std::vector<uint8_t>> L = {}
    );

    // Must be called after execute() — reads active_valid_clusters state
    // left by the last run().  Not exposed to Python.
    ClusterSolutions get_cluster_solutions();

    // Single-shot.  When decode=false, overall_logical_flip is empty.
    ExecuteResult execute(
        const std::vector<uint8_t>& syndrome,
        int  gap_type       = GAP_BINARY,
        int  aggregate      = AGG_MIN,
        int  over_grow_step = 0,
        int  bits_per_step  = 1,
        bool asb            = false,
        bool decode         = false
    );

    // Batch.  data is row-major (n_shots, n_dets) uint8.
    // flips in BatchResult is flat (n_shots * n_logical_); empty when decode=false.
    BatchResult execute_batch(
        const uint8_t* data,
        int  n_shots,
        int  n_dets,
        int  gap_type       = GAP_BINARY,
        int  aggregate      = AGG_MIN,
        int  over_grow_step = 0,
        int  bits_per_step  = 1,
        bool asb            = false,
        bool decode         = false
    );

private:
    ClusteringOvgBatch  engine_;
    std::vector<double> weights_;   // kept separately for gap arithmetic
    int                 n_logical_;
    int                 n_fault_;

    // Extract RREF pivot correction for one cluster (local fault index space).
    // pivot_map[i] == -1 means no pivot for row i.
    static std::vector<uint8_t> _get_correction(const ClusterStateOGB& cl);

    double _gap_binary(
        const std::unordered_map<int, ClusterCycleInfo>& regions);

    double _gap_hamming(
        const std::unordered_map<int, ClusterCycleInfo>& regions,
        int aggregate);

    double _gap_prior(
        const std::unordered_map<int, ClusterCycleInfo>& regions,
        int aggregate);

    double _gap_weight_diff(
        const std::unordered_map<int, ClusterCycleInfo>& regions,
        int aggregate, bool asb);

    double _compute_gap(
        const std::unordered_map<int, ClusterCycleInfo>& regions,
        int gap_type, int aggregate, bool asb);
};
