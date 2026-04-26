// pybind11 bindings for the C++ Clustering engine.
//
// Exposes the following classes in the _clustering_cpp extension module:
//
//   IncrementalRREF          — read-only RREF state (n_checks, n_bits, Z,
//                              s_prime, pivot_map, is_valid())
//   ClusterState             — read-only C++ fields + Python-settable .L
//                              (cluster_id, active, valid, fault_nodes,
//                               check_nodes, cluster_fault_idx_to_pcm_fault_idx,
//                               cluster_check_idx_to_pcm_check_idx, rref)
//   ClusteringEngine         — wraps Clustering (run(syndrome),
//                              active_valid_clusters, active_clusters)
//   ClusterStateOGB          — same as ClusterState + overgrow_budget field
//   ClusteringOvgBatchEngine — wraps ClusteringOvgBatch
//                              (run(syndrome, over_grow_step, bits_per_step),
//                               create_degenerate_cycle_regions(),
//                               run_and_create_degenerate_cycle_regions(),
//                               active_valid_clusters, active_clusters)
//   PriorGapEstimatorEngine  — wraps PriorGapEstimator
//                              (execute(syndrome, gap_type, aggregate, ...),
//                               execute_batch(syndromes, gap_type, aggregate, ...))
//
// Lifetime note: active_valid_clusters and active_clusters return references
// into the engine's internal cluster storage, valid until the next run().
//
// Build: see setup.py in the parent directory (prior_gap_estimation_cpp/).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "clustering.hpp"
#include "clustering_overgrow_batch.hpp"
#include "prior_gap_estimator.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helpers: convert gap_type / aggregate strings → int enums.
// Called once at the Python boundary so the hot path has no string comparisons.
// ---------------------------------------------------------------------------

static int gap_type_from_str(const std::string& s)
{
    if (s == "binary")       return GAP_BINARY;
    if (s == "hamming")      return GAP_HAMMING;
    if (s == "prior_weight") return GAP_PRIOR;
    if (s == "weight_diff")  return GAP_WEIGHT_DIFF;
    throw py::value_error(
        "gap_type must be 'binary', 'hamming', 'prior_weight', or "
        "'weight_diff'; got '" + s + "'");
}

static int aggregate_from_str(const std::string& s)
{
    if (s == "min") return AGG_MIN;
    if (s == "sum") return AGG_SUM;
    throw py::value_error(
        "aggregate must be 'min' or 'sum'; got '" + s + "'");
}

// ---------------------------------------------------------------------------
// Helper: convert ClusterCycleInfo map → Python dict matching the format of
// Python ClusteringOvergrowBatch.create_degenerate_cycle_regions():
//
//   { cluster_id: {'logical_error': [[z_arr, flip_int], ...],
//                  'stabilizer':    [[z_arr], ...]} }
// ---------------------------------------------------------------------------

static py::dict convert_cycle_regions(
    const std::unordered_map<int, ClusterCycleInfo>& regions)
{
    py::dict result;
    for (const auto& [cl_id, info] : regions) {
        py::list logical_errors;
        for (const auto& [z, flip_int] : info.logical_errors) {
            auto arr = py::array_t<uint8_t>(z.size());
            if (!z.empty())
                std::memcpy(arr.mutable_data(), z.data(), z.size());
            py::list entry;
            entry.append(arr);
            entry.append(py::int_(static_cast<long long>(flip_int)));
            logical_errors.append(std::move(entry));
        }
        py::list stabilizers;
        for (const auto& z : info.stabilizers) {
            auto arr = py::array_t<uint8_t>(z.size());
            if (!z.empty())
                std::memcpy(arr.mutable_data(), z.data(), z.size());
            py::list entry;
            entry.append(arr);
            stabilizers.append(std::move(entry));
        }
        py::dict cl_dict;
        cl_dict[py::str("logical_error")] = std::move(logical_errors);
        cl_dict[py::str("stabilizer")]    = std::move(stabilizers);
        result[py::int_(cl_id)] = std::move(cl_dict);
    }
    return result;
}

PYBIND11_MODULE(_clustering_cpp, m)
{
    m.doc() = "C++ Clustering engine — pybind11 bindings.";

    // -----------------------------------------------------------------------
    // IncrementalRREF — read-only view
    // -----------------------------------------------------------------------
    py::class_<IncrementalRREF>(m, "IncrementalRREF")
        .def_readonly("n_checks", &IncrementalRREF::n_checks)
        .def_readonly("n_bits",   &IncrementalRREF::n_bits)
        // Z: list of 1-D uint8 numpy arrays (each is a null-space basis vector).
        .def_property_readonly("Z",
            [](const IncrementalRREF& r) {
                py::list lst;
                for (const auto& z : r.Z) {
                    auto arr = py::array_t<uint8_t>(z.size());
                    if (!z.empty())
                        std::memcpy(arr.mutable_data(), z.data(), z.size());
                    lst.append(arr);
                }
                return lst;
            })
        // s_prime: 1-D uint8 numpy array (T_C @ syndrome, restricted to cluster).
        .def_property_readonly("s_prime",
            [](const IncrementalRREF& r) {
                auto arr = py::array_t<uint8_t>(r.s_prime.size());
                if (!r.s_prime.empty())
                    std::memcpy(arr.mutable_data(), r.s_prime.data(), r.s_prime.size());
                return arr;
            })
        // pivot_map: list[int], -1 means no pivot for that row.
        .def_property_readonly("pivot_map",
            [](const IncrementalRREF& r) { return r.pivot_map; })
        .def("is_valid", &IncrementalRREF::is_valid,
             "True iff the cluster syndrome lies in col(H_cluster).");

    // -----------------------------------------------------------------------
    // ClusterState — read-only C++ fields + dynamic Python attrs (e.g. .L)
    // -----------------------------------------------------------------------
    py::class_<ClusterState>(m, "ClusterState", py::dynamic_attr())
        .def_readonly("cluster_id", &ClusterState::cluster_id)
        .def_readonly("active",     &ClusterState::active)
        .def_readonly("valid",      &ClusterState::valid)
        .def_property_readonly("fault_nodes",
            [](const ClusterState& cl) { return cl.fault_nodes; })
        .def_property_readonly("check_nodes",
            [](const ClusterState& cl) { return cl.check_nodes; })
        .def_readonly("cluster_fault_idx_to_pcm_fault_idx",
                      &ClusterState::cluster_fault_idx_to_pcm_fault_idx)
        .def_readonly("cluster_check_idx_to_pcm_check_idx",
                      &ClusterState::cluster_check_idx_to_pcm_check_idx)
        // rref: returns a reference kept alive by this ClusterState.
        .def_property_readonly("rref",
            [](py::object self) {
                auto& cl = self.cast<ClusterState&>();
                return py::cast(&cl.rref,
                                py::return_value_policy::reference_internal,
                                self);
            });

    // -----------------------------------------------------------------------
    // ClusteringEngine — wraps Clustering
    // -----------------------------------------------------------------------
    py::class_<Clustering>(m, "ClusteringEngine")
        .def(py::init<int, int,
                      std::vector<std::vector<int>>,
                      std::vector<std::vector<int>>,
                      std::vector<double>>(),
             py::arg("n_det"), py::arg("n_fault"),
             py::arg("check_to_faults"), py::arg("fault_to_checks"),
             py::arg("weights"),
             R"doc(
Construct the C++ clustering engine.

Parameters
----------
n_det          : int
n_fault        : int
check_to_faults: list[list[int]]   check_to_faults[i] = fault indices for check i
fault_to_checks: list[list[int]]   fault_to_checks[j] = check indices for fault j
weights        : list[float]       log((1-p_j)/p_j) for each fault j
)doc")
        .def("run",
            [](Clustering& eng, py::array_t<uint8_t> syn)
            {
                auto buf = syn.request();
                std::vector<uint8_t> s(
                    static_cast<const uint8_t*>(buf.ptr),
                    static_cast<const uint8_t*>(buf.ptr) + buf.size);
                eng.run(s);
            },
            py::arg("syndrome"),
            "Run clustering for one syndrome vector (numpy uint8 array, length n_det).")
        // active_valid_clusters: dict[int, ClusterState] — references kept alive
        // by the engine.  Invalidated by the next call to run().
        .def_property_readonly("active_valid_clusters",
            [](py::object self) {
                auto& eng = self.cast<Clustering&>();
                py::dict d;
                for (auto& [cid, cl] : eng.active_valid_clusters)
                    d[py::int_(cid)] = py::cast(
                        cl,
                        py::return_value_policy::reference_internal,
                        self);
                return d;
            })
        // active_clusters: list[ClusterState] — all active clusters.
        // References kept alive by the engine.  Invalidated by next run().
        .def_property_readonly("active_clusters",
            [](py::object self) {
                auto& eng = self.cast<Clustering&>();
                py::list lst;
                for (const auto& up : eng.clusters())
                    if (up->active)
                        lst.append(py::cast(
                            up.get(),
                            py::return_value_policy::reference_internal,
                            self));
                return lst;
            });

    // -----------------------------------------------------------------------
    // ClusterStateOGB — read-only C++ fields + dynamic Python attrs (e.g. .L)
    // Same as ClusterState but with an additional overgrow_budget field.
    // -----------------------------------------------------------------------
    py::class_<ClusterStateOGB>(m, "ClusterStateOGB", py::dynamic_attr())
        .def_readonly("cluster_id",      &ClusterStateOGB::cluster_id)
        .def_readonly("active",          &ClusterStateOGB::active)
        .def_readonly("valid",           &ClusterStateOGB::valid)
        .def_readonly("overgrow_budget", &ClusterStateOGB::overgrow_budget)
        .def_property_readonly("fault_nodes",
            [](const ClusterStateOGB& cl) { return cl.fault_nodes; })
        .def_property_readonly("check_nodes",
            [](const ClusterStateOGB& cl) { return cl.check_nodes; })
        .def_readonly("cluster_fault_idx_to_pcm_fault_idx",
                      &ClusterStateOGB::cluster_fault_idx_to_pcm_fault_idx)
        .def_readonly("cluster_check_idx_to_pcm_check_idx",
                      &ClusterStateOGB::cluster_check_idx_to_pcm_check_idx)
        // rref: reference kept alive by this ClusterStateOGB.
        .def_property_readonly("rref",
            [](py::object self) {
                auto& cl = self.cast<ClusterStateOGB&>();
                return py::cast(&cl.rref,
                                py::return_value_policy::reference_internal,
                                self);
            });

    // -----------------------------------------------------------------------
    // ClusteringOvgBatchEngine — wraps ClusteringOvgBatch
    // -----------------------------------------------------------------------
    py::class_<ClusteringOvgBatch>(m, "ClusteringOvgBatchEngine")
        .def(py::init<int, int,
                      std::vector<std::vector<int>>,
                      std::vector<std::vector<int>>,
                      std::vector<double>,
                      int,
                      std::vector<std::vector<uint8_t>>>(),
             py::arg("n_det"), py::arg("n_fault"),
             py::arg("check_to_faults"), py::arg("fault_to_checks"),
             py::arg("weights"),
             py::arg("n_logical") = 0,
             py::arg("L") = std::vector<std::vector<uint8_t>>{},
             R"doc(
Construct the C++ clustering-with-overgrow-batch engine.

Parameters
----------
n_det            : int
n_fault          : int
check_to_faults  : list[list[int]]
fault_to_checks  : list[list[int]]
weights          : list[float]   log((1-p_j)/p_j) per fault
n_logical        : int, default 0   number of logical observables
L                : list[list[int]], default []
    Logical matrix, shape (n_logical, n_fault), row-major.
    Consumed into a precomputed column-packed form; the raw matrix is
    not retained.  Pass [] when no L is available; all null-space
    vectors will be classified as stabilizers.
)doc")
        .def("run",
            [](ClusteringOvgBatch& eng,
               py::array_t<uint8_t> syn,
               int over_grow_step,
               int bits_per_step)
            {
                auto buf = syn.request();
                std::vector<uint8_t> s(
                    static_cast<const uint8_t*>(buf.ptr),
                    static_cast<const uint8_t*>(buf.ptr) + buf.size);
                eng.run(s, over_grow_step, bits_per_step);
            },
            py::arg("syndrome"),
            py::arg("over_grow_step") = 0,
            py::arg("bits_per_step")  = 1,
            "Run clustering for one syndrome (numpy uint8 array, length n_det).")
        // active_valid_clusters: dict[int, ClusterStateOGB*]
        .def_property_readonly("active_valid_clusters",
            [](py::object self) {
                auto& eng = self.cast<ClusteringOvgBatch&>();
                py::dict d;
                for (auto& [cid, cl] : eng.active_valid_clusters)
                    d[py::int_(cid)] = py::cast(
                        cl,
                        py::return_value_policy::reference_internal,
                        self);
                return d;
            })
        // active_clusters: list[ClusterStateOGB*] — all active clusters.
        .def_property_readonly("active_clusters",
            [](py::object self) {
                auto& eng = self.cast<ClusteringOvgBatch&>();
                py::list lst;
                for (const auto& up : eng.clusters())
                    if (up->active)
                        lst.append(py::cast(
                            up.get(),
                            py::return_value_policy::reference_internal,
                            self));
                return lst;
            })
        // create_degenerate_cycle_regions(): classify null-space vectors.
        // Returns the same dict structure as the Python implementation.
        .def("create_degenerate_cycle_regions",
            [](const ClusteringOvgBatch& eng) {
                return convert_cycle_regions(
                    eng.create_degenerate_cycle_regions());
            },
            "Classify each null-space basis vector as a logical error or "
            "stabilizer.  Call after run().  Returns "
            "dict[int, dict[str, list]].")
        // run_and_create_degenerate_cycle_regions(): combined convenience.
        .def("run_and_create_degenerate_cycle_regions",
            [](ClusteringOvgBatch& eng,
               py::array_t<uint8_t> syn,
               int over_grow_step,
               int bits_per_step) {
                auto buf = syn.request();
                std::vector<uint8_t> s(
                    static_cast<const uint8_t*>(buf.ptr),
                    static_cast<const uint8_t*>(buf.ptr) + buf.size);
                return convert_cycle_regions(
                    eng.run_and_create_degenerate_cycle_regions(
                        s, over_grow_step, bits_per_step));
            },
            py::arg("syndrome"),
            py::arg("over_grow_step") = 0,
            py::arg("bits_per_step")  = 1,
            "run() then create_degenerate_cycle_regions() in one call.");

    // -----------------------------------------------------------------------
    // PriorGapEstimatorEngine — wraps PriorGapEstimator
    // -----------------------------------------------------------------------
    py::class_<PriorGapEstimator>(m, "PriorGapEstimatorEngine")
        .def(py::init<int, int,
                      std::vector<std::vector<int>>,
                      std::vector<std::vector<int>>,
                      std::vector<double>,
                      int,
                      std::vector<std::vector<uint8_t>>>(),
             py::arg("n_det"), py::arg("n_fault"),
             py::arg("check_to_faults"), py::arg("fault_to_checks"),
             py::arg("weights"),
             py::arg("n_logical") = 0,
             py::arg("L") = std::vector<std::vector<uint8_t>>{},
             R"doc(
Construct the gap estimator.

Parameters
----------
n_det, n_fault     : PCM dimensions
check_to_faults    : list[list[int]]
fault_to_checks    : list[list[int]]
weights            : list[float]   log((1-p_j)/p_j) per fault
n_logical          : int, default 0
L                  : list[list[int]], default []  logical matrix (n_logical, n_fault)
)doc")
        // ----------------------------------------------------------------
        // execute  — single shot
        // Returns (gap: float, nonzero_count: int, flip: np.ndarray | None)
        // flip is None when decode=False or n_logical=0.
        // ----------------------------------------------------------------
        .def("execute",
            [](PriorGapEstimator& est,
               py::array_t<uint8_t> syn,
               const std::string& gap_type,
               const std::string& aggregate,
               int  over_grow_step,
               int  bits_per_step,
               bool asb,
               bool decode) -> py::tuple
            {
                auto buf = syn.request();
                std::vector<uint8_t> s(
                    static_cast<const uint8_t*>(buf.ptr),
                    static_cast<const uint8_t*>(buf.ptr) + buf.size);

                ExecuteResult res = est.execute(
                    s,
                    gap_type_from_str(gap_type),
                    aggregate_from_str(aggregate),
                    over_grow_step, bits_per_step,
                    asb, decode);

                py::object flip;
                if (!res.overall_logical_flip.empty()) {
                    auto arr = py::array_t<uint8_t>(
                        static_cast<py::ssize_t>(res.overall_logical_flip.size()));
                    std::copy(res.overall_logical_flip.begin(),
                              res.overall_logical_flip.end(),
                              arr.mutable_data());
                    flip = arr;
                } else {
                    flip = py::none();
                }
                return py::make_tuple(res.gap, res.nonzero_count, flip);
            },
            py::arg("syndrome"),
            py::arg("gap_type")       = "binary",
            py::arg("aggregate")      = "min",
            py::arg("over_grow_step") = 0,
            py::arg("bits_per_step")  = 1,
            py::arg("asb")            = false,
            py::arg("decode")         = false,
            "Single-shot execute.  Returns (gap, nonzero_count, flip | None).")
        // ----------------------------------------------------------------
        // execute_batch  — batch
        // syndromes: uint8 (n_shots, n_dets), C-contiguous.
        // Returns (gaps float64 (n_shots,),
        //          nonzero_counts int32 (n_shots,),
        //          flips uint8 (n_shots, n_logical) | None)
        // flips is None when decode=False or n_logical=0.
        // ----------------------------------------------------------------
        .def("execute_batch",
            [](PriorGapEstimator& est,
               py::array_t<uint8_t,
                   py::array::c_style | py::array::forcecast> syndromes,
               const std::string& gap_type,
               const std::string& aggregate,
               int  over_grow_step,
               int  bits_per_step,
               bool asb,
               bool decode) -> py::tuple
            {
                auto buf = syndromes.request();
                if (buf.ndim != 2)
                    throw py::value_error("syndromes must be a 2-D array");
                const int n_shots = static_cast<int>(buf.shape[0]);
                const int n_dets  = static_cast<int>(buf.shape[1]);
                const uint8_t* data = static_cast<const uint8_t*>(buf.ptr);

                BatchResult batch = est.execute_batch(
                    data, n_shots, n_dets,
                    gap_type_from_str(gap_type),
                    aggregate_from_str(aggregate),
                    over_grow_step, bits_per_step,
                    asb, decode);

                auto gaps_arr = py::array_t<double>(n_shots);
                std::copy(batch.gaps.begin(), batch.gaps.end(),
                          gaps_arr.mutable_data());

                auto counts_arr = py::array_t<int32_t>(n_shots);
                std::copy(batch.nonzero_counts.begin(),
                          batch.nonzero_counts.end(),
                          counts_arr.mutable_data());

                py::object flips_obj;
                if (!batch.flips.empty()) {
                    const int n_logical =
                        static_cast<int>(batch.flips.size()) / n_shots;
                    auto flips_arr = py::array_t<uint8_t>(
                        {n_shots, n_logical});
                    std::copy(batch.flips.begin(), batch.flips.end(),
                              flips_arr.mutable_data());
                    flips_obj = flips_arr;
                } else {
                    flips_obj = py::none();
                }
                return py::make_tuple(gaps_arr, counts_arr, flips_obj);
            },
            py::arg("syndromes"),
            py::arg("gap_type")       = "binary",
            py::arg("aggregate")      = "min",
            py::arg("over_grow_step") = 0,
            py::arg("bits_per_step")  = 1,
            py::arg("asb")            = false,
            py::arg("decode")         = false,
            R"doc(
Batch execute.  syndromes must be uint8 with shape (n_shots, n_dets).
Returns (gaps, nonzero_counts, flips | None).
  gaps           : float64 (n_shots,)
  nonzero_counts : int32   (n_shots,)
  flips          : uint8   (n_shots, n_logical), or None when decode=False
                   or n_logical=0.
)doc");
}
