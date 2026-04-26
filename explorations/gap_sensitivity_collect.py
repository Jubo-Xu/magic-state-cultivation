import collections
import pathlib
import sys
import tempfile
from typing import Any

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import sinter
import stim

src_path = pathlib.Path(__file__).parent.parent / 'src'
assert src_path.exists()
sys.path.append(str(src_path))

import gen
import cultiv

# The class of the Gap collect
class GapSensitivityCollect:
    def __init__(
        self, 
        *,
        circuit: Optional[stim.Circuit] = None,
        circuit_generator: Optional[Any] = None
    ):
        if circuit_generator is not None:
            if getattr(circuit_generator, "ideal_circuit", None) is None:
                raise ValueError("circuit_generator.ideal_circuit is None. Call generate() first.")
            circuit = getattr(circuit_generator, "noisy_circuit", None) or circuit_generator.ideal_circuit
        if circuit is None:
            raise ValueError("Provide either circuit or circuit_generator.")
        
        self.circuit = circuit
        
        codes = gen.circuit_to_cycle_code_slices(circuit)
        self.ticks = codes.keys()
        self.codes = [code.with_transformed_coords(lambda e: e * (1 + 1j)) for code in codes.values()]

        self.single_dec_gap_scores = []
        self.overall_gap_scores = {
            "count": [],
            "sum_gap": [],
            "sum_err": [],
            "mean_gap_when_active": [],
            "error_rate_when_active": [],
        }
        self.partial_gap_sens_data: dict[str, Any] = {
            "remained_shots": 0,
            "complete_gaps": [],
            "partial_gaps": [],
            "closest_matches": [],
            "combos": [],
            "left_lengths": None,
            "top_lengths": None,
            "t_lengths": None,
            "mode": None,
            "stage_name": None,
            "complete_accept_and_reject": [],
            "closest_partial_accept_and_reject": [],
            "weighted_multi_partial_accept_and_reject": [],
            "pure_avg_accept_and_reject": [],
            "pure_avg_adjust_threshold_accept_and_reject": [],
            "gap_average_parameters": [],
        }
        self.best_combo_distribution: Optional[dict[str, Any]] = None

    # Helper: unpack one bit-packed detector row into 0/1 array up to num_detectors.
    def _unpack_det_row(self, row_packed: np.ndarray) -> np.ndarray:
        unpacked = np.unpackbits(np.asarray(row_packed, dtype=np.uint8), bitorder="little")
        return unpacked[: self.circuit.num_detectors].astype(np.uint8, copy=False)

    # Helper: detector density = number of non-trivial detectors in one packed row.
    def _det_density_from_packed_list(self, packed_list: list[int]) -> int:
        return int(np.count_nonzero(self._unpack_det_row(np.asarray(packed_list, dtype=np.uint8))))

    # NEW: total number of detectors selected by a packed mask (popcount of mask within num_detectors).
    def _mask_size(self, full_mask: np.ndarray) -> int:
        unpacked = np.unpackbits(full_mask, bitorder="little")
        return int(np.count_nonzero(unpacked[: self.circuit.num_detectors]))

    # Helper: container for false-shot diagnostics (accept-false / reject-false).
    def _empty_false_data_block(self, *, valid_best_combo: bool) -> dict[str, Any]:
        return {
            "best_combo": {
                "valid": bool(valid_best_combo),
                "det_density": [],
                "complete_det_density": [],
                "gap_value": [],
                "complete_gap_value": [],
                # --- NEW: pre-computed ratios (one value per false shot) ---
                "ratio_over_complete_nontrivial": [],   # det_density / complete_det_density
                "ratio_over_region_all": [],            # det_density / region_total_detectors
                "ratio_over_complete_all": [],          # det_density / circuit.num_detectors
            },
            "weighted_regions": {
                "det_density": [[], [], [], [], []],
                "complete_det_density": [],
                "gap_value": [[], [], [], [], [], []],  # 5 region gaps + 1 weighted average
                "complete_gap_value": [],
                # --- NEW: pre-computed ratios (list of 5 lists, one per region per false shot) ---
                "ratio_over_complete_nontrivial": [[], [], [], [], []],
                "ratio_over_region_all": [[], [], [], [], []],
                "ratio_over_complete_all": [[], [], [], [], []],
            },
            "pure_avg": {
                "det_density": [],
                "complete_det_density": [],
                "gap_value": [],
                "complete_gap_value": [],
                # --- NEW: pre-computed ratios (one value per false shot) ---
                "ratio_over_complete_nontrivial": [],
                "ratio_over_region_all": [],
                "ratio_over_complete_all": [],
            },
        }
        
    # Define the functions to collect the gap sensitivity data and generate the SVG plots.
    def collect_single_detector_gap_sens(self, sampler = cultiv.DesaturationSampler()):
        dec = sampler.compiled_sampler_for_task(sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model()))
        for d in range(self.circuit.num_detectors):
            self.single_dec_gap_scores.append(dec.decode_det_set({d}))
    
    def single_detector_gap_sens_svg(self, path: pathlib.Path):
        self.codes[0].write_svg(
            path,
            canvas_height=1000,
            other=self.codes[1:],
            title=[f'tick={e}' for e in sorted(self.ticks)],
            tile_color_func=lambda tile: GapSensitivityCollect.tile_coloring(tile, self.single_dec_gap_scores),
            show_coords=False,
            show_obs=False,
        )

    def single_detector_gap_sens_plot(self):
        """Return an inline-displayable SVG object for Jupyter notebooks."""
        if not self.single_dec_gap_scores:
            raise ValueError("No single-detector gap data collected. Call collect_single_detector_gap_sens() first.")
        return self._plot_from_svg_writer(lambda path: self.single_detector_gap_sens_svg(path))

    # Define the functions to collect the overall gap sensitivity data and generate the SVG plots.
    def collect_overall_gap_sens(self, shots: int, sampler = cultiv.DesaturationSampler()):
        if shots <= 0:
            raise ValueError("shots must be positive.")
        dec = sampler.compiled_sampler_for_task(
            sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model())
        )
        dets, actual_obs = dec.gap_circuit_sampler.sample(shots, separate_observables=True, bit_packed=True)
        keep_mask = ~np.any(dets & dec._discard_mask, axis=1)
        dets = dets[keep_mask]
        actual_obs = actual_obs[keep_mask]

        if dets.shape[0] == 0:
            num_dets = self.circuit.num_detectors
            self.overall_gap_scores = {
                "count": [0] * num_dets,
                "sum_gap": [0.0] * num_dets,
                "sum_err": [0.0] * num_dets,
                "mean_gap_when_active": [0.0] * num_dets,
                "error_rate_when_active": [0.0] * num_dets,
            }
            return

        predictions, gaps = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets.copy())
        if actual_obs.shape[1] != 1:
            raise ValueError(f"Expected exactly one observable, got shape={actual_obs.shape}.")
        actual_obs_bits = (actual_obs[:, 0] & 1).astype(np.bool_)
        errs = (predictions ^ actual_obs_bits).astype(np.float64)

        det_bits = np.unpackbits(dets, axis=1, bitorder='little')[:, :self.circuit.num_detectors].astype(np.float64)
        count = det_bits.sum(axis=0)
        sum_gap = det_bits.T @ gaps.astype(np.float64)
        sum_err = det_bits.T @ errs

        mean_gap_when_active = np.divide(sum_gap, count, out=np.zeros_like(sum_gap), where=count > 0)
        error_rate_when_active = np.divide(sum_err, count, out=np.zeros_like(sum_err), where=count > 0)

        self.overall_gap_scores = {
            "count": count.tolist(),
            "sum_gap": sum_gap.tolist(),
            "sum_err": sum_err.tolist(),
            "mean_gap_when_active": mean_gap_when_active.tolist(),
            "error_rate_when_active": error_rate_when_active.tolist(),
        }
    

    def overall_detector_gap_sens_svg(self, path: pathlib.Path):
        if not self.overall_gap_scores["mean_gap_when_active"]:
            raise ValueError("No overall gap data collected. Call collect_overall_gap_sens(shots=...) first.")
        gap_tile_color = self._make_tile_color_func(
            values=self.overall_gap_scores["mean_gap_when_active"],
            counts=self.overall_gap_scores["count"],
            invert=False,
            low_q=0.05,
            high_q=0.95,
            gamma=1.6,
        )
        self.codes[0].write_svg(
            path,
            canvas_height=1000,
            other=self.codes[1:],
            title=[f'tick={e}' for e in sorted(self.ticks)],
            tile_color_func=gap_tile_color,
            show_coords=False,
            show_obs=False,
        )
        self._append_svg_legend(
            path,
            title="Overall Gap Sensitivity",
            low_label="black = low gap (bad)",
            high_label="red = high gap (good)",
            no_data_label="gray = no data/postselected",
            red_is_high=True,
        )

    def overall_detector_gap_sens_plot(self):
        if not self.overall_gap_scores["mean_gap_when_active"]:
            raise ValueError("No overall gap data collected. Call collect_overall_gap_sens(shots=...) first.")
        return self._plot_from_svg_writer(lambda path: self.overall_detector_gap_sens_svg(path))

    def overall_detector_err_sens_svg(self, path: pathlib.Path):
        if not self.overall_gap_scores["error_rate_when_active"]:
            raise ValueError("No overall gap data collected. Call collect_overall_gap_sens(shots=...) first.")
        err_tile_color = self._make_tile_color_func(
            values=self.overall_gap_scores["error_rate_when_active"],
            counts=self.overall_gap_scores["count"],
            invert=True,
            low_q=0.00,
            high_q=0.99,
            gamma=1.0,
        )
        self.codes[0].write_svg(
            path,
            canvas_height=1000,
            other=self.codes[1:],
            title=[f'tick={e}' for e in sorted(self.ticks)],
            tile_color_func=err_tile_color,
            show_coords=False,
            show_obs=False,
        )
        self._append_svg_legend(
            path,
            title="Overall Error Sensitivity",
            low_label="red = low error rate (good)",
            high_label="black = high error rate (bad)",
            no_data_label="gray = no data/postselected",
            red_is_high=False,
        )

    def overall_detector_err_sens_plot(self):
        if not self.overall_gap_scores["error_rate_when_active"]:
            raise ValueError("No overall gap data collected. Call collect_overall_gap_sens(shots=...) first.")
        return self._plot_from_svg_writer(lambda path: self.overall_detector_err_sens_svg(path))
    
    
    ## Collection and Visualization for Partial Detector Gap Sensitivity
    def collect_partial_detector_gap_sens(
        self,
        shots: int,
        sampler = cultiv.DesaturationSampler(),
        selector = None,
        left_lengths: tuple[int, int] = (1, 1),
        top_lengths: tuple[int, int] = (1, 1),
        t_lengths: tuple[int, int] = (1, 1),
        mode: str = "rectangle",
        stage_name: str = "escape",
        check_bestcombo: bool = True,
        gap_average_parameters: Optional[list[float]] = None,
        left_len_avg: int = 7,
        top_len_avg: int = 8,
        t_avg: int = 5,
        mode_avg: str = "rectangle",
    ):
        if selector is None:
            raise ValueError("selector is required and should be an instance of DetectorSelection3D.")
        if shots <= 0:
            raise ValueError("shots must be positive.")
        l0, l1 = left_lengths
        u0, u1 = top_lengths
        t0, t1 = t_lengths
        if not (l1 >= l0 and u1 >= u0 and t1 >= t0):
            raise ValueError("Each *_lengths tuple must be (start, end) with end >= start.")
        
        if gap_average_parameters is None:
            gap_average_parameters = [1.0, 0.0, 0.0, 0.0, 0.0]
        if len(gap_average_parameters) != 5:
            raise ValueError("gap_average_parameters must have length 5.")
        gap_average_parameters = [float(x) for x in gap_average_parameters]

        dec = sampler.compiled_sampler_for_task(
            sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model())
        )
        dets, actual_obs = dec.gap_circuit_sampler.sample(shots, separate_observables=True, bit_packed=True)
        keep_mask = ~np.any(dets & dec._discard_mask, axis=1)
        dets = dets[keep_mask]
        actual_obs = actual_obs[keep_mask]
        remained_shots = int(dets.shape[0])

        if remained_shots == 0:
            self.partial_gap_sens_data = {
                "remained_shots": 0,
                "complete_gaps": [],
                "partial_gaps": [],
                "closest_matches": [],
                "combos": [],
                "left_lengths": left_lengths,
                "top_lengths": top_lengths,
                "t_lengths": t_lengths,
                "mode": mode,
                "stage_name": stage_name,
                "complete_accept_and_reject": [],
                "closest_partial_accept_and_reject": [],
                "weighted_multi_partial_accept_and_reject": [],
                "pure_avg_accept_and_reject": [],
                "pure_avg_adjust_threshold_accept_and_reject": [],
                "gap_average_parameters": gap_average_parameters,
                "check_bestcombo": bool(check_bestcombo),
            }
            self.best_combo_distribution = None
            return

        _, complete_gaps = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets.copy())
        complete_gaps = complete_gaps.astype(np.float64)
        
        
        # Calculate the gaps for best combo and multi masks
        combos = [
            (left_len, top_len, t)
            for left_len in range(l0, l1 + 1)
            for top_len in range(u0, u1 + 1)
            for t in range(t0, t1 + 1)
        ]

        combo_to_gaps: dict[tuple[int, int, int], np.ndarray] = {}
        combo_to_full_mask: dict[tuple[int, int, int], np.ndarray] = {}
        if check_bestcombo:
            for left_len, top_len, t in combos:
                _, mask_packed = selector.build_color_region_mask(
                    left_len=left_len,
                    top_len=top_len,
                    t=t,
                    mode=mode,
                    stage_name=stage_name,
                )
                full_mask = np.zeros(dets.shape[1], dtype=np.uint8)
                copy_n = min(len(mask_packed), len(full_mask))
                full_mask[:copy_n] = mask_packed[:copy_n]
                combo_to_full_mask[(left_len, top_len, t)] = full_mask.copy()

                dets_partial = dets.copy()
                dets_partial &= full_mask.reshape(1, -1)
                _, partial_gaps = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets_partial)
                combo_to_gaps[(left_len, top_len, t)] = partial_gaps.astype(np.float64)

        # NEW: popcount of each combo mask = total detectors selected by that region.
        combo_to_region_size: dict[tuple[int, int, int], int] = {}
        if check_bestcombo:
            for combo, mask in combo_to_full_mask.items():
                combo_to_region_size[combo] = self._mask_size(mask)

        # Build reusable masks once for weighted-region and pure-avg paths.
        _, avg_masks_packed, _ = selector.build_partial_region_multi_mask(
            left_len=left_len_avg,
            top_len=top_len_avg,
            t=t_avg,
            mode=mode_avg,
            stage_name=stage_name,
        )
        avg_masks_full: list[np.ndarray] = []
        for mask_packed_avg in avg_masks_packed:
            full_mask = np.zeros(dets.shape[1], dtype=np.uint8)
            copy_n = min(len(mask_packed_avg), len(full_mask))
            full_mask[:copy_n] = mask_packed_avg[:copy_n]
            avg_masks_full.append(full_mask)
        # NEW: total detectors selected by each of the 5 weighted regions (constant across shots).
        avg_region_sizes: list[int] = [self._mask_size(m) for m in avg_masks_full]

        _, mask_packed_pure_avg = selector.build_color_region_mask(
            left_len=left_len_avg,
            top_len=top_len_avg,
            t=t_avg,
            mode=mode_avg,
            stage_name=stage_name,
        )
        pure_avg_full_mask = np.zeros(dets.shape[1], dtype=np.uint8)
        copy_n_pure_avg = min(len(mask_packed_pure_avg), len(pure_avg_full_mask))
        pure_avg_full_mask[:copy_n_pure_avg] = mask_packed_pure_avg[:copy_n_pure_avg]
        # NEW: total detectors selected by the pure-avg region (constant across shots).
        pure_avg_region_size: int = self._mask_size(pure_avg_full_mask)

        partial_gaps_per_shot: list[dict[tuple[int, int, int], float]] = []
        closest_matches: list[dict[str, Any]] = []
        
        for shot_idx in range(remained_shots):
            shot_det = dets[shot_idx:shot_idx + 1].copy()
            if check_bestcombo:
                gap_map: dict[tuple[int, int, int], float] = {}
                for combo in combos:
                    gap_map[combo] = float(combo_to_gaps[combo][shot_idx])
                partial_gaps_per_shot.append(gap_map)
            else:
                gap_map = {}
                partial_gaps_per_shot.append({})
            complete_gap = float(complete_gaps[shot_idx])
            if check_bestcombo:
                best_combo = min(combos, key=lambda c: abs(gap_map[c] - complete_gap))
                best_partial = float(gap_map[best_combo])
                best_abs_diff = abs(best_partial - complete_gap)
            else:
                best_combo = (np.nan, np.nan, np.nan)
                best_partial = np.nan
                best_abs_diff = np.nan
            complete_det_packed = shot_det[0].astype(np.uint8).tolist()

            # Best-combo selected detectors (invalid/all-zero when check_bestcombo=False).
            if check_bestcombo:
                best_mask_full = combo_to_full_mask[tuple(best_combo)]
                dets_best_partial = shot_det.copy()
                dets_best_partial &= best_mask_full.reshape(1, -1)
                best_partial_det_packed = dets_best_partial[0].astype(np.uint8).tolist()
            else:
                best_partial_det_packed = [0 for _ in range(shot_det.shape[1])]

            region_gaps_vals: list[float] = []
            region_det_packed: list[list[int]] = []
            for full_mask in avg_masks_full:
                dets_partial = shot_det.copy()
                dets_partial &= full_mask.reshape(1, -1)
                _, rg = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets_partial)
                region_gaps_vals.append(float(rg[0]))
                region_det_packed.append(dets_partial[0].astype(np.uint8).tolist())

            weighted_components = np.asarray(region_gaps_vals, dtype=np.float64)
            weighted_avg_gap = float(np.dot(np.asarray(gap_average_parameters, dtype=np.float64), weighted_components))
            weighted_abs_diff = abs(weighted_avg_gap - complete_gap)
            
            # Calculate the gap for pure-average selected region.
            dets_partial_pure_avg = shot_det.copy()
            dets_partial_pure_avg &= pure_avg_full_mask.reshape(1, -1)
            _, partial_gaps_pure_avg = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets_partial_pure_avg)
            
            closest_matches.append({
                "shot_idx": int(shot_idx),
                "combo": tuple(best_combo),
                "partial_gap": best_partial,
                "complete_gap": complete_gap,
                "abs_diff": best_abs_diff,
                "weighted_avg_gap": weighted_avg_gap,
                "weighted_abs_diff": weighted_abs_diff,
                "weighted_region_gaps": weighted_components.tolist(),
                "pure_avg_gap": partial_gaps_pure_avg[0],
                "pure_avg_abs_diff": abs(partial_gaps_pure_avg[0] - complete_gap),
                # Newly added detector snapshots for downstream false-shot analysis.
                "complete_det_packed": complete_det_packed,
                "best_combo_det_packed": best_partial_det_packed,
                "weighted_region_det_packed": region_det_packed,
                "pure_avg_det_packed": dets_partial_pure_avg[0].astype(np.uint8).tolist(),
                # NEW: total detectors in the best-combo region for this shot (varies per shot).
                "best_combo_region_size": combo_to_region_size.get(tuple(best_combo), 0) if check_bestcombo else 0,
            })
        self.partial_gap_sens_data = {
            "remained_shots": remained_shots,
            "complete_gaps": complete_gaps.tolist(),
            "partial_gaps": partial_gaps_per_shot,
            "closest_matches": closest_matches,
            "combos": combos,
            "left_lengths": left_lengths,
            "top_lengths": top_lengths,
            "t_lengths": t_lengths,
            "mode": mode,
            "stage_name": stage_name,
            "complete_accept_and_reject": [],
            "closest_partial_accept_and_reject": [],
            "weighted_multi_partial_accept_and_reject": [],
            "pure_avg_accept_and_reject": [],
            "pure_avg_adjust_threshold_accept_and_reject": [],
            "gap_average_parameters": gap_average_parameters,
            "check_bestcombo": bool(check_bestcombo),
            # NEW: region sizes (mask popcounts) for ratio_over_region_all computation.
            "avg_region_sizes": avg_region_sizes,
            "pure_avg_region_size": pure_avg_region_size,
        }
        self.best_combo_distribution = None

    def collect_partial_accpet_and_reject(
        self,
        *,
        gap_threshold: Optional[float] = None,
        gap_threshold_pure_avg: Optional[float] = None,
    ) -> None:
        if not self.partial_gap_sens_data.get("closest_matches"):
            self.partial_gap_sens_data["complete_accept_and_reject"] = []
            self.partial_gap_sens_data["closest_partial_accept_and_reject"] = []
            self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"] = []
            self.partial_gap_sens_data["pure_avg_accept_and_reject"] = []
            self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"] = []
            if gap_threshold is not None:
                self.partial_gap_sens_data["gap_threshold"] = float(gap_threshold)
            if gap_threshold_pure_avg is not None:
                self.partial_gap_sens_data["gap_threshold_pure_avg"] = float(gap_threshold_pure_avg)
            return

        if gap_threshold is None:
            gap_threshold = float(self.partial_gap_sens_data.get("gap_threshold", 0.0))
        if gap_threshold_pure_avg is None:
            gap_threshold_pure_avg = float(self.partial_gap_sens_data.get("gap_threshold_pure_avg", gap_threshold))

        complete_accept_and_reject: list[bool] = []
        closest_partial_accept_and_reject: list[bool] = []
        weighted_multi_partial_accept_and_reject: list[bool] = []
        pure_avg_accept_and_reject: list[bool] = []
        pure_avg_adjust_threshold_accept_and_reject: list[bool] = []
        for rec in self.partial_gap_sens_data["closest_matches"]:
            complete_gap = float(rec["complete_gap"])
            partial_gap = float(rec["partial_gap"])
            weighted_avg_gap = float(rec["weighted_avg_gap"])
            pure_avg_gap = float(rec["pure_avg_gap"])

            complete_accept_and_reject.append(complete_gap >= gap_threshold)
            closest_partial_accept_and_reject.append(partial_gap >= gap_threshold)
            weighted_multi_partial_accept_and_reject.append(weighted_avg_gap >= gap_threshold)
            pure_avg_accept_and_reject.append(pure_avg_gap >= gap_threshold)
            pure_avg_adjust_threshold_accept_and_reject.append(pure_avg_gap >= gap_threshold_pure_avg)

        self.partial_gap_sens_data["complete_accept_and_reject"] = complete_accept_and_reject
        self.partial_gap_sens_data["closest_partial_accept_and_reject"] = closest_partial_accept_and_reject
        self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"] = weighted_multi_partial_accept_and_reject
        self.partial_gap_sens_data["pure_avg_accept_and_reject"] = pure_avg_accept_and_reject
        self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"] = pure_avg_adjust_threshold_accept_and_reject
        self.partial_gap_sens_data["gap_threshold"] = float(gap_threshold)
        self.partial_gap_sens_data["gap_threshold_pure_avg"] = float(gap_threshold_pure_avg)
        self.best_combo_distribution = None

    def paritial_detector_gap_sens_svg(self, shot_idx: int, out: Optional[pathlib.Path] = None, interactive: bool = True):
        if not self.partial_gap_sens_data["partial_gaps"]:
            raise ValueError("No partial gap data found. Call collect_partial_detector_gap_sens(...) first.")
        remained = self.partial_gap_sens_data["remained_shots"]
        if shot_idx < 0 or shot_idx >= remained:
            raise ValueError(f"shot_idx out of range: 0 <= shot_idx < {remained}.")

        shot_partial: dict[tuple[int, int, int], float] = self.partial_gap_sens_data["partial_gaps"][shot_idx]
        complete_gap = float(self.partial_gap_sens_data["complete_gaps"][shot_idx])

        combos = list(shot_partial.keys())
        x = [c[0] for c in combos]
        y = [c[1] for c in combos]
        z = [c[2] for c in combos]
        partial_vals = [float(shot_partial[c]) for c in combos]

        all_vals = np.asarray(partial_vals + [complete_gap], dtype=np.float64)
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        if vmax <= vmin:
            vmax = vmin + 1e-9

        def scale_size(v: float, lo: float = 4.0, hi: float = 24.0) -> float:
            norm = (v - vmin) / (vmax - vmin)
            return float(lo + (hi - lo) * norm)

        partial_sizes = [scale_size(v) for v in partial_vals]
        complete_size = scale_size(complete_gap)
        complete_sizes = [complete_size] * len(combos)

        hover_black = [
            f"(left,top,t)=({a},{b},{c})<br>partial_gap={g:.4f}"
            for (a, b, c), g in zip(combos, partial_vals)
        ]
        hover_red = [
            f"(left,top,t)=({a},{b},{c})<br>complete_gap={complete_gap:.4f}"
            for (a, b, c) in combos
        ]
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                name="Partial Gap",
                marker=dict(size=partial_sizes, color="black", symbol="circle", opacity=0.92),
                text=hover_black,
                hoverinfo="text",
            ))
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                name="Complete Gap (Reference)",
                marker=dict(size=complete_sizes, color="rgba(0,0,0,0)", symbol="circle-open", line=dict(color="red", width=2)),
                text=hover_red,
                hoverinfo="text",
            ))
            fig.update_layout(
                title=f"Shot {shot_idx}: Partial vs Complete Gap",
                scene=dict(
                    xaxis_title="left_len",
                    yaxis_title="top_len",
                    zaxis_title="t",
                ),
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            if out is not None:
                if out.suffix.lower() == ".html":
                    fig.write_html(str(out))
                elif out.suffix.lower() == ".svg":
                    fig.write_image(str(out))
                else:
                    fig.write_html(str(out.with_suffix(".html")))
            return fig
        except ImportError as ex:
            if interactive:
                raise ImportError(
                    "Interactive 3D rotation requires plotly. Install it (e.g. `pip install plotly`) "
                    "or call with interactive=False for static matplotlib output."
                ) from ex
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x, y, z, s=partial_sizes, c="black", alpha=0.92, label="Partial Gap")
            ax.scatter(x, y, z, s=complete_sizes, facecolors="none", edgecolors="red", linewidths=1.5, label="Complete Gap (Reference)")
            ax.set_xlabel("left_len")
            ax.set_ylabel("top_len")
            ax.set_zlabel("t")
            ax.set_title(f"Shot {shot_idx}: Partial vs Complete Gap")
            ax.legend(loc="upper left")
            if out is not None:
                if out.suffix.lower() == ".svg":
                    fig.savefig(out, format="svg", bbox_inches="tight")
                else:
                    fig.savefig(out.with_suffix(".svg"), format="svg", bbox_inches="tight")
            return fig

    def partial_detector_gap_sens_svg(self, shot_idx: int, out: Optional[pathlib.Path] = None):
        # Alias with corrected spelling.
        return self.paritial_detector_gap_sens_svg(shot_idx=shot_idx, out=out)

    def print_partial_gap_match_for_shot(self, shot_idx: int) -> dict[str, Any]:
        if not self.partial_gap_sens_data["closest_matches"]:
            raise ValueError("No partial gap match data found. Call collect_partial_detector_gap_sens(...) first.")
        remained = self.partial_gap_sens_data["remained_shots"]
        if shot_idx < 0 or shot_idx >= remained:
            raise ValueError(f"shot_idx out of range: 0 <= shot_idx < {remained}.")
        rec = self.partial_gap_sens_data["closest_matches"][shot_idx]
        left_len, top_len, t = rec["combo"]
        print(
            f"shot_idx={rec['shot_idx']}, "
            f"closest_combo=(left_len={left_len}, top_len={top_len}, t={t}), "
            f"partial_gap={rec['partial_gap']:.6f}, "
            f"complete_gap={rec['complete_gap']:.6f}, "
            f"abs_diff={rec['abs_diff']:.6f}"
        )
        return rec


    def calculate_partial_gap_best_combo_distribution(self) -> None:
        if not self.partial_gap_sens_data["closest_matches"]:
            raise ValueError("No partial gap data found. Call collect_partial_detector_gap_sens(...) first.")

        best = self.partial_gap_sens_data["closest_matches"]
        remained = int(self.partial_gap_sens_data["remained_shots"])
        check_bestcombo = bool(self.partial_gap_sens_data.get("check_bestcombo", True))
        if remained == 0:
            self.best_combo_distribution = {
                "remained_shots": 0,
                "mean_combo": {"left_len": 0.0, "top_len": 0.0, "t": 0.0},
                "var_combo": {"left_len": 0.0, "top_len": 0.0, "t": 0.0},
                "std_combo": {"left_len": 0.0, "top_len": 0.0, "t": 0.0},
                "mean_abs_diff": 0.0,
                "var_abs_diff": 0.0,
                "std_abs_diff": 0.0,
                "mean_complete_gap": 0.0,
                "best_combo_samples": [],
                "abs_diff_samples": [],
                "complete_accept_rate": 1.0,
                "complete_reject_rate": 0.0,
                "closest_partial_accept_rate": 1.0,
                "closest_partial_reject_rate": 0.0,
                "partial_accept_false_num": 0.0,
                "partial_reject_false_num": 0.0,
                "partial_accept_false_rate(over_remained_shots)": 0.0,
                "partial_reject_false_rate(over_remained_shots)": 0.0,
                "partial_accept_false_rate(over_complete_accept_num)": 0.0,
                "partial_reject_false_rate(over_complete_reject_num)": 0.0,
                "weighted_mean_abs_diff": 0.0,
                "weighted_var_abs_diff": 0.0,
                "weighted_std_abs_diff": 0.0,
                "weighted_multi_partial_accept_rate": 1.0,
                "weighted_multi_partial_reject_rate": 0.0,
                "weighted_accept_false_num": 0.0,
                "weighted_reject_false_num": 0.0,
                "weighted_accept_false_rate(over_remained_shots)": 0.0,
                "weighted_reject_false_rate(over_remained_shots)": 0.0,
                "weighted_accept_false_rate(over_complete_accept_num)": 0.0,
                "weighted_reject_false_rate(over_complete_reject_num)": 0.0,
                "pure_avg_mean_abs_diff": 0.0,
                "pure_avg_var_abs_diff": 0.0,
                "pure_avg_std_abs_diff": 0.0,
                "pure_avg_accept_rate": 1.0,
                "pure_avg_reject_rate": 0.0,
                "pure_avg_accept_false_num": 0.0,
                "pure_avg_reject_false_num": 0.0,
                "pure_avg_accept_false_rate(over_remained_shots)": 0.0,
                "pure_avg_reject_false_rate(over_remained_shots)": 0.0,
                "pure_avg_accept_false_rate(over_complete_accept_num)": 0.0,
                "pure_avg_reject_false_rate(over_complete_reject_num)": 0.0,
                "pure_avg_adjust_threshold_accept_rate": 1.0,
                "pure_avg_adjust_threshold_reject_rate": 0.0,
                "pure_avg_adjust_threshold_accept_false_num": 0.0,
                "pure_avg_adjust_threshold_reject_false_num": 0.0,
                "pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)": 0.0,
                "pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)": 0.0,
                "pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)": 0.0,
                "pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)": 0.0,
                "accept_false_data": self._empty_false_data_block(valid_best_combo=check_bestcombo),
                "reject_false_data": self._empty_false_data_block(valid_best_combo=check_bestcombo),
                "check_bestcombo": check_bestcombo,
            }
            return

        if check_bestcombo:
            left_vals = np.asarray([float(rec["combo"][0]) for rec in best], dtype=np.float64)
            top_vals = np.asarray([float(rec["combo"][1]) for rec in best], dtype=np.float64)
            t_vals = np.asarray([float(rec["combo"][2]) for rec in best], dtype=np.float64)
            abs_diff_vals = np.asarray([float(rec["abs_diff"]) for rec in best], dtype=np.float64)
        else:
            left_vals = np.asarray([], dtype=np.float64)
            top_vals = np.asarray([], dtype=np.float64)
            t_vals = np.asarray([], dtype=np.float64)
            abs_diff_vals = np.asarray([], dtype=np.float64)
        weighted_abs_diff_vals = np.asarray([float(rec["weighted_abs_diff"]) for rec in best], dtype=np.float64)
        pure_avg_abs_diff_vals = np.asarray([float(rec["pure_avg_abs_diff"]) for rec in best], dtype=np.float64)
        complete = np.asarray(self.partial_gap_sens_data["complete_gaps"], dtype=np.float64)
        
        assert len(self.partial_gap_sens_data["complete_accept_and_reject"]) == remained
        assert len(self.partial_gap_sens_data["closest_partial_accept_and_reject"]) == remained
        assert len(self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"]) == remained
        assert len(self.partial_gap_sens_data["pure_avg_accept_and_reject"]) == remained
        assert len(self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"]) == remained
        
        complete_accept_num = 0
        complete_reject_num = 0
        partial_accept_num = 0
        partial_reject_num = 0
        partial_accept_false_num = 0
        partial_reject_false_num = 0
        weighted_accept_num = 0
        weighted_reject_num = 0
        weighted_accept_false_num = 0
        weighted_reject_false_num = 0
        pure_avg_accept_num = 0
        pure_avg_reject_num = 0
        pure_avg_accept_false_num = 0
        pure_avg_reject_false_num = 0
        pure_avg_adjust_threshold_accept_num = 0
        pure_avg_adjust_threshold_reject_num = 0
        pure_avg_adjust_threshold_accept_false_num = 0
        pure_avg_adjust_threshold_reject_false_num = 0
        for i in range(remained):
            if self.partial_gap_sens_data["complete_accept_and_reject"][i]:
                complete_accept_num += 1
            else:
                complete_reject_num += 1
            if check_bestcombo:
                if self.partial_gap_sens_data["closest_partial_accept_and_reject"][i]:
                    partial_accept_num += 1
                else:
                    partial_reject_num += 1
                if self.partial_gap_sens_data["complete_accept_and_reject"][i] and (not self.partial_gap_sens_data["closest_partial_accept_and_reject"][i]):
                    partial_reject_false_num += 1
                if (not self.partial_gap_sens_data["complete_accept_and_reject"][i]) and self.partial_gap_sens_data["closest_partial_accept_and_reject"][i]:
                    partial_accept_false_num += 1

            if self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"][i]:
                weighted_accept_num += 1
            else:
                weighted_reject_num += 1
            if self.partial_gap_sens_data["complete_accept_and_reject"][i] and (not self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"][i]):
                weighted_reject_false_num += 1
            if (not self.partial_gap_sens_data["complete_accept_and_reject"][i]) and self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"][i]:
                weighted_accept_false_num += 1
            
            if self.partial_gap_sens_data["pure_avg_accept_and_reject"][i]:
                pure_avg_accept_num += 1
            else:
                pure_avg_reject_num += 1
            if self.partial_gap_sens_data["complete_accept_and_reject"][i] and (not self.partial_gap_sens_data["pure_avg_accept_and_reject"][i]):
                pure_avg_reject_false_num += 1
            if (not self.partial_gap_sens_data["complete_accept_and_reject"][i]) and self.partial_gap_sens_data["pure_avg_accept_and_reject"][i]:
                pure_avg_accept_false_num += 1
            
            if self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"][i]:
                pure_avg_adjust_threshold_accept_num += 1
            else:
                pure_avg_adjust_threshold_reject_num += 1
            if self.partial_gap_sens_data["complete_accept_and_reject"][i] and (not self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"][i]):
                pure_avg_adjust_threshold_reject_false_num += 1
            if (not self.partial_gap_sens_data["complete_accept_and_reject"][i]) and self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"][i]:
                pure_avg_adjust_threshold_accept_false_num += 1

        # Newly added: gather false-shot detector-density/gap diagnostics by selection type.
        # NEW: read region sizes for ratio_over_region_all; get total detector count for ratio_over_complete_all.
        complete_total_detectors: int = self.circuit.num_detectors
        avg_region_sizes: list[int] = self.partial_gap_sens_data.get("avg_region_sizes", [0, 0, 0, 0, 0])
        pure_avg_region_size: int = int(self.partial_gap_sens_data.get("pure_avg_region_size", 0))

        accept_false_data = self._empty_false_data_block(valid_best_combo=check_bestcombo)
        reject_false_data = self._empty_false_data_block(valid_best_combo=check_bestcombo)
        for i, rec in enumerate(best):
            complete_accept = bool(self.partial_gap_sens_data["complete_accept_and_reject"][i])
            partial_accept = bool(self.partial_gap_sens_data["closest_partial_accept_and_reject"][i])
            weighted_accept = bool(self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"][i])
            pure_accept = bool(self.partial_gap_sens_data["pure_avg_accept_and_reject"][i])

            complete_density = self._det_density_from_packed_list(rec["complete_det_packed"])
            complete_gap_value = float(rec["complete_gap"])

            # Best-combo false data (invalid when check_bestcombo=False).
            if check_bestcombo:
                best_density = self._det_density_from_packed_list(rec["best_combo_det_packed"])
                best_gap_value = float(rec["partial_gap"])
                # NEW: denominators for the three ratios for this shot's best-combo region.
                best_region_size = int(rec.get("best_combo_region_size", 0))
                best_r1 = best_density / complete_density if complete_density > 0 else 0.0
                best_r2 = best_density / best_region_size if best_region_size > 0 else 0.0
                best_r3 = best_density / complete_total_detectors if complete_total_detectors > 0 else 0.0
                if partial_accept and (not complete_accept):
                    accept_false_data["best_combo"]["det_density"].append(best_density)
                    accept_false_data["best_combo"]["complete_det_density"].append(complete_density)
                    accept_false_data["best_combo"]["gap_value"].append(best_gap_value)
                    accept_false_data["best_combo"]["complete_gap_value"].append(complete_gap_value)
                    # NEW
                    accept_false_data["best_combo"]["ratio_over_complete_nontrivial"].append(best_r1)
                    accept_false_data["best_combo"]["ratio_over_region_all"].append(best_r2)
                    accept_false_data["best_combo"]["ratio_over_complete_all"].append(best_r3)
                if (not partial_accept) and complete_accept:
                    reject_false_data["best_combo"]["det_density"].append(best_density)
                    reject_false_data["best_combo"]["complete_det_density"].append(complete_density)
                    reject_false_data["best_combo"]["gap_value"].append(best_gap_value)
                    reject_false_data["best_combo"]["complete_gap_value"].append(complete_gap_value)
                    # NEW
                    reject_false_data["best_combo"]["ratio_over_complete_nontrivial"].append(best_r1)
                    reject_false_data["best_combo"]["ratio_over_region_all"].append(best_r2)
                    reject_false_data["best_combo"]["ratio_over_complete_all"].append(best_r3)

            # Weighted-regions false data.
            weighted_region_density = [
                self._det_density_from_packed_list(packed) for packed in rec["weighted_region_det_packed"]
            ]
            weighted_region_gaps = [float(v) for v in rec["weighted_region_gaps"]]
            weighted_all_gaps = weighted_region_gaps + [float(rec["weighted_avg_gap"])]
            # NEW: compute three ratios per region.
            weighted_r1 = [
                (weighted_region_density[ri] / complete_density if complete_density > 0 else 0.0)
                for ri in range(5)
            ]
            weighted_r2 = [
                (weighted_region_density[ri] / avg_region_sizes[ri] if avg_region_sizes[ri] > 0 else 0.0)
                for ri in range(5)
            ]
            weighted_r3 = [
                (weighted_region_density[ri] / complete_total_detectors if complete_total_detectors > 0 else 0.0)
                for ri in range(5)
            ]
            if weighted_accept and (not complete_accept):
                for region_i in range(5):
                    accept_false_data["weighted_regions"]["det_density"][region_i].append(weighted_region_density[region_i])
                    # NEW
                    accept_false_data["weighted_regions"]["ratio_over_complete_nontrivial"][region_i].append(weighted_r1[region_i])
                    accept_false_data["weighted_regions"]["ratio_over_region_all"][region_i].append(weighted_r2[region_i])
                    accept_false_data["weighted_regions"]["ratio_over_complete_all"][region_i].append(weighted_r3[region_i])
                for gap_i in range(6):
                    accept_false_data["weighted_regions"]["gap_value"][gap_i].append(weighted_all_gaps[gap_i])
                accept_false_data["weighted_regions"]["complete_det_density"].append(complete_density)
                accept_false_data["weighted_regions"]["complete_gap_value"].append(complete_gap_value)
            if (not weighted_accept) and complete_accept:
                for region_i in range(5):
                    reject_false_data["weighted_regions"]["det_density"][region_i].append(weighted_region_density[region_i])
                    # NEW
                    reject_false_data["weighted_regions"]["ratio_over_complete_nontrivial"][region_i].append(weighted_r1[region_i])
                    reject_false_data["weighted_regions"]["ratio_over_region_all"][region_i].append(weighted_r2[region_i])
                    reject_false_data["weighted_regions"]["ratio_over_complete_all"][region_i].append(weighted_r3[region_i])
                for gap_i in range(6):
                    reject_false_data["weighted_regions"]["gap_value"][gap_i].append(weighted_all_gaps[gap_i])
                reject_false_data["weighted_regions"]["complete_det_density"].append(complete_density)
                reject_false_data["weighted_regions"]["complete_gap_value"].append(complete_gap_value)

            # Pure-avg false data.
            pure_density = self._det_density_from_packed_list(rec["pure_avg_det_packed"])
            pure_gap_value = float(rec["pure_avg_gap"])
            # NEW: compute three ratios for pure-avg region.
            pure_r1 = pure_density / complete_density if complete_density > 0 else 0.0
            pure_r2 = pure_density / pure_avg_region_size if pure_avg_region_size > 0 else 0.0
            pure_r3 = pure_density / complete_total_detectors if complete_total_detectors > 0 else 0.0
            if pure_accept and (not complete_accept):
                accept_false_data["pure_avg"]["det_density"].append(pure_density)
                accept_false_data["pure_avg"]["complete_det_density"].append(complete_density)
                accept_false_data["pure_avg"]["gap_value"].append(pure_gap_value)
                accept_false_data["pure_avg"]["complete_gap_value"].append(complete_gap_value)
                # NEW
                accept_false_data["pure_avg"]["ratio_over_complete_nontrivial"].append(pure_r1)
                accept_false_data["pure_avg"]["ratio_over_region_all"].append(pure_r2)
                accept_false_data["pure_avg"]["ratio_over_complete_all"].append(pure_r3)
            if (not pure_accept) and complete_accept:
                reject_false_data["pure_avg"]["det_density"].append(pure_density)
                reject_false_data["pure_avg"]["complete_det_density"].append(complete_density)
                reject_false_data["pure_avg"]["gap_value"].append(pure_gap_value)
                reject_false_data["pure_avg"]["complete_gap_value"].append(complete_gap_value)
                # NEW
                reject_false_data["pure_avg"]["ratio_over_complete_nontrivial"].append(pure_r1)
                reject_false_data["pure_avg"]["ratio_over_region_all"].append(pure_r2)
                reject_false_data["pure_avg"]["ratio_over_complete_all"].append(pure_r3)

        self.best_combo_distribution = {
            "remained_shots": remained,
            "mean_combo": {
                "left_len": float(np.mean(left_vals)) if check_bestcombo else None,
                "top_len": float(np.mean(top_vals)) if check_bestcombo else None,
                "t": float(np.mean(t_vals)) if check_bestcombo else None,
            },
            "var_combo": {
                "left_len": float(np.var(left_vals)) if check_bestcombo else None,
                "top_len": float(np.var(top_vals)) if check_bestcombo else None,
                "t": float(np.var(t_vals)) if check_bestcombo else None,
            },
            "std_combo": {
                "left_len": float(np.std(left_vals)) if check_bestcombo else None,
                "top_len": float(np.std(top_vals)) if check_bestcombo else None,
                "t": float(np.std(t_vals)) if check_bestcombo else None,
            },
            "mean_abs_diff": float(np.mean(abs_diff_vals)) if check_bestcombo else None,
            "var_abs_diff": float(np.var(abs_diff_vals)) if check_bestcombo else None,
            "std_abs_diff": float(np.std(abs_diff_vals)) if check_bestcombo else None,
            "mean_complete_gap": float(np.mean(complete)) if complete.size else 0.0,
            "best_combo_samples": [tuple(rec["combo"]) for rec in best] if check_bestcombo else [],
            "abs_diff_samples": abs_diff_vals.tolist() if check_bestcombo else [],
            "complete_accept_rate": float(complete_accept_num) / remained,
            "complete_reject_rate": float(complete_reject_num) / remained,
            "closest_partial_accept_rate": (float(partial_accept_num) / remained) if check_bestcombo else None,
            "closest_partial_reject_rate": (float(partial_reject_num) / remained) if check_bestcombo else None,
            "partial_accept_false_num": partial_accept_false_num if check_bestcombo else None,
            "partial_reject_false_num": partial_reject_false_num if check_bestcombo else None,
            "partial_accept_false_rate(over_remained_shots)": (float(partial_accept_false_num) / remained) if check_bestcombo else None,
            "partial_reject_false_rate(over_remained_shots)": (float(partial_reject_false_num) / remained) if check_bestcombo else None,
            "partial_accept_false_rate(over_complete_accept_num)": (float(partial_accept_false_num) / complete_accept_num if complete_accept_num > 0 else 0.0) if check_bestcombo else None,
            "partial_reject_false_rate(over_complete_reject_num)": (float(partial_reject_false_num) / complete_reject_num if complete_reject_num > 0 else 0.0) if check_bestcombo else None,
            "weighted_mean_abs_diff": float(np.mean(weighted_abs_diff_vals)),
            "weighted_var_abs_diff": float(np.var(weighted_abs_diff_vals)),
            "weighted_std_abs_diff": float(np.std(weighted_abs_diff_vals)),
            "weighted_multi_partial_accept_rate": float(weighted_accept_num) / remained,
            "weighted_multi_partial_reject_rate": float(weighted_reject_num) / remained,
            "weighted_accept_false_num": weighted_accept_false_num,
            "weighted_reject_false_num": weighted_reject_false_num,
            "weighted_accept_false_rate(over_remained_shots)": float(weighted_accept_false_num) / remained,
            "weighted_reject_false_rate(over_remained_shots)": float(weighted_reject_false_num) / remained,
            "weighted_accept_false_rate(over_complete_accept_num)": float(weighted_accept_false_num) / complete_accept_num if complete_accept_num > 0 else 0.0,
            "weighted_reject_false_rate(over_complete_reject_num)": float(weighted_reject_false_num) / complete_reject_num if complete_reject_num > 0 else 0.0,
            "pure_avg_mean_abs_diff": float(np.mean(pure_avg_abs_diff_vals)),
            "pure_avg_var_abs_diff": float(np.var(pure_avg_abs_diff_vals)),
            "pure_avg_std_abs_diff": float(np.std(pure_avg_abs_diff_vals)),
            "pure_avg_accept_rate": float(pure_avg_accept_num) / remained,
            "pure_avg_reject_rate": float(pure_avg_reject_num) / remained,
            "pure_avg_accept_false_num": pure_avg_accept_false_num,
            "pure_avg_reject_false_num": pure_avg_reject_false_num,
            "pure_avg_accept_false_rate(over_remained_shots)": float(pure_avg_accept_false_num) / remained,
            "pure_avg_reject_false_rate(over_remained_shots)": float(pure_avg_reject_false_num) / remained,
            "pure_avg_accept_false_rate(over_complete_accept_num)": float(pure_avg_accept_false_num) / complete_accept_num if complete_accept_num > 0 else 0.0,
            "pure_avg_reject_false_rate(over_complete_reject_num)": float(pure_avg_reject_false_num) / complete_reject_num if complete_reject_num > 0 else 0.0,
            "pure_avg_adjust_threshold_accept_rate": float(pure_avg_adjust_threshold_accept_num) / remained,
            "pure_avg_adjust_threshold_reject_rate": float(pure_avg_adjust_threshold_reject_num) / remained,
            "pure_avg_adjust_threshold_accept_false_num": pure_avg_adjust_threshold_accept_false_num,
            "pure_avg_adjust_threshold_reject_false_num": pure_avg_adjust_threshold_reject_false_num,
            "pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)": float(pure_avg_adjust_threshold_accept_false_num) / remained,
            "pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)": float(pure_avg_adjust_threshold_reject_false_num) / remained,
            "pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)": float(pure_avg_adjust_threshold_accept_false_num) / complete_accept_num if complete_accept_num > 0 else 0.0,
            "pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)": float(pure_avg_adjust_threshold_reject_false_num) / complete_reject_num if complete_reject_num > 0 else 0.0,
            "accept_false_data": accept_false_data,
            "reject_false_data": reject_false_data,
            "check_bestcombo": check_bestcombo,
            # NEW: total detector count used as denominator for ratio_over_complete_all.
            "complete_total_detectors": complete_total_detectors,
        }

    def print_partial_gap_best_combo_distribution(self) -> dict[str, Any]:
        if self.best_combo_distribution is None:
            raise ValueError(
                "No best combo distribution found. Call calculate_partial_gap_best_combo_distribution() first."
            )
        d = self.best_combo_distribution
        check_bestcombo = bool(d.get("check_bestcombo", True))
        print(f"remained_shots={d['remained_shots']}")
        if check_bestcombo:
            print(
                "mean_combo="
                f"(left_len={d['mean_combo']['left_len']:.6f}, "
                f"top_len={d['mean_combo']['top_len']:.6f}, "
                f"t={d['mean_combo']['t']:.6f})"
            )
            print(
                "var_combo="
                f"(left_len={d['var_combo']['left_len']:.6f}, "
                f"top_len={d['var_combo']['top_len']:.6f}, "
                f"t={d['var_combo']['t']:.6f})"
            )
            print(
                "std_combo="
                f"(left_len={d['std_combo']['left_len']:.6f}, "
                f"top_len={d['std_combo']['top_len']:.6f}, "
                f"t={d['std_combo']['t']:.6f})"
            )
            print(f"mean_abs_diff={d['mean_abs_diff']:.6f}")
            print(f"var_abs_diff={d['var_abs_diff']:.6f}")
            print(f"std_abs_diff={d['std_abs_diff']:.6f}")
        else:
            print("best_combo_metrics=SKIPPED (check_bestcombo=False)")
        print(f"mean_complete_gap={d['mean_complete_gap']:.6f}")
        print(f"complete_accept_rate={d['complete_accept_rate']:.4f}")
        print(f"complete_reject_rate={d['complete_reject_rate']:.4f}")
        if check_bestcombo:
            print(f"closest_partial_accept_rate={d['closest_partial_accept_rate']:.4f}")
            print(f"closest_partial_reject_rate={d['closest_partial_reject_rate']:.4f}")
            print(f"partial_accept_false_num={d['partial_accept_false_num']}")
            print(f"partial_reject_false_num={d['partial_reject_false_num']}")
        else:
            print("closest_partial_metrics=SKIPPED (check_bestcombo=False)")
        # print(f"partial_accept_false_rate(over_remained_shots)={d['partial_accept_false_rate(over_remained_shots)']:.4f}")
        # print(f"partial_reject_false_rate(over_remained_shots)={d['partial_reject_false_rate(over_remained_shots)']:.4f}")
        # print(f"partial_accept_false_rate(over_complete_accept_num)={d['partial_accept_false_rate(over_complete_accept_num)']:.4f}")
        # print(f"partial_reject_false_rate(over_complete_reject_num)={d['partial_reject_false_rate(over_complete_reject_num)']:.4f}")
        print(f"weighted_mean_abs_diff={d['weighted_mean_abs_diff']:.6f}")
        print(f"weighted_var_abs_diff={d['weighted_var_abs_diff']:.6f}")
        print(f"weighted_std_abs_diff={d['weighted_std_abs_diff']:.6f}")
        print(f"weighted_multi_partial_accept_rate={d['weighted_multi_partial_accept_rate']:.4f}")
        print(f"weighted_multi_partial_reject_rate={d['weighted_multi_partial_reject_rate']:.4f}")
        print(f"weighted_accept_false_num={d['weighted_accept_false_num']}")
        print(f"weighted_reject_false_num={d['weighted_reject_false_num']}")
        # print(f"weighted_accept_false_rate(over_remained_shots)={d['weighted_accept_false_rate(over_remained_shots)']:.4f}")
        # print(f"weighted_reject_false_rate(over_remained_shots)={d['weighted_reject_false_rate(over_remained_shots)']:.4f}")
        # print(f"weighted_accept_false_rate(over_complete_accept_num)={d['weighted_accept_false_rate(over_complete_accept_num)']:.4f}")
        # print(f"weighted_reject_false_rate(over_complete_reject_num)={d['weighted_reject_false_rate(over_complete_reject_num)']:.4f}")
        print(f"pure_avg_mean_abs_diff={d['pure_avg_mean_abs_diff']:.6f}")
        print(f"pure_avg_var_abs_diff={d['pure_avg_var_abs_diff']:.6f}")
        print(f"pure_avg_std_abs_diff={d['pure_avg_std_abs_diff']:.6f}")
        print(f"pure_avg_accept_rate={d['pure_avg_accept_rate']:.4f}")
        print(f"pure_avg_reject_rate={d['pure_avg_reject_rate']:.4f}")
        print(f"pure_avg_accept_false_num={d['pure_avg_accept_false_num']}")
        print(f"pure_avg_reject_false_num={d['pure_avg_reject_false_num']}")
        print(f"pure_avg_adjust_threshold_accept_rate={d['pure_avg_adjust_threshold_accept_rate']:.4f}")
        print(f"pure_avg_adjust_threshold_reject_rate={d['pure_avg_adjust_threshold_reject_rate']:.4f}")
        print(f"pure_avg_adjust_threshold_accept_false_num={d['pure_avg_adjust_threshold_accept_false_num']}")
        print(f"pure_avg_adjust_threshold_reject_false_num={d['pure_avg_adjust_threshold_reject_false_num']}")
        # print(f"pure_avg_accept_false_rate(over_remained_shots)={d['pure_avg_accept_false_rate(over_remained_shots)']:.4f}")
        # print(f"pure_avg_reject_false_rate(over_remained_shots)={d['pure_avg_reject_false_rate(over_remained_shots)']:.4f}")
        # print(f"pure_avg_accept_false_rate(over_complete_accept_num)={d['pure_avg_accept_false_rate(over_complete_accept_num)']:.4f}")
        # print(f"pure_avg_reject_false_rate(over_complete_reject_num)={d['pure_avg_reject_false_rate(over_complete_reject_num)']:.4f}")
        # print(f"pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)={d['pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)']:.4f}")
        # print(f"pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)={d['pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)']:.4f}")
        # print(f"pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)={d['pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)']:.4f}")
        # print(f"pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)={d['pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)']:.4f}")

        return {
            "remained_shots": d["remained_shots"],
            "mean_combo": d["mean_combo"],
            "var_combo": d["var_combo"],
            "std_combo": d["std_combo"],
            "mean_abs_diff": d["mean_abs_diff"],
            "var_abs_diff": d["var_abs_diff"],
            "std_abs_diff": d["std_abs_diff"],
            "mean_complete_gap": d["mean_complete_gap"],
            "complete_accept_rate": d["complete_accept_rate"],
            "complete_reject_rate": d["complete_reject_rate"],
            "closest_partial_accept_rate": d["closest_partial_accept_rate"],
            "closest_partial_reject_rate": d["closest_partial_reject_rate"],
            "partial_accept_false_num": d["partial_accept_false_num"],
            "partial_reject_false_num": d["partial_reject_false_num"],
            "partial_accept_false_rate(over_remained_shots)": d["partial_accept_false_rate(over_remained_shots)"],
            "partial_reject_false_rate(over_remained_shots)": d["partial_reject_false_rate(over_remained_shots)"],
            "partial_accept_false_rate(over_complete_accept_num)": d["partial_accept_false_rate(over_complete_accept_num)"],
            "partial_reject_false_rate(over_complete_reject_num)": d["partial_reject_false_rate(over_complete_reject_num)"],
            "weighted_mean_abs_diff": d["weighted_mean_abs_diff"],
            "weighted_var_abs_diff": d["weighted_var_abs_diff"],
            "weighted_std_abs_diff": d["weighted_std_abs_diff"],
            "weighted_multi_partial_accept_rate": d["weighted_multi_partial_accept_rate"],
            "weighted_multi_partial_reject_rate": d["weighted_multi_partial_reject_rate"],
            "weighted_accept_false_num": d["weighted_accept_false_num"],
            "weighted_reject_false_num": d["weighted_reject_false_num"],
            "weighted_accept_false_rate(over_remained_shots)": d["weighted_accept_false_rate(over_remained_shots)"],
            "weighted_reject_false_rate(over_remained_shots)": d["weighted_reject_false_rate(over_remained_shots)"],
            "weighted_accept_false_rate(over_complete_accept_num)": d["weighted_accept_false_rate(over_complete_accept_num)"],
            "weighted_reject_false_rate(over_complete_reject_num)": d["weighted_reject_false_rate(over_complete_reject_num)"],
            "pure_avg_mean_abs_diff": d["pure_avg_mean_abs_diff"],
            "pure_avg_var_abs_diff": d["pure_avg_var_abs_diff"],
            "pure_avg_std_abs_diff": d["pure_avg_std_abs_diff"],
            "pure_avg_accept_rate": d["pure_avg_accept_rate"],
            "pure_avg_reject_rate": d["pure_avg_reject_rate"],
            "pure_avg_accept_false_num": d["pure_avg_accept_false_num"],
            "pure_avg_reject_false_num": d["pure_avg_reject_false_num"],
            "pure_avg_accept_false_rate(over_remained_shots)": d["pure_avg_accept_false_rate(over_remained_shots)"],
            "pure_avg_reject_false_rate(over_remained_shots)": d["pure_avg_reject_false_rate(over_remained_shots)"],
            "pure_avg_accept_false_rate(over_complete_accept_num)": d["pure_avg_accept_false_rate(over_complete_accept_num)"],
            "pure_avg_reject_false_rate(over_complete_reject_num)": d["pure_avg_reject_false_rate(over_complete_reject_num)"],
            "pure_avg_adjust_threshold_accept_rate": d["pure_avg_adjust_threshold_accept_rate"],
            "pure_avg_adjust_threshold_reject_rate": d["pure_avg_adjust_threshold_reject_rate"],
            "pure_avg_adjust_threshold_accept_false_num": d["pure_avg_adjust_threshold_accept_false_num"],
            "pure_avg_adjust_threshold_reject_false_num": d["pure_avg_adjust_threshold_reject_false_num"],
            "pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)": d["pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)"],
            "pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)": d["pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)"],
            "pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)": d["pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)"],
            "pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)": d["pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)"],
            "accept_false_data": d["accept_false_data"],
            "reject_false_data": d["reject_false_data"],
        }

    def false_accept_reject_detail_plot(
        self,
        *,
        select_type: str = "best_combo",
        first_shots_num: Optional[int] = None,
        path: Optional[pathlib.Path] = None,
        # NEW: if not None, replaces the detector-density panels with the chosen pre-computed ratio.
        # Options: "over_complete_nontrivial", "over_region_all", "over_complete_all".
        ratio_type: Optional[str] = None,
    ):
        if self.best_combo_distribution is None:
            raise ValueError("No best combo distribution found. Call calculate_partial_gap_best_combo_distribution() first.")
        if select_type not in {"best_combo", "weighted_regions", "pure_avg"}:
            raise ValueError("select_type must be one of: 'best_combo', 'weighted_regions', 'pure_avg'.")
        # NEW: validate ratio_type.
        _valid_ratio_types = {"over_complete_nontrivial", "over_region_all", "over_complete_all"}
        if ratio_type is not None and ratio_type not in _valid_ratio_types:
            raise ValueError(f"ratio_type must be one of {_valid_ratio_types} or None.")

        data_accept = self.best_combo_distribution["accept_false_data"][select_type]
        data_reject = self.best_combo_distribution["reject_false_data"][select_type]
        if first_shots_num is not None and first_shots_num <= 0:
            raise ValueError("first_shots_num must be positive when provided.")

        # NEW: determine which data field and y-axis label to use for the density panels.
        # ratio_type is e.g. "over_complete_nontrivial"; the stored key has a "ratio_" prefix.
        density_field = f"ratio_{ratio_type}" if ratio_type is not None else "det_density"
        density_ylabel = ratio_type if ratio_type is not None else "Detector density"

        fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=120)
        if select_type == "weighted_regions":
            from matplotlib.patches import Patch

            density_colors = ["#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2"]
            gap_colors = ["#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2", "#EDC948"]
            shot_sep = 2.0  # Extra separation between false shots.

            # Newly added weighted plot mode: no spacing inside one shot-group, only between shots.
            def plot_weighted(ax, data: dict[str, Any], *, is_gap: bool, title: str) -> None:
                n_total = len(data["complete_gap_value"] if is_gap else data["complete_det_density"])
                n = min(n_total, int(first_shots_num)) if first_shots_num is not None else n_total
                cat_n = 6 if is_gap else 5
                if n == 0:
                    ax.set_title(f"{select_type} | {title} (no data)")
                    ax.grid(True, axis="y", alpha=0.25)
                    return

                xs: list[float] = []
                masked_vals: list[float] = []
                complete_vals: list[float] = []
                color_vals: list[str] = []
                centers: list[float] = []
                for i in range(n):
                    base = i * (cat_n + shot_sep)
                    centers.append(base + (cat_n - 1) / 2.0)
                    for c in range(cat_n):
                        x = base + c
                        xs.append(x)
                        if is_gap:
                            masked_vals.append(float(data["gap_value"][c][i]))
                            complete_vals.append(float(data["complete_gap_value"][i]))
                            color_vals.append(gap_colors[c])
                        else:
                            # NEW: use density_field to select det_density or a pre-computed ratio.
                            masked_vals.append(float(data[density_field][c][i]))
                            complete_vals.append(float(data["complete_det_density"][i]))
                            color_vals.append(density_colors[c])

                # NEW: only draw the complete reference bar when showing raw density (ratio plots omit it
                # because the complete bar would use a different denominator and be misleading).
                if not is_gap and ratio_type is None:
                    ax.bar(xs, complete_vals, width=0.95, color="#4c78a8", alpha=0.35, label="Complete")
                elif is_gap:
                    ax.bar(xs, complete_vals, width=0.95, color="#4c78a8", alpha=0.35, label="Complete")
                ax.bar(xs, masked_vals, width=0.72, color=color_vals, alpha=0.85)
                ax.set_title(f"{select_type} | {title}")
                ax.set_xticks(centers)
                ax.set_xticklabels([f"s{i}" for i in range(n)], rotation=0, fontsize=8)
                ax.grid(True, axis="y", alpha=0.25)

                if is_gap:
                    handles = [
                        Patch(color="#4c78a8", alpha=0.35, label="Complete"),
                        Patch(color=gap_colors[0], label="Region1 gap"),
                        Patch(color=gap_colors[1], label="Region2 gap"),
                        Patch(color=gap_colors[2], label="Region3 gap"),
                        Patch(color=gap_colors[3], label="Region4 gap"),
                        Patch(color=gap_colors[4], label="Region5 gap"),
                        Patch(color=gap_colors[5], label="Weighted avg gap"),
                    ]
                else:
                    # NEW: legend label changes based on whether we're showing density or a ratio.
                    region_label = "density" if ratio_type is None else ratio_type
                    handles_list = [] if ratio_type is not None else [Patch(color="#4c78a8", alpha=0.35, label="Complete")]
                    handles_list += [
                        Patch(color=density_colors[0], label=f"Region1 {region_label}"),
                        Patch(color=density_colors[1], label=f"Region2 {region_label}"),
                        Patch(color=density_colors[2], label=f"Region3 {region_label}"),
                        Patch(color=density_colors[3], label=f"Region4 {region_label}"),
                        Patch(color=density_colors[4], label=f"Region5 {region_label}"),
                    ]
                    handles = handles_list
                ax.legend(handles=handles, loc="upper right", fontsize=8)

            plot_weighted(axes[0, 0], data_accept, is_gap=False, title=f"Accept False: {density_ylabel}")
            plot_weighted(axes[0, 1], data_accept, is_gap=True, title="Accept False: Gap")
            plot_weighted(axes[1, 0], data_reject, is_gap=False, title=f"Reject False: {density_ylabel}")
            plot_weighted(axes[1, 1], data_reject, is_gap=True, title="Reject False: Gap")
        else:
            # Build plotting vectors (x labels + masked/complete values) for non-weighted select types.
            # NEW: build_density_points reads from density_field instead of always "det_density".
            def build_density_points(data: dict[str, Any]) -> tuple[list[str], list[float], list[float]]:
                labels: list[str] = []
                masked: list[float] = []
                complete: list[float] = []
                n_total = len(data["complete_det_density"])
                n = min(n_total, int(first_shots_num)) if first_shots_num is not None else n_total
                for i in range(n):
                    labels.append(f"s{i}")
                    masked.append(float(data[density_field][i]))
                    complete.append(float(data["complete_det_density"][i]))
                return labels, masked, complete

            def build_gap_points(data: dict[str, Any]) -> tuple[list[str], list[float], list[float]]:
                labels: list[str] = []
                masked: list[float] = []
                complete: list[float] = []
                n_total = len(data["complete_gap_value"])
                n = min(n_total, int(first_shots_num)) if first_shots_num is not None else n_total
                for i in range(n):
                    labels.append(f"s{i}")
                    masked.append(float(data["gap_value"][i]))
                    complete.append(float(data["complete_gap_value"][i]))
                return labels, masked, complete

            a_dx, a_dm, a_dc = build_density_points(data_accept)
            a_gx, a_gm, a_gc = build_gap_points(data_accept)
            r_dx, r_dm, r_dc = build_density_points(data_reject)
            r_gx, r_gm, r_gc = build_gap_points(data_reject)

            density_panels = [
                (axes[0, 0], a_dx, a_dm, a_dc, f"Accept False: {density_ylabel}"),
                (axes[1, 0], r_dx, r_dm, r_dc, f"Reject False: {density_ylabel}"),
            ]
            gap_panels = [
                (axes[0, 1], a_gx, a_gm, a_gc, "Accept False: Gap"),
                (axes[1, 1], r_gx, r_gm, r_gc, "Reject False: Gap"),
            ]
            for ax, labels, masked_vals, complete_vals, title in density_panels:
                x = np.arange(len(labels), dtype=np.float64)
                # NEW: only draw complete reference bar when showing raw density.
                if ratio_type is None:
                    ax.bar(x, complete_vals, width=0.80, color="#4c78a8", alpha=0.40, label="Complete")
                ax.bar(x, masked_vals, width=0.58, color="#f28e2b", alpha=0.75, label=select_type)
                ax.set_title(f"{select_type} | {title}")
                ax.set_xticks(x)
                if len(labels) <= 60:
                    ax.set_xticklabels(labels, rotation=90, fontsize=7)
                else:
                    ax.set_xticklabels([])
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(loc="upper right", fontsize=8)
            for ax, labels, masked_vals, complete_vals, title in gap_panels:
                x = np.arange(len(labels), dtype=np.float64)
                ax.bar(x, complete_vals, width=0.80, color="#4c78a8", alpha=0.40, label="Complete")
                ax.bar(x, masked_vals, width=0.58, color="#f28e2b", alpha=0.75, label=select_type)
                ax.set_title(f"{select_type} | {title}")
                ax.set_xticks(x)
                if len(labels) <= 60:
                    ax.set_xticklabels(labels, rotation=90, fontsize=7)
                else:
                    ax.set_xticklabels([])
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(loc="upper right", fontsize=8)
        axes[0, 0].set_ylabel(density_ylabel)
        axes[1, 0].set_ylabel(density_ylabel)
        axes[0, 1].set_ylabel("Gap value")
        axes[1, 1].set_ylabel("Gap value")
        fig.tight_layout()

        if path is not None:
            if path.suffix.lower() == ".svg":
                fig.savefig(path, format="svg")
            else:
                fig.savefig(path.with_suffix(".svg"), format="svg")
        return fig

    def complete_gap_vs_pure_avg_gap_plot(
        self,
        *,
        path: Optional[pathlib.Path] = None,
        bins: int = 40,
        gap_threshold: Optional[float] = None,
    ):
        if not self.partial_gap_sens_data.get("closest_matches"):
            raise ValueError("No partial gap data found. Call collect_partial_detector_gap_sens(...) first.")

        complete = np.asarray(self.partial_gap_sens_data["complete_gaps"], dtype=np.float64)
        pure_avg = np.asarray(
            [float(rec["pure_avg_gap"]) for rec in self.partial_gap_sens_data["closest_matches"]],
            dtype=np.float64,
        )
        if complete.size == 0 or pure_avg.size == 0:
            raise ValueError("No values available for complete/pure-avg distribution plotting.")

        if gap_threshold is None:
            gap_threshold = self.partial_gap_sens_data.get("gap_threshold", 0.0)

        all_vals = np.concatenate([complete, pure_avg])
        x_min = float(np.min(all_vals))
        x_max = float(np.max(all_vals))
        if x_max <= x_min:
            x_max = x_min + 1e-9

        c_mean = float(np.mean(complete))
        c_std = float(np.std(complete))
        p_mean = float(np.mean(pure_avg))
        p_std = float(np.std(pure_avg))

        fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.2), dpi=120)

        # Histogram overlays.
        ax.hist(
            complete,
            bins=bins,
            density=True,
            alpha=0.25,
            color="#4c78a8",
            edgecolor="#2b5d9a",
            linewidth=0.8,
            label="Complete gap histogram",
        )
        ax.hist(
            pure_avg,
            bins=bins,
            density=True,
            alpha=0.25,
            color="#f28e2b",
            edgecolor="#c96a08",
            linewidth=0.8,
            label="Pure-avg gap histogram",
        )

        # Smoothed envelopes from histogram densities.
        c_hist, c_edges = np.histogram(complete, bins=bins, density=True)
        p_hist, p_edges = np.histogram(pure_avg, bins=bins, density=True)
        c_x = 0.5 * (c_edges[:-1] + c_edges[1:])
        p_x = 0.5 * (p_edges[:-1] + p_edges[1:])
        if len(c_hist) >= 3 and len(p_hist) >= 3:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
            kernel /= np.sum(kernel)
            c_smooth = np.convolve(c_hist, kernel, mode="same")
            p_smooth = np.convolve(p_hist, kernel, mode="same")
            ax.plot(c_x, c_smooth, color="#1f4f99", linewidth=2.3, label="Complete gap envelope")
            ax.plot(p_x, p_smooth, color="#b85a00", linewidth=2.3, label="Pure-avg gap envelope")

        # Mean lines.
        ax.axvline(c_mean, color="#1f4f99", linewidth=2.0, linestyle="-", label="Complete mean")
        ax.axvline(p_mean, color="#b85a00", linewidth=2.0, linestyle="-", label="Pure-avg mean")

        # Std shaded bands.
        ax.axvspan(c_mean - c_std, c_mean + c_std, color="#4c78a8", alpha=0.12, label="Complete ±1σ")
        ax.axvspan(p_mean - p_std, p_mean + p_std, color="#f28e2b", alpha=0.12, label="Pure-avg ±1σ")

        # Threshold line.
        if gap_threshold is not None:
            ax.axvline(float(gap_threshold), color="#d62728", linewidth=2.2, linestyle="--", label="Gap threshold")

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Gap")
        ax.set_ylabel("Density")
        ax.set_title("Complete Gap vs Pure-Avg Gap Distribution")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        if path is not None:
            if path.suffix.lower() == ".svg":
                fig.savefig(path, format="svg")
            else:
                fig.savefig(path.with_suffix(".svg"), format="svg")
        return fig

    def partial_gap_best_combo_distribution_svg(
        self,
        *,
        path: pathlib.Path,
        value_type: str = "abs_diff",
        bins: int = 40,
    ) -> None:
        import matplotlib.pyplot as plt

        if self.best_combo_distribution is None:
            raise ValueError(
                "No best combo distribution found. Call calculate_partial_gap_best_combo_distribution() first."
            )
        if value_type not in {"left_len", "top_len", "t", "abs_diff"}:
            raise ValueError("value_type must be one of {'left_len', 'top_len', 't', 'abs_diff'}.")

        d = self.best_combo_distribution
        if value_type == "left_len":
            values = np.asarray([float(v[0]) for v in d["best_combo_samples"]], dtype=np.float64)
        elif value_type == "top_len":
            values = np.asarray([float(v[1]) for v in d["best_combo_samples"]], dtype=np.float64)
        elif value_type == "t":
            values = np.asarray([float(v[2]) for v in d["best_combo_samples"]], dtype=np.float64)
        else:
            values = np.asarray(d["abs_diff_samples"], dtype=np.float64)

        if len(values) == 0:
            raise ValueError("No values available for the selected distribution.")

        mean = float(np.mean(values))
        std = float(np.std(values))
        var = float(np.var(values))

        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), dpi=120)
        ax.hist(values, bins=bins, density=True, alpha=0.35, color="#4c78a8", edgecolor="white", linewidth=0.8)
        hist_y, hist_edges = np.histogram(values, bins=bins, density=True)
        hist_x = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        if len(hist_y) >= 3:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
            kernel /= np.sum(kernel)
            smooth_y = np.convolve(hist_y, kernel, mode="same")
            ax.plot(hist_x, smooth_y, color="#1f2a44", linewidth=2.0, label="Smoothed density")

        ax.axvline(mean, color="#d62728", linewidth=2.0, linestyle="-", label="Mean")
        ax.axvline(mean - std, color="#d62728", linewidth=1.5, linestyle="--", alpha=0.9, label="Mean ± 1σ")
        ax.axvline(mean + std, color="#d62728", linewidth=1.5, linestyle="--", alpha=0.9)

        ax.set_title(f"Best Combo Distribution ({value_type})")
        ax.set_xlabel(value_type)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        stats_text = f"n={len(values)}\nmean={mean:.4f}\nstd={std:.4f}\nvar={var:.4f}"
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.9),
        )
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(path, format="svg")
        plt.close(fig)

    def partial_gap_best_combo_distribution_notebook(
        self,
        *,
        value_type: str = "abs_diff",
        bins: int = 40,
    ):
        return self._plot_from_svg_writer(
            lambda path: self.partial_gap_best_combo_distribution_svg(
                path=path,
                value_type=value_type,
                bins=bins,
            )
        )


    # Helper function to create an SVG plot from a writing function, suitable for Jupyter notebook display.
    @staticmethod
    def _plot_from_svg_writer(write_svg_func):
        try:
            from IPython.display import SVG
        except ImportError as ex:
            raise ImportError("IPython is required for notebook display.") from ex

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            write_svg_func(tmp_path)
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def tile_coloring(tile: gen.Tile, gap_scores: list[tuple[Any, float]]) -> tuple[float, float, float] | None:
        dr, = tile.flags
        dr = int(dr)
        prediction, gap = gap_scores[dr]
        if gap == 0:
            return 0.5, 0.5, 0.5
        max_gap = max(e for _, e in gap_scores)
        return min(gap / max_gap, 1), 0, 0

    @staticmethod
    def tile_coloring_with_count(tile: gen.Tile, values: list[float], counts: list[float]) -> tuple[float, float, float] | None:
        dr, = tile.flags
        dr = int(dr)
        if dr >= len(values) or dr >= len(counts) or counts[dr] <= 0:
            return 0.5, 0.5, 0.5
        value = values[dr]
        max_value = max(values) if values else 0.0
        if max_value <= 0:
            return 0.5, 0.5, 0.5
        return min(value / max_value, 1), 0, 0

    @staticmethod
    def _make_tile_color_func(
        *,
        values: list[float],
        counts: list[float],
        invert: bool,
        low_q: float,
        high_q: float,
        gamma: float,
    ):
        if not values:
            low = 0.0
            high = 1.0
        else:
            arr = np.asarray(values, dtype=np.float64)
            cnt = np.asarray(counts, dtype=np.float64)
            active = arr[cnt > 0]
            if active.size == 0:
                low = 0.0
                high = 1.0
            else:
                low = float(np.quantile(active, low_q))
                high = float(np.quantile(active, high_q))
                if high <= low:
                    high = low + 1e-9

        def tile_color(tile: gen.Tile) -> tuple[float, float, float] | None:
            dr, = tile.flags
            dr = int(dr)
            if dr >= len(values) or dr >= len(counts) or counts[dr] <= 0:
                return 0.5, 0.5, 0.5
            v = float(values[dr])
            norm = (v - low) / (high - low)
            norm = min(max(norm, 0.0), 1.0)
            norm = norm ** gamma
            if invert:
                norm = 1.0 - norm
            return norm, 0, 0

        return tile_color

    @staticmethod
    def _append_svg_legend(
        path: pathlib.Path,
        *,
        title: str,
        low_label: str,
        high_label: str,
        no_data_label: str,
        red_is_high: bool,
    ) -> None:
        text = path.read_text()
        grad_colors = [
            "rgb(0,0,0)",
            "rgb(51,0,0)",
            "rgb(102,0,0)",
            "rgb(153,0,0)",
            "rgb(204,0,0)",
            "rgb(255,0,0)",
        ]
        if not red_is_high:
            grad_colors = list(reversed(grad_colors))
        legend = [
            '<g id="legend" transform="translate(10,10)">',
            '<rect x="0" y="0" width="260" height="66" fill="white" fill-opacity="0.82" stroke="#444" stroke-width="0.6"/>',
            f'<text x="8" y="14" font-size="10" font-weight="700" fill="#111">{title}</text>',
            f'<rect x="8" y="22" width="14" height="10" fill="{grad_colors[0]}" stroke="none"/>',
            f'<rect x="22" y="22" width="14" height="10" fill="{grad_colors[1]}" stroke="none"/>',
            f'<rect x="36" y="22" width="14" height="10" fill="{grad_colors[2]}" stroke="none"/>',
            f'<rect x="50" y="22" width="14" height="10" fill="{grad_colors[3]}" stroke="none"/>',
            f'<rect x="64" y="22" width="14" height="10" fill="{grad_colors[4]}" stroke="none"/>',
            f'<rect x="78" y="22" width="14" height="10" fill="{grad_colors[5]}" stroke="none"/>',
            f'<text x="98" y="30" font-size="9" fill="#111">{low_label} -> {high_label}</text>',
            '<rect x="8" y="40" width="14" height="10" fill="rgb(128,128,128)" stroke="none"/>',
            f'<text x="28" y="48" font-size="9" fill="#111">{no_data_label}</text>',
            '</g>',
        ]
        legend_block = "\n".join(legend)
        if "</svg>" in text:
            text = text.replace("</svg>", f"{legend_block}\n</svg>", 1)
        else:
            text += "\n" + legend_block
        path.write_text(text)
