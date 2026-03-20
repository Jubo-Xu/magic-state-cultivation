import collections
import pathlib
import sys
import tempfile
from typing import Any

from typing import Optional
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
            }
            return

        _, complete_gaps = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets.copy())
        complete_gaps = complete_gaps.astype(np.float64)

        combos = [
            (left_len, top_len, t)
            for left_len in range(l0, l1 + 1)
            for top_len in range(u0, u1 + 1)
            for t in range(t0, t1 + 1)
        ]

        combo_to_gaps: dict[tuple[int, int, int], np.ndarray] = {}
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

            dets_partial = dets.copy()
            dets_partial &= full_mask.reshape(1, -1)
            _, partial_gaps = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets_partial)
            combo_to_gaps[(left_len, top_len, t)] = partial_gaps.astype(np.float64)

        partial_gaps_per_shot: list[dict[tuple[int, int, int], float]] = []
        closest_matches: list[dict[str, Any]] = []
        for shot_idx in range(remained_shots):
            gap_map: dict[tuple[int, int, int], float] = {}
            for combo in combos:
                gap_map[combo] = float(combo_to_gaps[combo][shot_idx])
            partial_gaps_per_shot.append(gap_map)
            complete_gap = float(complete_gaps[shot_idx])
            best_combo = min(combos, key=lambda c: abs(gap_map[c] - complete_gap))
            best_partial = float(gap_map[best_combo])
            closest_matches.append({
                "shot_idx": int(shot_idx),
                "combo": tuple(best_combo),
                "partial_gap": best_partial,
                "complete_gap": complete_gap,
                "abs_diff": abs(best_partial - complete_gap),
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
        }

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
