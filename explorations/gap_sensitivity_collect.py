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

# def write_gap_plot(path: pathlib.Path):
#     circuit = cultiv.make_end2end_cultivation_circuit(dcolor=5, dsurface=15, basis='Y', r_growing=5, r_end=4, inject_style='unitary')
#     circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
#     dec = cultiv.DesaturationSampler().compiled_sampler_for_task(sinter.Task(circuit=circuit, detector_error_model=circuit.detector_error_model()))

#     scores = []
#     for d in range(circuit.num_detectors):
#         scores.append(dec.decode_det_set({d}))

#     codes = gen.circuit_to_cycle_code_slices(circuit)
#     ticks = codes.keys()
#     codes = [code.with_transformed_coords(lambda e: e * (1 + 1j)) for code in codes.values()]

#     max_gap = max(e for _, e in scores)
#     def tile_coloring(tile: gen.Tile) -> tuple[float, float, float] | None:
#         dr, = tile.flags
#         dr = int(dr)
#         prediction, gap = scores[dr]
#         if gap == 0:
#             return 0.5, 0.5, 0.5
#         return min(gap / max_gap, 1), 0, 0

#     codes[0].write_svg(
#         path,
#         canvas_height=1000,
#         other=codes[1:],
#         title=[f'tick={e}' for e in sorted(ticks)],
#         tile_color_func=tile_coloring,
#         show_coords=False,
#         show_obs=False,
#     )

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
