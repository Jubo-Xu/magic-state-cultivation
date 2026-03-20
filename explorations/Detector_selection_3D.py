import pathlib
import sys
import tempfile
from typing import Any, Optional

import numpy as np
import stim

src_path = pathlib.Path(__file__).parent.parent / "src"
assert src_path.exists()
sys.path.append(str(src_path))

import gen


class DetectorSelection3D:
    def __init__(
        self,
        *,
        circuit: Optional[stim.Circuit] = None,
        circuit_generator: Optional[Any] = None,
        stage_timeline_map: Optional[dict[str, tuple[int, int]]] = None,
    ):
        if circuit_generator is not None:
            if getattr(circuit_generator, "ideal_circuit", None) is None:
                raise ValueError("circuit_generator.ideal_circuit is None. Call generate() first.")
            circuit = getattr(circuit_generator, "noisy_circuit", None) or circuit_generator.ideal_circuit
            if stage_timeline_map is None and getattr(circuit_generator, "params", None) is not None:
                stage_timeline_map = getattr(circuit_generator.params, "StageTimelineMap", None)
        if circuit is None:
            raise ValueError("Provide either circuit or circuit_generator.")

        self.circuit = circuit
        self.stage_timeline_map = stage_timeline_map or {}
        self.dem = self.circuit.detector_error_model()
        self.det_coords = self.dem.get_detector_coordinates()
        self.num_detectors = self.dem.num_detectors

        self.selected_mask_bool = np.zeros(self.num_detectors, dtype=np.bool_)
        self.selected_mask_packed = np.packbits(self.selected_mask_bool, bitorder="little")
        self.selection_metadata: dict[str, Any] = {}

    def _cycle_slices(self):
        codes = gen.circuit_to_cycle_code_slices(self.circuit)
        ticks = sorted(codes.keys())
        return ticks, codes

    def _find_first_hybrid_slice_index(
        self,
        *,
        growth_ratio_threshold: float = 1.20,
        growth_abs_threshold: int = 8,
    ) -> tuple[int, list[int], list[int]]:
        ticks, codes = self._cycle_slices()
        counts = [len(codes[t].used_set) for t in ticks]
        if len(ticks) <= 1:
            return 0, ticks, counts
        for i in range(1, len(ticks)):
            prev = max(counts[i - 1], 1)
            cur = counts[i]
            if cur - counts[i - 1] >= growth_abs_threshold or cur / prev >= growth_ratio_threshold:
                return i, ticks, counts
        return 0, ticks, counts

    @staticmethod
    def _map_slice_index_to_z_start(slice_index: int, z_levels: np.ndarray) -> int:
        # z-levels and cycle-slices are both monotonic but not in the same unit.
        # Use ordinal mapping (index-based) instead of timeline-map-based absolute ticks.
        if len(z_levels) == 0:
            return 0
        return int(z_levels[min(max(slice_index, 0), len(z_levels) - 1)])

    def _extract_spacetime(self):
        ids = []
        x = []
        y = []
        z = []
        for d in range(self.num_detectors):
            c = self.det_coords.get(d, [])
            if len(c) < 3:
                continue
            ids.append(d)
            x.append(float(c[0]))
            y.append(float(c[1]))
            z.append(int(round(float(c[2]))))
        if not ids:
            raise ValueError("No detector coordinates with at least 3 dimensions found.")
        ids_arr = np.asarray(ids, dtype=np.int64)
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        z_arr = np.asarray(z, dtype=np.int64)
        return ids_arr, x_arr, y_arr, z_arr

    def build_color_region_mask(
        self,
        *,
        left_len: int,
        top_len: int,
        t: int,
        mode: str = "rectangle",
        stage_name: str = "escape",
    ) -> tuple[np.ndarray, np.ndarray]:
        if left_len <= 0 or top_len <= 0 or t <= 0:
            raise ValueError("left_len, top_len, and t must be positive.")
        if mode not in {"rectangle", "triangle"}:
            raise ValueError("mode must be 'rectangle' or 'triangle'.")

        ids, x, y, z = self._extract_spacetime()
        hybrid_idx, slice_ticks, slice_counts = self._find_first_hybrid_slice_index()
        z_levels = np.unique(np.sort(z))
        z0_target = self._map_slice_index_to_z_start(hybrid_idx, z_levels)
        start_idx = int(np.searchsorted(z_levels, z0_target, side="left"))
        if start_idx >= len(z_levels):
            start_idx = len(z_levels) - 1
        z0 = int(z_levels[start_idx])
        z_window = z_levels[start_idx:start_idx + int(t)]
        if len(z_window) == 0:
            z_window = np.asarray([z0], dtype=np.int64)
        z1 = int(z_window[-1])

        base = z == z0

        base_x_levels = np.unique(np.sort(x[base]))
        base_y_levels = np.unique(np.sort(y[base]))

        x_idx_cut = min(top_len - 1, len(base_x_levels) - 1)
        y_idx_cut = min(left_len - 1, len(base_y_levels) - 1)
        x_cut = base_x_levels[x_idx_cut]
        y_cut = base_y_levels[y_idx_cut]

        in_time = np.isin(z, z_window)
        in_rect = (x <= x_cut) & (y <= y_cut)

        selected = in_time & in_rect
        if mode == "triangle":
            x_index = np.searchsorted(base_x_levels, x, side="right") - 1
            y_index = np.searchsorted(base_y_levels, y, side="right") - 1
            x_index = np.clip(x_index, 0, max(0, len(base_x_levels) - 1))
            y_index = np.clip(y_index, 0, max(0, len(base_y_levels) - 1))

            top_den = max(top_len - 1, 1)
            left_den = max(left_len - 1, 1)
            tri = (x_index / top_den + y_index / left_den) <= 1.0
            selected &= tri

        mask_bool = np.zeros(self.num_detectors, dtype=np.bool_)
        mask_bool[ids[selected]] = True
        mask_packed = np.packbits(mask_bool, bitorder="little")

        self.selected_mask_bool = mask_bool
        self.selected_mask_packed = mask_packed
        self.selection_metadata = {
            "mode": mode,
            "stage_name": stage_name,
            "left_len": int(left_len),
            "top_len": int(top_len),
            "t": int(t),
            "hybrid_slice_index": int(hybrid_idx),
            "hybrid_slice_tick": int(slice_ticks[hybrid_idx]) if slice_ticks else 0,
            "slice_ticks": [int(e) for e in slice_ticks],
            "slice_qubit_counts": [int(e) for e in slice_counts],
            "z_target_start": int(z0_target),
            "z_start": int(z0),
            "z_end": int(z1),
            "z_levels_selected": [int(e) for e in z_window.tolist()],
            "selected_detector_count": int(np.count_nonzero(mask_bool)),
        }
        return mask_bool, mask_packed

    def visualize_selected_region_svg(
        self,
        path: pathlib.Path,
        *,
        canvas_height: int = 900,
    ) -> None:
        if not np.any(self.selected_mask_bool):
            raise ValueError("No selected region mask found. Call build_color_region_mask(...) first.")

        z0 = int(self.selection_metadata.get("z_start", 0))
        z1 = int(self.selection_metadata.get("z_end", z0))

        codes = gen.circuit_to_cycle_code_slices(self.circuit)
        ticks = sorted(codes.keys())
        hybrid_idx = int(self.selection_metadata.get("hybrid_slice_index", 0))
        t_layers = int(self.selection_metadata.get("t", 1))
        sel_ticks = ticks[hybrid_idx:hybrid_idx + t_layers]
        if not sel_ticks and ticks:
            sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        panels = [codes[k].with_transformed_coords(lambda e: e * (1 + 1j)) for k in sel_ticks]

        def tile_color(tile: gen.Tile):
            dr, = tile.flags
            d = int(dr)
            if d >= len(self.selected_mask_bool):
                return 0.6, 0.6, 0.6
            if self.selected_mask_bool[d]:
                return 0.95, 0.1, 0.1
            return 0.82, 0.82, 0.82

        titles = [f"tick={k}" for k in sel_ticks]
        title_prefix = (
            f"Selected region ({self.selection_metadata['mode']}): "
            f"left_len={self.selection_metadata['left_len']}, "
            f"top_len={self.selection_metadata['top_len']}, "
            f"t={self.selection_metadata['t']}, "
            f"z=[{self.selection_metadata['z_start']}..{self.selection_metadata['z_end']}]"
        )
        if titles:
            titles[0] = f"{title_prefix} | {titles[0]}"

        panels[0].write_svg(
            path,
            canvas_height=canvas_height,
            other=panels[1:],
            title=titles,
            tile_color_func=tile_color,
            show_coords=False,
            show_obs=False,
        )

    def visualize_selected_region_plot(self):
        try:
            from IPython.display import SVG
        except ImportError as ex:
            raise ImportError("IPython is required for notebook display.") from ex

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.visualize_selected_region_svg(tmp_path)
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)
