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


    def _detector_ids_for_tick(self, code: Any) -> np.ndarray:
        ids: list[int] = []
        for tile in code.tiles:
            for f in tile.flags:
                try:
                    d = int(f)
                except ValueError:
                    continue
                if 0 <= d < self.num_detectors:
                    ids.append(d)
        if not ids:
            return np.empty(0, dtype=np.int64)
        return np.asarray(sorted(set(ids)), dtype=np.int64)

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

        hybrid_idx, slice_ticks, slice_counts = self._find_first_hybrid_slice_index()
        ticks, codes = self._cycle_slices()
        if not ticks:
            raise ValueError("No cycle slices found.")

        sel_ticks = ticks[hybrid_idx:hybrid_idx + int(t)]
        if not sel_ticks:
            sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        base_tick = sel_ticks[0]
        base_ids = self._detector_ids_for_tick(codes[base_tick])
        if len(base_ids) == 0:
            raise ValueError("No detector ids found on the first selected hybrid slice.")

        mask_bool = np.zeros(self.num_detectors, dtype=np.bool_)
        z_selected_all: list[int] = []
        eps = 1e-9

        for tick in sel_ticks:
            tick_ids = self._detector_ids_for_tick(codes[tick])
            if len(tick_ids) == 0:
                continue

            x_tick = np.asarray([float(self.det_coords[d][0]) for d in tick_ids], dtype=np.float64)
            y_tick = np.asarray([float(self.det_coords[d][1]) for d in tick_ids], dtype=np.float64)
            z_tick = np.asarray([int(round(float(self.det_coords[d][2]))) for d in tick_ids], dtype=np.int64)
            z_selected_all.extend(z_tick.tolist())

            # Patch-aligned coordinates.
            u_tick = x_tick + y_tick
            v_tick = y_tick - x_tick
            u_levels = np.unique(np.sort(u_tick))
            v_levels = np.unique(np.sort(v_tick))

            # Left-top corner selection orientation.
            # Left edge: v is maximal. Top edge: u is minimal.
            u_edge = float(u_levels[0])
            v_edge = float(v_levels[-1])

            # One-step inside lines used for length counting.
            u_inner = float(u_levels[1]) if len(u_levels) > 1 else u_edge
            v_inner = float(v_levels[-2]) if len(v_levels) > 1 else v_edge

            left_inner_idx = np.flatnonzero(np.abs(v_tick - v_inner) <= eps)
            top_inner_idx = np.flatnonzero(np.abs(u_tick - u_inner) <= eps)
            if len(left_inner_idx) == 0:
                left_inner_idx = np.flatnonzero(np.abs(v_tick - v_edge) <= eps)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.flatnonzero(np.abs(u_tick - u_edge) <= eps)
            if len(left_inner_idx) == 0:
                left_inner_idx = np.asarray([int(np.argmax(v_tick))], dtype=np.int64)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.asarray([int(np.argmin(u_tick))], dtype=np.int64)

            # top_len: count on one-step-inside top row, from left to right.
            top_order = top_inner_idx[np.argsort(-v_tick[top_inner_idx], kind="mergesort")]
            top_k = min(int(top_len), len(top_order))
            v_cut = float(v_tick[top_order[top_k - 1]])

            # left_len: count downward on the column selected by top_len (i + top_len).
            right_col_idx = np.flatnonzero(np.abs(v_tick - v_cut) <= eps)
            if len(right_col_idx) == 0:
                nearest = int(np.argmin(np.abs(v_tick - v_cut)))
                right_col_idx = np.asarray([nearest], dtype=np.int64)
            right_col_order = right_col_idx[np.argsort(u_tick[right_col_idx], kind="mergesort")]
            left_k = min(int(left_len), len(right_col_order))
            u_cut = float(u_tick[right_col_order[left_k - 1]])

            # Rectangle bounded by top boundary and right boundary; include detectors left of right boundary.
            in_rect = (u_tick <= u_cut + eps) & (v_tick >= v_cut - eps)
            keep = in_rect

            if mode == "triangle":
                # Left-half right triangle inside the rectangle.
                du_den = max(u_cut - u_edge, eps)
                dv_den = max(v_edge - v_cut, eps)
                du = (u_tick - u_edge) / du_den
                dv = (v_edge - v_tick) / dv_den
                keep = keep & ((du + dv) <= 1.0 + 1e-9)

            mask_bool[tick_ids[keep]] = True

        if z_selected_all:
            z_levels_selected = sorted(set(int(e) for e in z_selected_all))
            z0 = int(z_levels_selected[0])
            z1 = int(z_levels_selected[-1])
        else:
            z_levels_selected = []
            z0 = 0
            z1 = 0

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
            "selected_slice_ticks": [int(e) for e in sel_ticks],
            "z_start": int(z0),
            "z_end": int(z1),
            "z_levels_selected": [int(e) for e in z_levels_selected],
            "selected_detector_count": int(np.count_nonzero(mask_bool)),
        }
        return mask_bool, mask_packed

    def build_partial_region_apart_color_mask(
        self,
        *,
        left_len_start: int,
        left_len_end: int,
        top_len_start: int,
        top_len_end: int,
        t: int,
        mode: str = "rectangle",
        triangle_half: str = "left",
        stage_name: str = "escape",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if t <= 0:
            raise ValueError("t must be positive.")
        if left_len_start <= 0 or top_len_start <= 0:
            raise ValueError("left_len_start and top_len_start must be positive.")
        if left_len_end <= left_len_start or top_len_end <= top_len_start:
            raise ValueError("Require left_len_end > left_len_start and top_len_end > top_len_start.")
        if mode not in {"rectangle", "triangle"}:
            raise ValueError("mode must be 'rectangle' or 'triangle'.")
        if triangle_half not in {"left", "right"}:
            raise ValueError("triangle_half must be 'left' or 'right'.")

        hybrid_idx, slice_ticks, slice_counts = self._find_first_hybrid_slice_index()
        ticks, codes = self._cycle_slices()
        if not ticks:
            raise ValueError("No cycle slices found.")

        sel_ticks = ticks[hybrid_idx:hybrid_idx + int(t)]
        if not sel_ticks:
            sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        mask_bool = np.zeros(self.num_detectors, dtype=np.bool_)
        z_selected_all: list[int] = []
        eps = 1e-9

        width = top_len_end - top_len_start
        height = left_len_end - left_len_start

        for tick in sel_ticks:
            tick_ids = self._detector_ids_for_tick(codes[tick])
            if len(tick_ids) == 0:
                continue

            x_tick = np.asarray([float(self.det_coords[d][0]) for d in tick_ids], dtype=np.float64)
            y_tick = np.asarray([float(self.det_coords[d][1]) for d in tick_ids], dtype=np.float64)
            z_tick = np.asarray([int(round(float(self.det_coords[d][2]))) for d in tick_ids], dtype=np.int64)
            z_selected_all.extend(z_tick.tolist())

            u_tick = x_tick + y_tick
            v_tick = y_tick - x_tick
            u_levels = np.unique(np.sort(u_tick))
            u_inner = float(u_levels[1]) if len(u_levels) > 1 else float(u_levels[0])

            top_inner_idx = np.flatnonzero(np.abs(u_tick - u_inner) <= eps)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.flatnonzero(np.abs(u_tick - float(u_levels[0])) <= eps)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.arange(len(tick_ids), dtype=np.int64)

            top_order = top_inner_idx[np.argsort(-v_tick[top_inner_idx], kind="mergesort")]
            top_v_values = v_tick[top_order]
            if len(top_v_values) == 0:
                continue

            col_start = min(max(top_len_start - 1, 0), len(top_v_values) - 1)
            col_end_excl = min(max(top_len_end - 1, 1), len(top_v_values))
            if col_end_excl <= col_start:
                continue

            # Column index from left-to-right order defined on the top-inner row.
            col_rank = np.argmin(np.abs(v_tick[:, None] - top_v_values[None, :]), axis=1) + 1

            # Row index from top-to-bottom order.
            row_rank = np.argmin(np.abs(u_tick[:, None] - u_levels[None, :]), axis=1) + 1

            in_rect = (
                (col_rank >= top_len_start)
                & (col_rank < top_len_end)
                & (row_rank >= left_len_start)
                & (row_rank < left_len_end)
            )
            keep = in_rect

            if mode == "triangle" and width > 1 and height > 1:
                c_local = (col_rank.astype(np.float64) - float(top_len_start)) / float(width - 1)
                r_local = (row_rank.astype(np.float64) - float(left_len_start)) / float(height - 1)
                if triangle_half == "left":
                    keep = keep & ((r_local + c_local) <= 1.0 + 1e-9)
                else:
                    keep = keep & ((r_local + c_local) >= 1.0 - 1e-9)

            mask_bool[tick_ids[keep]] = True

        if z_selected_all:
            z_levels_selected = sorted(set(int(e) for e in z_selected_all))
            z0 = int(z_levels_selected[0])
            z1 = int(z_levels_selected[-1])
        else:
            z_levels_selected = []
            z0 = 0
            z1 = 0

        mask_packed = np.packbits(mask_bool, bitorder="little")
        metadata = {
            "mode": mode,
            "triangle_half": triangle_half,
            "stage_name": stage_name,
            "left_len_start": int(left_len_start),
            "left_len_end": int(left_len_end),
            "top_len_start": int(top_len_start),
            "top_len_end": int(top_len_end),
            "t": int(t),
            "hybrid_slice_index": int(hybrid_idx),
            "hybrid_slice_tick": int(slice_ticks[hybrid_idx]) if slice_ticks else 0,
            "slice_ticks": [int(e) for e in slice_ticks],
            "slice_qubit_counts": [int(e) for e in slice_counts],
            "selected_slice_ticks": [int(e) for e in sel_ticks],
            "z_start": int(z0),
            "z_end": int(z1),
            "z_levels_selected": [int(e) for e in z_levels_selected],
            "selected_detector_count": int(np.count_nonzero(mask_bool)),
        }
        return mask_bool, mask_packed, metadata

    def visualize_partial_region_apart_color_svg(
        self,
        path: pathlib.Path,
        *,
        mask_bool: np.ndarray,
        metadata: dict[str, Any],
        canvas_height: int = 900,
    ) -> None:
        if mask_bool is None or not np.any(mask_bool):
            raise ValueError("No selected region mask found in mask_bool.")

        codes = gen.circuit_to_cycle_code_slices(self.circuit)
        ticks = sorted(codes.keys())
        sel_ticks = [int(e) for e in metadata.get("selected_slice_ticks", [])]
        if not sel_ticks:
            hybrid_idx = int(metadata.get("hybrid_slice_index", 0))
            t_layers = int(metadata.get("t", 1))
            sel_ticks = ticks[hybrid_idx:hybrid_idx + t_layers]
            if not sel_ticks and ticks:
                sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        panels = [codes[k].with_transformed_coords(lambda e: e * (1 + 1j)) for k in sel_ticks]

        def tile_color(tile: gen.Tile):
            dr, = tile.flags
            d = int(dr)
            if d >= len(mask_bool):
                return 0.6, 0.6, 0.6
            if mask_bool[d]:
                return 0.95, 0.1, 0.1
            return 0.82, 0.82, 0.82

        titles = [f"tick={k}" for k in sel_ticks]
        title_prefix = (
            f"Apart region ({metadata.get('mode', 'rectangle')}, half={metadata.get('triangle_half', 'left')}): "
            f"left=[{metadata.get('left_len_start')}..{metadata.get('left_len_end')}), "
            f"top=[{metadata.get('top_len_start')}..{metadata.get('top_len_end')}), "
            f"t={metadata.get('t')}, "
            f"z=[{metadata.get('z_start')}..{metadata.get('z_end')}]"
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

    def visualize_partial_region_apart_color_plot(
        self,
        *,
        mask_bool: np.ndarray,
        metadata: dict[str, Any],
    ):
        try:
            from IPython.display import SVG
        except ImportError as ex:
            raise ImportError("IPython is required for notebook display.") from ex

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.visualize_partial_region_apart_color_svg(
                tmp_path,
                mask_bool=mask_bool,
                metadata=metadata,
            )
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

    def build_partial_region_multi_mask(
        self,
        *,
        left_len: int,
        top_len: int,
        t: int,
        mode: str = "rectangle",
        stage_name: str = "escape",
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
        if left_len <= 0 or top_len <= 0 or t <= 0:
            raise ValueError("left_len, top_len, and t must be positive.")
        if mode not in {"rectangle", "triangle"}:
            raise ValueError("mode must be 'rectangle' or 'triangle'.")

        hybrid_idx, slice_ticks, slice_counts = self._find_first_hybrid_slice_index()
        ticks, codes = self._cycle_slices()
        if not ticks:
            raise ValueError("No cycle slices found.")

        sel_ticks = ticks[hybrid_idx:hybrid_idx + int(t)]
        if not sel_ticks:
            sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        # Region order:
        # 0: color-code region (same as build_color_region_mask)
        # 1: right-half triangle (only when mode='triangle'; else all zero)
        # 2: top-right rectangle (top..left_len, top_len..rightmost)
        # 3: bottom rectangle left half
        # 4: bottom rectangle right half
        masks_bool = [np.zeros(self.num_detectors, dtype=np.bool_) for _ in range(5)]

        z_selected_all: list[int] = []
        eps = 1e-9

        for tick in sel_ticks:
            tick_ids = self._detector_ids_for_tick(codes[tick])
            if len(tick_ids) == 0:
                continue

            x_tick = np.asarray([float(self.det_coords[d][0]) for d in tick_ids], dtype=np.float64)
            y_tick = np.asarray([float(self.det_coords[d][1]) for d in tick_ids], dtype=np.float64)
            z_tick = np.asarray([int(round(float(self.det_coords[d][2]))) for d in tick_ids], dtype=np.int64)
            z_selected_all.extend(z_tick.tolist())

            u_tick = x_tick + y_tick
            v_tick = y_tick - x_tick
            u_levels = np.unique(np.sort(u_tick))
            v_levels = np.unique(np.sort(v_tick))

            u_edge = float(u_levels[0])
            v_edge = float(v_levels[-1])
            u_inner = float(u_levels[1]) if len(u_levels) > 1 else u_edge
            v_inner = float(v_levels[-2]) if len(v_levels) > 1 else v_edge

            left_inner_idx = np.flatnonzero(np.abs(v_tick - v_inner) <= eps)
            top_inner_idx = np.flatnonzero(np.abs(u_tick - u_inner) <= eps)
            if len(left_inner_idx) == 0:
                left_inner_idx = np.flatnonzero(np.abs(v_tick - v_edge) <= eps)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.flatnonzero(np.abs(u_tick - u_edge) <= eps)
            if len(left_inner_idx) == 0:
                left_inner_idx = np.asarray([int(np.argmax(v_tick))], dtype=np.int64)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.asarray([int(np.argmin(u_tick))], dtype=np.int64)

            top_order = top_inner_idx[np.argsort(-v_tick[top_inner_idx], kind="mergesort")]
            top_k = min(int(top_len), len(top_order))
            v_cut = float(v_tick[top_order[top_k - 1]])

            right_col_idx = np.flatnonzero(np.abs(v_tick - v_cut) <= eps)
            if len(right_col_idx) == 0:
                nearest = int(np.argmin(np.abs(v_tick - v_cut)))
                right_col_idx = np.asarray([nearest], dtype=np.int64)
            right_col_order = right_col_idx[np.argsort(u_tick[right_col_idx], kind="mergesort")]
            left_k = min(int(left_len), len(right_col_order))
            u_cut = float(u_tick[right_col_order[left_k - 1]])

            # Base color/triangle rectangle from build_color_region_mask semantics.
            top_left_rect = (u_tick <= u_cut + eps) & (v_tick >= v_cut - eps)
            # Disjoint top-right rectangle: same top rows, columns strictly right of cut.
            top_right_rect = (u_tick <= u_cut + eps) & (v_tick < v_cut - eps)

            # region 2: top-right rectangle (disjoint from region 0/1)
            masks_bool[2][tick_ids[top_right_rect]] = True

            # region 0 and region 1 from triangle split (or rectangle fallback), inside top-left rect.
            if mode == "triangle":
                du_den = max(u_cut - u_edge, eps)
                dv_den = max(v_edge - v_cut, eps)
                du = (u_tick - u_edge) / du_den
                dv = (v_edge - v_tick) / dv_den
                tri_left = top_left_rect & ((du + dv) <= 1.0 + 1e-9)
                tri_right = top_left_rect & (~tri_left)
                masks_bool[0][tick_ids[tri_left]] = True
                masks_bool[1][tick_ids[tri_right]] = True
            else:
                masks_bool[0][tick_ids[top_left_rect]] = True
                # region 1 stays all zero for rectangle mode.

            # Remaining rectangle: rows below left_len cut, all columns.
            remaining = u_tick > u_cut + eps
            n_cols = len(top_order)
            if n_cols <= 1:
                left_half = remaining
                right_half = np.zeros_like(remaining)
            else:
                # Map each detector to left-to-right column rank based on top-inner row ordering.
                top_v_values = v_tick[top_order]
                col_rank = np.argmin(np.abs(v_tick[:, None] - top_v_values[None, :]), axis=1) + 1
                split_col = (n_cols + 1) // 2
                left_half = remaining & (col_rank <= split_col)
                right_half = remaining & (col_rank > split_col)

            masks_bool[3][tick_ids[left_half]] = True
            masks_bool[4][tick_ids[right_half]] = True

        masks_packed = [np.packbits(mask, bitorder="little") for mask in masks_bool]

        if z_selected_all:
            z_levels_selected = sorted(set(int(e) for e in z_selected_all))
            z0 = int(z_levels_selected[0])
            z1 = int(z_levels_selected[-1])
        else:
            z_levels_selected = []
            z0 = 0
            z1 = 0

        metadata = {
            "mode": mode,
            "stage_name": stage_name,
            "left_len": int(left_len),
            "top_len": int(top_len),
            "t": int(t),
            "hybrid_slice_index": int(hybrid_idx),
            "hybrid_slice_tick": int(slice_ticks[hybrid_idx]) if slice_ticks else 0,
            "slice_ticks": [int(e) for e in slice_ticks],
            "slice_qubit_counts": [int(e) for e in slice_counts],
            "selected_slice_ticks": [int(e) for e in sel_ticks],
            "z_start": int(z0),
            "z_end": int(z1),
            "z_levels_selected": [int(e) for e in z_levels_selected],
            "region_sizes": [int(np.count_nonzero(m)) for m in masks_bool],
            "region_names": [
                "region0_color_or_left_triangle",
                "region1_right_triangle",
                "region2_top_right_rectangle",
                "region3_bottom_left_half",
                "region4_bottom_right_half",
            ],
        }
        return masks_bool, masks_packed, metadata

    def visualize_partial_region_multi_svg(
        self,
        path: pathlib.Path,
        *,
        masks_bool: list[np.ndarray],
        metadata: dict[str, Any],
        canvas_height: int = 900,
    ) -> None:
        if masks_bool is None or len(masks_bool) != 5:
            raise ValueError("masks_bool must be a list of 5 boolean masks.")

        codes = gen.circuit_to_cycle_code_slices(self.circuit)
        ticks = sorted(codes.keys())
        sel_ticks = [int(e) for e in metadata.get("selected_slice_ticks", [])]
        if not sel_ticks:
            hybrid_idx = int(metadata.get("hybrid_slice_index", 0))
            t_layers = int(metadata.get("t", 1))
            sel_ticks = ticks[hybrid_idx:hybrid_idx + t_layers]
            if not sel_ticks and ticks:
                sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        panels = [codes[k].with_transformed_coords(lambda e: e * (1 + 1j)) for k in sel_ticks]

        colors = [
            (0.95, 0.10, 0.10),
            (0.98, 0.55, 0.10),
            (1.00, 0.41, 0.71),
            (0.12, 0.70, 0.30),
            (0.62, 0.24, 0.85),
        ]

        def tile_color(tile: gen.Tile):
            dr, = tile.flags
            d = int(dr)
            for i, mask in enumerate(masks_bool):
                if d < len(mask) and mask[d]:
                    return colors[i]
            return 0.82, 0.82, 0.82

        titles = [f"tick={k}" for k in sel_ticks]
        title_prefix = (
            f"Multi-region ({metadata.get('mode', 'rectangle')}): "
            f"left_len={metadata.get('left_len')}, "
            f"top_len={metadata.get('top_len')}, "
            f"t={metadata.get('t')}, "
            f"z=[{metadata.get('z_start')}..{metadata.get('z_end')}]"
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

    def visualize_partial_region_multi_plot(
        self,
        *,
        masks_bool: list[np.ndarray],
        metadata: dict[str, Any],
    ):
        try:
            from IPython.display import SVG
        except ImportError as ex:
            raise ImportError("IPython is required for notebook display.") from ex

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.visualize_partial_region_multi_svg(
                tmp_path,
                masks_bool=masks_bool,
                metadata=metadata,
            )
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

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
        sel_ticks = [int(e) for e in self.selection_metadata.get("selected_slice_ticks", [])]
        if not sel_ticks:
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
