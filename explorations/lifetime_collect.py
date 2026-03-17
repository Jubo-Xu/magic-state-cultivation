import collections
import pathlib
import sys
import tempfile
from typing import Any, Optional

import numpy as np
import sinter
import stim
from matplotlib import pyplot as plt

src_path = pathlib.Path(__file__).parent.parent / "src"
assert src_path.exists()
sys.path.append(str(src_path))

import cultiv


class LifeTimeCollect:
    def __init__(
        self,
        *,
        distance: int,
        circuit: Optional[stim.Circuit] = None,
        circuit_generator: Optional[Any] = None,
        wait_rounds: Optional[int] = None,
    ):
        if circuit_generator is not None:
            if getattr(circuit_generator, "ideal_circuit", None) is None:
                raise ValueError("circuit_generator.ideal_circuit is None. Call generate() first.")
            circuit = getattr(circuit_generator, "noisy_circuit", None) or circuit_generator.ideal_circuit
        if circuit is None:
            raise ValueError("Provide either circuit or circuit_generator.")
        if distance not in [3, 5]:
            raise ValueError("distance must be 3 or 5.")

        self.distance = distance
        self.circuit = circuit
        self.circuit_generator = circuit_generator
        self.wait_rounds = wait_rounds if wait_rounds is not None else self._infer_wait_rounds(circuit_generator)

        self.survival_counts: Optional[collections.Counter] = None
        self.qubit_counts: Optional[list[int]] = None
        self.success_rate: Optional[float] = None
        self.num_shots: Optional[int] = None
        self.gap_threshold: Optional[int] = None
        self.selected_attempts_per_kept: Optional[float] = None

    @staticmethod
    def circuit_to_layers(circuit: stim.Circuit) -> list[stim.Circuit]:
        cur_layer = stim.Circuit()
        prev_layers: list[stim.Circuit] = []
        saw_two_qubit_gate = False
        saw_measurement = False
        for inst in circuit.flattened():
            data = stim.GateData(inst.name)
            if inst.name == "QUBIT_COORDS":
                continue
            elif data.is_two_qubit_gate:
                if saw_measurement:
                    saw_measurement = False
                    prev_layers.append(cur_layer)
                    cur_layer = stim.Circuit()
                saw_two_qubit_gate = True
                cur_layer.append(inst)
            elif data.produces_measurements:
                cur_layer.append(inst)
                saw_measurement = True
            elif data.is_reset or data.produces_measurements:
                if saw_two_qubit_gate:
                    saw_two_qubit_gate = False
                    saw_measurement = False
                    prev_layers.append(cur_layer)
                    cur_layer = stim.Circuit()
                cur_layer.append(inst)
            else:
                cur_layer.append(inst)
        if len(cur_layer):
            prev_layers.append(cur_layer)
        return prev_layers

    @staticmethod
    def sample_times(circuit: stim.Circuit, shots: int) -> tuple[collections.Counter, list[int]]:
        sim = stim.FlipSimulator(batch_size=1024, num_qubits=circuit.num_qubits)
        layers = LifeTimeCollect.circuit_to_layers(circuit)

        qubit_counts: list[int] = []
        used_qubits: set[int] = set()
        for layer in layers:
            for inst in layer:
                if inst.name in ["R", "RX"]:
                    for t in inst.targets_copy():
                        used_qubits.add(t.qubit_value)
            qubit_counts.append(len(used_qubits))

        survivors = np.zeros(1024, dtype=np.bool_)
        postselected_detectors: set[int] = set()
        for det, coord in circuit.get_detector_coordinates().items():
            if len(coord) == 3 or coord[-1] == -9 or (len(coord) > 4 and (coord[4] == 0 or coord[4] == 4)):
                postselected_detectors.add(det)
        counts = collections.Counter()

        shots_left = shots
        while shots_left > 0:
            sim.clear()
            cur_det = 0
            tick = 0
            survivors[:] = True
            for layer in layers:
                tick += 1
                sim.do(layer)
                if cur_det < sim.num_detectors:
                    while cur_det < sim.num_detectors:
                        if cur_det in postselected_detectors:
                            fired = sim.get_detector_flips(detector_index=cur_det)
                            survivors &= ~fired
                        cur_det += 1
                counts[tick] += np.count_nonzero(survivors)

            counts[0] += 1024
            shots_left -= 1024

        return counts, qubit_counts

    def collect(
        self,
        *,
        num_shots: int = 1024 * 100,
        sampler: Optional[Any] = None,
        gap_threshold: Optional[int] = None,
        success_rate_mode: str = "raw",
        target_attempts_per_kept: Optional[float] = None,
    ) -> None:
        self.num_shots = num_shots
        self.survival_counts, self.qubit_counts = self.sample_times(self.circuit, num_shots)
        self.success_rate, chosen_gap, chosen_attempts = self._estimate_success_rate(
            shots=num_shots,
            sampler=sampler,
            gap_threshold=gap_threshold,
            success_rate_mode=success_rate_mode,
            target_attempts_per_kept=target_attempts_per_kept,
        )
        self.gap_threshold = chosen_gap
        self.selected_attempts_per_kept = chosen_attempts

        max_k = max(self.survival_counts.keys())
        self.survival_counts[max_k] = num_shots * self.success_rate
        if max_k - 1 in self.survival_counts:
            self.survival_counts[max_k - 1] = self.survival_counts[max_k]

    def save_svg(self, path: pathlib.Path) -> None:
        fig = self._build_figure()
        fig.savefig(path, format="svg")
        plt.close(fig)

    def plot_for_notebook(self):
        try:
            from IPython.display import SVG
        except ImportError as ex:
            raise ImportError("IPython is required for notebook display.") from ex

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.save_svg(tmp_path)
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

    def _build_figure(self):
        if self.survival_counts is None or self.qubit_counts is None or self.num_shots is None:
            raise ValueError("No lifetime data collected. Call collect(...) first.")

        xs_1: list[int] = []
        ys_1: list[float] = []
        xs_2: list[int] = []
        ys_2: list[float] = []

        max_k = max(self.survival_counts.keys())
        for k in range(max_k + 1):
            if k > 0:
                xs_1.append(k)
                ys_1.append(ys_1[-1])
            xs_1.append(k)
            ys_1.append(self.survival_counts[k])

        for k, q in enumerate(self.qubit_counts):
            xs_2.append(k)
            xs_2.append(k + 1)
            ys_2.append(q)
            ys_2.append(q)

        xs_1_arr = np.array(xs_1)
        ys_1_arr = np.array(ys_1) / self.num_shots
        xs_2_arr = np.array(xs_2)
        ys_2_arr = np.array(ys_2) / max(self.qubit_counts)

        fig, ax = plt.subplots(1, 1)
        ax.plot(xs_1_arr, ys_1_arr, label="Surviving Shots")
        ax.plot(xs_2_arr, ys_2_arr, label="Qubits Activated")
        ax.fill_between(xs_1_arr, ys_1_arr * 0, ys_1_arr, alpha=0.2, color="C0")
        ax.fill_between(xs_2_arr, ys_2_arr * 0, ys_2_arr, alpha=0.2, color="C1")
        ax.set_ylim(0, 1.01)
        ax.set_xlim(0, max(xs_1_arr))
        ax.set_yticks([x * 0.1 for x in range(11)])
        ax.set_ylabel("Proportion")

        labels = self._build_labels()
        ax.set_xticks(range(len(labels) + 1), [""] * (len(labels) + 1))
        ax.set_xticks([e + 0.5 for e in range(len(labels))], labels, rotation=90, minor=True)
        ax.xaxis.set_tick_params(length=0, which="minor")
        ax.grid()
        ax.legend()

        sr_text = "unknown" if self.success_rate is None else f"{self.success_rate:.4g}"
        mode_text = (
            "raw keep"
            if self.gap_threshold is None
            else f"gap>={self.gap_threshold}, attempts~{self.selected_attempts_per_kept:.3g}"
        )
        ax.set_title(
            f"Life of a Fault-Distance-{self.distance} Cultivation\n"
            f"(estimated success rate={sr_text}, {mode_text}, wait_rounds={self.wait_rounds})"
        )
        fig.set_size_inches(1024 / 100, 512 / 100)
        fig.set_dpi(100)
        fig.tight_layout()
        return fig

    def _build_labels(self) -> list[str]:
        labels = [
            "Encode T",
            "Stabilize",
            "Check T",
            "Check T",
            *(
                [
                    "Stabilize",
                    "Stabilize",
                    "Stabilize",
                    "Check T",
                    "Check T",
                ]
                * (self.distance == 5)
            ),
            *(["Stabilize"] * self.distance),
            "Escaped!",
            *(["[wait for gap]"] * max(self.wait_rounds - 1, 0)),
            "ready",
        ]
        return labels

    def _infer_wait_rounds(self, circuit_generator: Optional[Any]) -> int:
        if circuit_generator is not None and hasattr(circuit_generator, "params"):
            params = circuit_generator.params
            for name in ["r_post_escape", "r_end", "r2"]:
                value = getattr(params, name, None)
                if isinstance(value, int) and value > 0:
                    return value
        return 10

    def _estimate_success_rate(
        self,
        *,
        shots: int,
        sampler: Optional[Any],
        gap_threshold: Optional[int],
        success_rate_mode: str,
        target_attempts_per_kept: Optional[float],
    ) -> tuple[float, Optional[int], Optional[float]]:
        sampler = sampler if sampler is not None else cultiv.DesaturationSampler()
        compiled = sampler.compiled_sampler_for_task(
            sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model())
        )
        stat = compiled.sample(shots)

        if success_rate_mode == "raw":
            keep = (stat.shots - stat.discards) / stat.shots
            keep = float(max(min(keep, 1.0), 0.0))
            attempts = None if keep == 0 else 1 / keep
            return keep, None, attempts

        if success_rate_mode not in ["manual_gap_threshold", "auto_high_rejection"]:
            raise ValueError("success_rate_mode must be one of: raw, manual_gap_threshold, auto_high_rejection")

        threshold_rows = self._compute_threshold_rows(stat)
        if not threshold_rows:
            keep = (stat.shots - stat.discards) / stat.shots
            keep = float(max(min(keep, 1.0), 0.0))
            attempts = None if keep == 0 else 1 / keep
            return keep, None, attempts

        if success_rate_mode == "manual_gap_threshold":
            if gap_threshold is None:
                raise ValueError("gap_threshold is required when success_rate_mode='manual_gap_threshold'.")
            candidates = [row for row in threshold_rows if row["threshold"] == gap_threshold]
            if not candidates:
                raise ValueError(f"gap_threshold={gap_threshold} not available in sampled gap bins.")
            row = candidates[0]
            return row["keep_rate"], row["threshold"], row["attempts_per_kept"]

        # Auto-select a high-rejection operating point from this circuit's own tradeoff curve.
        if target_attempts_per_kept is None:
            target_attempts_per_kept = 4.0 if self.distance == 3 else 100.0
        valid_rows = [row for row in threshold_rows if row["keep_rate"] > 0]
        row = min(valid_rows, key=lambda r: abs(r["attempts_per_kept"] - target_attempts_per_kept))
        return row["keep_rate"], row["threshold"], row["attempts_per_kept"]

    @staticmethod
    def _compute_threshold_rows(stat: sinter.AnonTaskStats) -> list[dict[str, float | int]]:
        bins: dict[int, dict[str, int]] = {}
        for key, count in stat.custom_counts.items():
            if not (key.startswith("C") or key.startswith("E")):
                continue
            gap = int(key[1:])
            if gap not in bins:
                bins[gap] = {"C": 0, "E": 0}
            if key.startswith("C"):
                bins[gap]["C"] += int(count)
            else:
                bins[gap]["E"] += int(count)
        if not bins:
            return []

        rows: list[dict[str, float | int]] = []
        for threshold in sorted(bins.keys()):
            below = 0
            kept_errors = 0
            for gap, ce in bins.items():
                total = ce["C"] + ce["E"]
                if gap < threshold:
                    below += total
                else:
                    kept_errors += ce["E"]
            effective_discards = int(stat.discards) + below
            kept = int(stat.shots) - effective_discards
            keep_rate = max(min(kept / stat.shots, 1.0), 0.0)
            attempts = float("inf") if keep_rate == 0 else 1 / keep_rate
            kept_error_rate = (kept_errors / kept) if kept > 0 else 0.0
            rows.append(
                {
                    "threshold": threshold,
                    "keep_rate": float(keep_rate),
                    "attempts_per_kept": float(attempts),
                    "kept_error_rate": float(kept_error_rate),
                }
            )
        return rows
