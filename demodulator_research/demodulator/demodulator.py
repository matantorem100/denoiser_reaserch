import numpy as np
import pydantic

from synthetic_dataset_creation.coding.coding import Coding, CodingConfig
from synthetic_dataset_creation.constellation.constellation import Constellation, ConstellationConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShape, PulseShapeConfig


class DemodulatorConfig(pydantic.BaseModel):
    pulse_shape_config: PulseShapeConfig
    constellation_config: ConstellationConfig


class Demodulator:
    def __init__(self, config: DemodulatorConfig):
        self.config = config
        self._pulse_shape_instance = PulseShape(config.pulse_shape_config)
        self._constellation_instance = Constellation(config.constellation_config)
        self._coding_instance = Coding(CodingConfig(constellation=config.constellation_config))


    def demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float,
                   method: str = "differentiate", frequency_sensitivity: float = 0.5) -> np.ndarray:
        if method == "differentiate":
            return self.differentiate_demodulate(rx_signal, symbol_time, sample_rate, frequency_sensitivity)
        elif method == "coherent":
            return self.coherent_demodulate(rx_signal, symbol_time, sample_rate, frequency_sensitivity)
        elif method == "non_coherent":
            return self.non_coherent_demodulate(rx_signal, symbol_time, sample_rate, frequency_sensitivity)
        else:
            raise ValueError(f"Unknown demodulation method: {method}")

    def differentiate_demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float,
                                 frequency_sensitivity: float = 0.5) -> np.ndarray:

        sps = int(symbol_time * sample_rate)

        rx_signal = (np.diff(np.unwrap(np.angle(rx_signal))) /
                     (2.0 * np.pi * frequency_sensitivity) * sample_rate)

        constellation_points = self._constellation_instance.generate_constellation_points()
        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)

        matched_filter = pulse_shape[::-1].conj()
        mf_energy = np.sum(np.abs(matched_filter) ** 2)
        filtered_signal = np.convolve(rx_signal, matched_filter, mode="full") / mf_energy

        sampling_offset = ((len(pulse_shape) - 1) // 2) * 2
        detected_symbols = filtered_signal[sampling_offset::sps]

        n_expected_symbols = int(np.ceil((len(rx_signal) - len(pulse_shape) + 1) / sps))
        n_expected_symbols = max(n_expected_symbols, 0)
        detected_symbols = detected_symbols[:n_expected_symbols]

        bits = self._coding_instance.generate_symbols_to_bits(detected_symbols, constellation_points)
        return bits

    def coherent_demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float,
                            frequency_sensitivity: float = 0.5) -> np.ndarray:

        rx_signal = np.asarray(rx_signal)
        sps = int(symbol_time * sample_rate)

        constellation_points = self._constellation_instance.generate_constellation_points()
        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)

        h = frequency_sensitivity
        dt = 1.0 / sample_rate

        n_symbols_in_signal = int(np.floor((len(rx_signal) - len(pulse_shape) + 1) / sps))

        decided_symbols = []
        phase_state = 0.0

        for current_symbol in range(n_symbols_in_signal):
            start_index_of_current_symbol = current_symbol * sps
            end_index_of_current_symbol = start_index_of_current_symbol + sps

            if end_index_of_current_symbol > len(rx_signal):
                break

            # extract the relevant signal (corresponding to the current symbol)
            r_seg = rx_signal[start_index_of_current_symbol: end_index_of_current_symbol]

            best_metric = -np.inf
            best_symbol = constellation_points[0]

            for candidate_constellation_point in constellation_points:
                m = candidate_constellation_point * pulse_shape
                phi = phase_state + 2.0 * np.pi * h * np.cumsum(m) * dt
                modulated_candidate = np.exp(1j * phi)

                denom = np.linalg.norm(r_seg) * np.linalg.norm(modulated_candidate)

                if denom < 1e-12:
                    metric = -np.inf
                else:
                    metric = np.real(np.vdot(modulated_candidate, r_seg)) / denom

                if metric > best_metric:
                    best_metric = metric
                    best_symbol = candidate_constellation_point

            decided_symbols.append(best_symbol)

            m_best = best_symbol * pulse_shape
            phase_state = phase_state + 2.0 * np.pi * h * np.sum(m_best) * dt
            phase_state = np.angle(np.exp(1j * phase_state))

        decided_symbols = np.asarray(decided_symbols, dtype=constellation_points.dtype)
        bits = self._coding_instance.generate_symbols_to_bits(decided_symbols, constellation_points)
        return bits


    @staticmethod
    def _wrap_phase(phase: float) -> float:
        return np.angle(np.exp(1j * phase))


    @staticmethod
    def _quantize_phase_to_state(phase: float, phase_grid: np.ndarray) -> int:
        """
        Map a wrapped phase to the nearest phase state index.
        """
        wrapped = np.angle(np.exp(1j * phase))
        # Circular distance
        circular_error = np.angle(np.exp(1j * (wrapped - phase_grid)))
        return int(np.argmin(np.abs(circular_error)))

    def coherent_fm_viterbi_demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float,
                                       n_phase_states: int = 8, frequency_sensitivity: float = 0.5) -> np.ndarray:
        """
        True Viterbi detector for full-response FM/CPFSK-like signal.

        State = quantized phase at symbol boundary.
        Branch metric = normalized correlation with the candidate waveform.
        """
        h = frequency_sensitivity
        dt = 1.0 / sample_rate
        sps = int(symbol_time * sample_rate)

        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)
        constellation_points = self._constellation_instance.generate_constellation_points()

        n_symbols_in_signal = int(np.floor((len(rx_signal) - len(pulse_shape) + 1) / sps))
        n_symbols_in_signal = max(n_symbols_in_signal, 0)

        # Uniform phase grid over [-pi, pi)
        phase_grid = np.linspace(-np.pi, np.pi, n_phase_states, endpoint=False)

        # Path metric at current time: one metric per state
        path_metrics = np.full(n_phase_states, -np.inf, dtype=float)

        # Initial state: phase = 0
        init_state = self._quantize_phase_to_state(0.0, phase_grid)
        path_metrics[init_state] = 0.0

        # Survivor memory for traceback
        predecessor_state = np.full((n_symbols_in_signal, n_phase_states), -1, dtype=int)
        predecessor_symbol_index = np.full((n_symbols_in_signal, n_phase_states), -1, dtype=int)

        for k in range(n_symbols_in_signal):
            start_index = k * sps
            end_index = start_index + sps
            if end_index > len(rx_signal):
                break

            r_seg = rx_signal[start_index:end_index]
            r_norm = np.linalg.norm(r_seg)

            next_metrics = np.full(n_phase_states, -np.inf, dtype=float)
            next_predecessor_state = np.full(n_phase_states, -1, dtype=int)
            next_predecessor_symbol_index = np.full(n_phase_states, -1, dtype=int)

            for state_idx in range(n_phase_states):
                current_metric = path_metrics[state_idx]
                if not np.isfinite(current_metric):
                    continue

                phase_state = phase_grid[state_idx]

                for symbol_idx, candidate_symbol in enumerate(constellation_points):
                    m = candidate_symbol * pulse_shape

                    # Hypothesized phase trajectory over the current symbol
                    phi = phase_state + 2.0 * np.pi * h * np.cumsum(m) * dt
                    modulated_candidate = np.exp(1j * phi)

                    cand_norm = np.linalg.norm(modulated_candidate)
                    denom = r_norm * cand_norm

                    if denom < 1e-12:
                        branch_metric = -np.inf
                    else:
                        branch_metric = np.real(np.vdot(modulated_candidate, r_seg)) / denom

                    # End-of-symbol phase
                    end_phase = phase_state + 2.0 * np.pi * h * np.sum(m) * dt
                    end_phase = self._wrap_phase(end_phase)

                    next_state_idx = self._quantize_phase_to_state(end_phase, phase_grid)
                    total_metric = current_metric + branch_metric

                    if total_metric > next_metrics[next_state_idx]:
                        next_metrics[next_state_idx] = total_metric
                        next_predecessor_state[next_state_idx] = state_idx
                        next_predecessor_symbol_index[next_state_idx] = symbol_idx

            path_metrics = next_metrics
            predecessor_state[k, :] = next_predecessor_state
            predecessor_symbol_index[k, :] = next_predecessor_symbol_index

        # Final best state
        final_state = int(np.argmax(path_metrics))
        if not np.isfinite(path_metrics[final_state]):
            raise RuntimeError("Viterbi failed: no valid survivor path found.")

        # Traceback
        decided_symbol_indices = np.full(n_symbols_in_signal, -1, dtype=int)
        state = final_state

        for k in range(n_symbols_in_signal - 1, -1, -1):
            sym_idx = predecessor_symbol_index[k, state]
            prev_state = predecessor_state[k, state]

            if sym_idx < 0 or prev_state < 0:
                raise RuntimeError(
                    f"Traceback failed at symbol {k}. "
                    "This usually means the trellis became disconnected."
                )

            decided_symbol_indices[k] = sym_idx
            state = prev_state

        decided_symbols = constellation_points[decided_symbol_indices]
        decided_symbols = np.asarray(decided_symbols, dtype=constellation_points.dtype)

        bits = self._coding_instance.generate_symbols_to_bits(decided_symbols, constellation_points)
        return bits

    def non_coherent_demodulate(
        self,
        rx_signal: np.ndarray,
        symbol_time: float,
        sample_rate: float,
        frequency_sensitivity: float = 0.5
    ) -> np.ndarray:
        """
        Non-coherent detector:
        - For linear modulation without FM: envelope/magnitude matched-filter approximation
        - For CPFSK/FM: symbol-by-symbol non-coherent detection using |correlation|
        """

        rx_signal = np.asarray(rx_signal)
        sps = int(symbol_time * sample_rate)

        constellation_points = self._constellation_instance.generate_constellation_points()
        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)

        h = frequency_sensitivity
        dt = 1.0 / sample_rate

        n_expected_symbols = int(np.floor((len(rx_signal) - len(pulse_shape) + 1) / sps))
        n_expected_symbols = max(n_expected_symbols, 0)

        decided_symbols = []
        phase_state = 0.0
        one_symbol_pulse = pulse_shape[:sps]

        for k in range(n_expected_symbols):
            start = k * sps
            stop = start + sps

            if stop > len(rx_signal):
                break

            r_seg = rx_signal[start:stop]
            if len(r_seg) != sps:
                break

            best_metric = -np.inf
            best_symbol = constellation_points[0]

            for a in constellation_points:
                m = a * one_symbol_pulse
                phi = phase_state + 2.0 * np.pi * h * np.cumsum(m) * dt
                s_ref = np.exp(1j * phi)

                denom = np.linalg.norm(r_seg) * np.linalg.norm(s_ref)
                if denom < 1e-12:
                    metric = -np.inf
                else:
                    metric = np.abs(np.vdot(s_ref, r_seg)) / denom

                if metric > best_metric:
                    best_metric = metric
                    best_symbol = a

            decided_symbols.append(best_symbol)

            # decision-directed phase update
            m_best = best_symbol * one_symbol_pulse
            phase_state = phase_state + 2.0 * np.pi * h * np.sum(m_best) * dt
            phase_state = np.angle(np.exp(1j * phase_state))

        decided_symbols = np.asarray(decided_symbols, dtype=constellation_points.dtype)
        bits = self._coding_instance.generate_symbols_to_bits(decided_symbols, constellation_points)
        return bits