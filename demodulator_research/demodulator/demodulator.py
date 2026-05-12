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
        elif method == "semi_coherent":
            return self.semi_coherent_demodulate(rx_signal, symbol_time, sample_rate, frequency_sensitivity)
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


    def semi_coherent_demodulate(
            self,
            rx_signal: np.ndarray,
            symbol_time: float,
            sample_rate: float,
            frequency_sensitivity: float = 0.5,
            memory_depth_symbols: int | None = None,
            alpha: float = 0.15,
            beta: float = 0.01,
            use_abs_metric: bool = False,
            initial_phase: float = 0.0,
            initial_frequency_hz: float = 0.0,
            return_trace: bool = False,
    ) -> np.ndarray:
        """
        Semi-coherent CPFSK/FM detector with decision-directed phase/frequency tracking
        and finite pulse-memory support.

        This is NOT Viterbi / MLSE.

        It tests only the current candidate symbol, but builds the reference waveform
        using:
            current candidate symbol
            + previous already-decided symbols that still affect the current segment.

        Assumes the TX pulse behaves like a causal frequency pulse:
            shaped_frequency[n] = sum_k a[k] * pulse[n - k*sps]

        Parameters
        ----------
        rx_signal:
            Complex received CPFSK/FM signal.

        symbol_time:
            Symbol duration in seconds.

        sample_rate:
            Sample rate in Hz.

        frequency_sensitivity:
            CPFSK/FM sensitivity h.

        memory_depth_symbols:
            Number of previous decided symbols to include.
            If None, inferred from pulse length.

            Example:
                rect pulse length = sps -> memory depth = 0
                pulse length = 8*sps -> memory depth = 7 or 8 depending convention

        alpha:
            Phase tracking gain.

        beta:
            Frequency tracking gain.

        use_abs_metric:
            If True, uses abs(correlation) as metric.
            If False, uses real(correlation), more coherent but more phase-sensitive.
        """

        rx_signal = np.asarray(rx_signal, dtype=np.complex128).reshape(-1)

        sps = int(round(symbol_time * sample_rate))
        if sps <= 0:
            raise ValueError("samples per symbol must be positive")

        constellation_points = self._constellation_instance.generate_constellation_points()
        constellation_points = np.asarray(constellation_points)

        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)
        pulse_shape = np.asarray(pulse_shape, dtype=np.float64).reshape(-1)

        h = frequency_sensitivity
        dt = 1.0 / sample_rate

        # ------------------------------------------------------------
        # Split pulse into symbol-spaced chunks.
        #
        # chunk[0] affects the segment of the symbol currently being tested.
        # chunk[1] is the tail from the previous symbol.
        # chunk[2] is the tail from two symbols ago.
        # etc.
        # ------------------------------------------------------------
        n_chunks = int(np.ceil(len(pulse_shape) / sps))

        pulse_chunks = []
        for i in range(n_chunks):
            start = i * sps
            stop = start + sps

            chunk = pulse_shape[start:stop]

            # Pad last chunk to exactly one symbol.
            if len(chunk) < sps:
                chunk = np.pad(chunk, (0, sps - len(chunk)))

            pulse_chunks.append(chunk)

        pulse_chunks = np.asarray(pulse_chunks)  # shape: (n_chunks, sps)

        if memory_depth_symbols is None:
            # Number of previous symbols that can affect the current segment.
            # chunk[0] is current symbol, so previous-symbol memory is n_chunks - 1.
            memory_depth_symbols = n_chunks - 1

        memory_depth_symbols = int(memory_depth_symbols)
        if memory_depth_symbols < 0:
            raise ValueError("memory_depth_symbols must be non-negative")

        # We cannot use more previous-symbol chunks than the pulse actually has.
        memory_depth_symbols = min(memory_depth_symbols, n_chunks - 1)

        n_symbols = len(rx_signal) // sps

        decided_symbols = []
        phase_errors_rad = []
        phase_states_rad = []
        residual_frequency_hz = []
        branch_metrics = []

        # CPFSK phase state at the beginning of the current symbol.
        phase_state = float(initial_phase)

        # Residual carrier frequency estimate in Hz.
        freq_state_hz = float(initial_frequency_hz)

        for k in range(n_symbols):
            seg_start = k * sps
            seg_stop = seg_start + sps

            r_seg = rx_signal[seg_start:seg_stop]
            if len(r_seg) != sps:
                break

            r_norm = np.linalg.norm(r_seg)

            best_metric = -np.inf
            best_symbol = constellation_points[0]
            best_ref = None
            best_m_seg = None

            # ------------------------------------------------------------
            # Try each current-symbol candidate.
            # Previous symbols are fixed using decision-directed decisions.
            # ------------------------------------------------------------
            for a_candidate in constellation_points:

                # Current frequency segment m[n] for this one-symbol interval.
                m_seg = np.zeros(sps, dtype=np.float64)

                # Current candidate contribution.
                m_seg += np.real(a_candidate) * pulse_chunks[0]

                # Previous decided-symbol tail contributions.
                #
                # d = 1 means previous symbol a[k-1] with pulse_chunks[1]
                # d = 2 means a[k-2] with pulse_chunks[2]
                # etc.
                max_d = min(memory_depth_symbols, len(decided_symbols), n_chunks - 1)

                for d in range(1, max_d + 1):
                    previous_symbol = decided_symbols[-d]
                    m_seg += np.real(previous_symbol) * pulse_chunks[d]

                # CPFSK phase contribution over the current segment.
                symbol_phase = 2.0 * np.pi * h * np.cumsum(m_seg) * dt

                # Residual CFO contribution over current segment.
                n = np.arange(sps)
                freq_phase = 2.0 * np.pi * freq_state_hz * n / sample_rate

                phi = phase_state + symbol_phase + freq_phase
                s_ref = np.exp(1j * phi)

                denom = r_norm * np.linalg.norm(s_ref)

                if denom < 1e-12:
                    metric = -np.inf
                else:
                    corr = np.vdot(s_ref, r_seg)

                    if use_abs_metric:
                        metric = np.abs(corr) / denom
                    else:
                        metric = np.real(corr) / denom

                if metric > best_metric:
                    best_metric = metric
                    best_symbol = a_candidate
                    best_ref = s_ref
                    best_m_seg = m_seg

            decided_symbols.append(best_symbol)

            # ------------------------------------------------------------
            # PLL / residual CFO update from the chosen branch.
            # ------------------------------------------------------------
            corr = np.vdot(best_ref, r_seg)
            phase_error = np.angle(corr)

            # Frequency update in Hz.
            freq_state_hz += beta * phase_error / (2.0 * np.pi * symbol_time)

            # Phase-state update:
            # integrate the actual selected m_seg over this symbol.
            selected_phase_increment = 2.0 * np.pi * h * np.sum(best_m_seg) * dt

            phase_state += selected_phase_increment
            phase_state += 2.0 * np.pi * freq_state_hz * symbol_time
            phase_state += alpha * phase_error

            # Keep phase bounded.
            phase_state = np.angle(np.exp(1j * phase_state))

            if return_trace:
                phase_errors_rad.append(float(phase_error))
                phase_states_rad.append(float(phase_state))
                residual_frequency_hz.append(float(freq_state_hz))
                branch_metrics.append(float(best_metric))

        decided_symbols = np.asarray(decided_symbols, dtype=constellation_points.dtype)

        bits = self._coding_instance.generate_symbols_to_bits(
            decided_symbols,
            constellation_points,
        )

        if return_trace:
            trace = {
                "phase_errors_rad": np.asarray(phase_errors_rad, dtype=np.float64),
                "phase_states_rad": np.asarray(phase_states_rad, dtype=np.float64),
                "residual_frequency_hz": np.asarray(residual_frequency_hz, dtype=np.float64),
                "branch_metrics": np.asarray(branch_metrics, dtype=np.float64),
            }
            return bits, trace

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
                        if len(modulated_candidate) % 2 == 1:
                            r_seg = rx_signal[start_index:end_index + 1]
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
