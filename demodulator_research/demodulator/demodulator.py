from fractions import Fraction
from itertools import product
from typing import Optional, Literal, Tuple, Dict, List

import numpy as np
import pydantic

from synthetic_dataset_creation.coding.coding import Coding, CodingConfig
from synthetic_dataset_creation.constellation.constellation import Constellation, ConstellationConfig
from synthetic_dataset_creation.modulator.modulator import FMConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShape, PulseShapeConfig


class DemodulatorConfig(pydantic.BaseModel):
    pulse_shape_config: PulseShapeConfig
    constellation_config: ConstellationConfig
    fm_config: FMConfig


class Demodulator:
    def __init__(self, config: DemodulatorConfig):
        self.config = config
        self._pulse_shape_instance = PulseShape(config.pulse_shape_config)
        self._constellation_instance = Constellation(config.constellation_config)
        self._coding_instance = Coding(CodingConfig(constellation=config.constellation_config))


    def demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float,
                   method: str = "differentiate") -> np.ndarray:

        if method == "differentiate":
            return self.differentiate_demodulate(rx_signal, symbol_time, sample_rate)
        elif method == "coherent":
            return self.coherent_demodulate(rx_signal, symbol_time, sample_rate)
        elif method == "non_coherent":
            return self.non_coherent_demodulate(rx_signal, symbol_time, sample_rate)
        else:
            raise ValueError(f"Unknown demodulation method: {method}")

    def differentiate_demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float) -> np.ndarray:

        sps = int(symbol_time * sample_rate)

        rx_signal = (np.diff(np.unwrap(np.angle(rx_signal))) /
                     (2.0 * np.pi * self.config.fm_config.frequency_sensitivity) * sample_rate)

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



    def coherent_demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float) -> np.ndarray:
        """
        Coherent symbol-by-symbol CPM/CPFSK demodulator that supports both:
          - full-response pulses
          - partial-response pulses

        Important:
        - This is NOT Viterbi / ML sequence detection.
        - For partial-response CPM, this is a suboptimal decision-directed detector:
          it uses previously decided symbols as the memory state.

        Assumptions:
        - pulse_shape is a *causal* frequency pulse g[n] starting at symbol onset.
        - If len(pulse_shape) = L * sps, then the CPM memory length is L symbols.
        """
        rx_signal = rx_signal.astype(np.complex64)

        sps = int(round(symbol_time * sample_rate))

        constellation_points = self._constellation_instance.generate_constellation_points()

        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)

        h = self.config.fm_config.frequency_sensitivity
        dt = 1.0 / sample_rate

        # Pad pulse to an integer number of symbol intervals
        pulse_len = pulse_shape.size
        memory_symbols = int(np.ceil(pulse_len / sps))

        padded_len = memory_symbols * sps
        if padded_len != pulse_len:
            pulse_shape = np.pad(pulse_shape, (0, padded_len - pulse_len))
            pulse_len = pulse_shape.size

        # Split pulse into symbol-length chunks: pulse_chunks[i] is the contribution during the current interval
        # from a symbol that occurred i symbols ago. i = 0  -> current symbol contribution i = 1  -> previous symbol contribution
        pulse_chunks = pulse_shape.reshape(memory_symbols, sps)

        # Number of complete symbol intervals in the received signal
        n_symbols_in_signal = rx_signal.size // sps

        decided_symbols = []

        # Phase at the START of the current symbol interval
        phase_state = 0.0

        # Store previously decided symbols:
        # history[0] = a_{k-1}, history[1] = a_{k-2}, ...
        symbol_history = []

        def build_interval_frequency_segment(current_symbol: complex) -> np.ndarray:
            """
            Build the instantaneous frequency/control signal m[n] over the CURRENT
            symbol interval, including contributions from:
              - current_symbol
              - previous decided symbols in symbol_history
            """
            m_seg = np.zeros(sps, dtype=np.complex64)

            # Current symbol contribution
            m_seg += current_symbol * pulse_chunks[0]

            # Previous-symbol contributions
            max_prev = min(len(symbol_history), memory_symbols - 1)
            for i in range(1, max_prev + 1):
                # symbol_history[i-1] is the symbol from i intervals ago
                m_seg += symbol_history[i - 1] * pulse_chunks[i]

            return m_seg

        for current_symbol_idx in range(n_symbols_in_signal):
            start = current_symbol_idx * sps
            end = start + sps
            r_seg = rx_signal[start:end]

            best_metric = -np.inf
            best_symbol = constellation_points[0]
            best_m_seg = None

            for candidate_symbol in constellation_points:
                m_seg = build_interval_frequency_segment(candidate_symbol)

                phi = phase_state + 2.0 * np.pi * h * np.cumsum(m_seg) * dt
                candidate_waveform = np.exp(1j * phi)

                denom = np.linalg.norm(r_seg) * np.linalg.norm(candidate_waveform)
                if denom < 1e-12:
                    metric = -np.inf
                else:
                    metric = np.real(np.vdot(candidate_waveform, r_seg)) / denom

                if metric > best_metric:
                    best_metric = metric
                    best_symbol = candidate_symbol
                    best_m_seg = m_seg

            decided_symbols.append(best_symbol)

            # Advance phase through the chosen current interval
            phase_state += 2.0 * np.pi * h * np.sum(best_m_seg) * dt
            phase_state = np.angle(np.exp(1j * phase_state))

            # Update symbol memory:
            # newest previous symbol first
            symbol_history.insert(0, best_symbol)
            if len(symbol_history) > memory_symbols - 1:
                symbol_history.pop()

        decided_symbols = np.asarray(decided_symbols, dtype=constellation_points.dtype)
        bits = self._coding_instance.generate_symbols_to_bits(
            decided_symbols,
            constellation_points
        )
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

    @staticmethod
    def _get_phase_grid_from_h(h: float, max_denominator: int = 1000):
        frac = Fraction(h).limit_denominator(max_denominator)
        p = frac.numerator
        q = frac.denominator

        # full-response CPM rule
        if p % 2 == 0:
            Q = q
        else:
            Q = 2 * q

        phase_grid = 2 * np.pi * np.arange(Q) / Q
        return phase_grid

    def coherent_fm_viterbi_demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float) -> np.ndarray:
        """
        True Viterbi detector for full-response FM/CPFSK-like signal.

        State = quantized phase at symbol boundary.
        Branch metric = normalized correlation with the candidate waveform.
        """
        h = self.config.fm_config.frequency_sensitivity
        dt = 1.0 / sample_rate
        sps = int(symbol_time * sample_rate)

        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)
        constellation_points = self._constellation_instance.generate_constellation_points()

        n_symbols_in_signal = int(np.floor((len(rx_signal) - len(pulse_shape) + 1) / sps))

        phase_grid = self._get_phase_grid_from_h(h)
        n_phase_states = len(phase_grid)

        # Path metric at current time: one metric per state
        path_metrics = np.full(n_phase_states, -np.inf, dtype=float)

        # Initial state: phase = 0
        init_state = self._quantize_phase_to_state(0.0, phase_grid)
        path_metrics[init_state] = 0.0

        # Survivor memory for traceback
        predecessor_state = np.full((n_symbols_in_signal, n_phase_states), -1, dtype=int)
        predecessor_symbol_index = np.full((n_symbols_in_signal, n_phase_states), -1, dtype=int)

        for symbol_index in range(n_symbols_in_signal):
            start_index = symbol_index * sps
            end_index = start_index + sps

            if end_index > len(rx_signal):
                break

            r_seg = rx_signal[start_index: end_index]
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
            predecessor_state[symbol_index, :] = next_predecessor_state
            predecessor_symbol_index[symbol_index, :] = next_predecessor_symbol_index

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
                raise RuntimeError(f"Traceback failed at symbol {k}. ""This usually means the trellis became disconnected.")

            decided_symbol_indices[k] = sym_idx
            state = prev_state

        decided_symbols = constellation_points[decided_symbol_indices]
        decided_symbols = np.asarray(decided_symbols, dtype=constellation_points.dtype)

        bits = self._coding_instance.generate_symbols_to_bits(decided_symbols, constellation_points)
        return bits


    @staticmethod
    def _get_memory_length_and_segments(pulse_shape: np.ndarray, sps: int) -> Tuple[int, np.ndarray]:
        """
        Returns:
            L: CPM memory length in symbols
            pulse_segments: shape (L, sps), where pulse_segments[i] is the
                            i-th symbol interval of the frequency pulse
        """
        pulse_shape = np.asarray(pulse_shape).reshape(-1)

        if len(pulse_shape) % sps != 0:
            raise ValueError(
                f"pulse_shape length ({len(pulse_shape)}) must be an integer multiple "
                f"of samples-per-symbol ({sps})."
            )

        L = len(pulse_shape) // sps
        if L < 1:
            raise ValueError("Invalid CPM pulse length.")

        pulse_segments = pulse_shape.reshape(L, sps)
        return L, pulse_segments

    @staticmethod
    def _build_memory_states(
        constellation_points: np.ndarray,
        memory_length: int,
        include_startup_zero: bool = True,
    ) -> Tuple[List[Tuple[complex, ...]], Dict[Tuple[complex, ...], int]]:
        """
        For partial-response CPM, we need memory of the previous L-1 symbols.

        During startup, before enough symbols have entered the pulse memory,
        zero-history states are physically meaningful, so we optionally include 0
        as a special startup symbol in the memory alphabet.
        """
        if memory_length == 0:
            states = [tuple()]
            return states, {tuple(): 0}

        if include_startup_zero:
            alphabet = [0.0] + list(constellation_points)
        else:
            alphabet = list(constellation_points)

        states = list(product(alphabet, repeat=memory_length))
        state_to_index = {state: idx for idx, state in enumerate(states)}
        return states, state_to_index

    @staticmethod
    def _compose_interval_frequency_waveform(
        state_memory: Tuple[complex, ...],
        candidate_symbol: complex,
        pulse_segments: np.ndarray,
    ) -> np.ndarray:
        """
        Build the instantaneous frequency/control waveform over ONE symbol interval.

        pulse_segments shape = (L, sps)
        state_memory contains the previous L-1 symbols in chronological order:
            (a_{k-L+1}, ..., a_{k-1})
        candidate_symbol = a_k

        Over interval k, the active symbols are:
            [a_{k-L+1}, ..., a_{k-1}, a_k]
        with:
            a_k      using pulse_segments[0]
            a_{k-1}  using pulse_segments[1]
            ...
            a_{k-L+1} using pulse_segments[L-1]
        """
        active_symbols = list(state_memory) + [candidate_symbol]
        L, sps = pulse_segments.shape

        if len(active_symbols) != L:
            raise ValueError("Active symbol history length does not match CPM pulse memory.")

        m = np.zeros(sps, dtype=np.complex128)

        # newest symbol uses pulse segment 0, oldest uses segment L-1
        for delay in range(L):
            symbol = active_symbols[-1 - delay]
            m += symbol * pulse_segments[delay]

        return m


    def coherent_cpm_viterbi_demodulate(
        self,
        rx_signal: np.ndarray,
        symbol_time: float,
        sample_rate: float,
    ) -> np.ndarray:
        """
        Coherent Viterbi detector for binary/M-ary CPM/CPFSK.

        Supports:
          - full-response CPM  (L = 1)
          - partial-response CPM (L > 1)

        State:
          (phase_state, previous L-1 symbols)

        Branch metric:
          normalized coherent correlation with the candidate CPM waveform over
          one symbol interval.
        """
        h = self.config.fm_config.frequency_sensitivity
        dt = 1.0 / sample_rate
        sps = int(round(symbol_time * sample_rate))

        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)

        constellation_points = self._constellation_instance.generate_constellation_points()

        phase_grid = self._get_phase_grid_from_h(h)
        n_phase_states = len(phase_grid)

        L, pulse_segments = self._get_memory_length_and_segments(pulse_shape, sps)
        memory_length = L - 1

        memory_states, memory_state_to_idx = self._build_memory_states(constellation_points=constellation_points,
                                                                       memory_length=memory_length,
                                                                       include_startup_zero=True)
        n_memory_states = len(memory_states)

        # Number of whole symbol intervals available in rx_signal
        n_symbols_in_signal = len(rx_signal) // sps

        # path metric indexed by [phase_state, memory_state]
        path_metrics = np.full((n_phase_states, n_memory_states), -np.inf, dtype=float)

        # initial condition: zero phase, zero memory
        init_phase_idx = self._quantize_phase_to_state(0.0, phase_grid)
        init_memory = tuple([0.0] * memory_length)
        init_memory_idx = memory_state_to_idx[init_memory]
        path_metrics[init_phase_idx, init_memory_idx] = 0.0

        predecessor_phase = np.full(
            (n_symbols_in_signal, n_phase_states, n_memory_states), -1, dtype=int
        )
        predecessor_memory = np.full(
            (n_symbols_in_signal, n_phase_states, n_memory_states), -1, dtype=int
        )
        predecessor_symbol_index = np.full(
            (n_symbols_in_signal, n_phase_states, n_memory_states), -1, dtype=int
        )

        for k in range(n_symbols_in_signal):
            start = k * sps
            end = start + sps
            r_seg = rx_signal[start:end]
            r_norm = np.linalg.norm(r_seg)

            next_metrics = np.full((n_phase_states, n_memory_states), -np.inf, dtype=float)

            for phase_idx in range(n_phase_states):
                phase_state = phase_grid[phase_idx]

                for mem_idx, state_memory in enumerate(memory_states):
                    current_metric = path_metrics[phase_idx, mem_idx]
                    if not np.isfinite(current_metric):
                        continue

                    for symbol_idx, candidate_symbol in enumerate(constellation_points):
                        # waveform over this symbol interval from all active symbols
                        m = self._compose_interval_frequency_waveform(
                            state_memory=state_memory,
                            candidate_symbol=candidate_symbol,
                            pulse_segments=pulse_segments,
                        )

                        # phase trajectory during this interval
                        phi = phase_state + 2.0 * np.pi * h * np.cumsum(m) * dt
                        s_hyp = np.exp(1j * phi)

                        cand_norm = np.linalg.norm(s_hyp)
                        denom = r_norm * cand_norm

                        if denom < 1e-12:
                            branch_metric = -np.inf
                        else:
                            branch_metric = np.real(np.vdot(s_hyp, r_seg)) / denom

                        # end phase at next symbol boundary
                        end_phase = phase_state + 2.0 * np.pi * h * np.sum(m) * dt
                        end_phase = self._wrap_phase(end_phase)
                        next_phase_idx = self._quantize_phase_to_state(end_phase, phase_grid)

                        # update memory: shift left and append current symbol
                        if memory_length == 0:
                            next_memory = tuple()
                        else:
                            next_memory = tuple(list(state_memory[1:]) + [candidate_symbol])

                        next_mem_idx = memory_state_to_idx[next_memory]

                        total_metric = current_metric + branch_metric

                        if total_metric > next_metrics[next_phase_idx, next_mem_idx]:
                            next_metrics[next_phase_idx, next_mem_idx] = total_metric
                            predecessor_phase[k, next_phase_idx, next_mem_idx] = phase_idx
                            predecessor_memory[k, next_phase_idx, next_mem_idx] = mem_idx
                            predecessor_symbol_index[k, next_phase_idx, next_mem_idx] = symbol_idx

            path_metrics = next_metrics

        # final best state
        flat_best = int(np.argmax(path_metrics))
        final_phase_idx, final_mem_idx = np.unravel_index(flat_best, path_metrics.shape)

        if not np.isfinite(path_metrics[final_phase_idx, final_mem_idx]):
            raise RuntimeError("Viterbi failed: no valid survivor path found.")

        # traceback
        decided_symbol_indices = np.full(n_symbols_in_signal, -1, dtype=int)

        phase_idx = final_phase_idx
        mem_idx = final_mem_idx

        for k in range(n_symbols_in_signal - 1, -1, -1):
            sym_idx = predecessor_symbol_index[k, phase_idx, mem_idx]
            prev_phase_idx = predecessor_phase[k, phase_idx, mem_idx]
            prev_mem_idx = predecessor_memory[k, phase_idx, mem_idx]

            if sym_idx < 0 or prev_phase_idx < 0 or prev_mem_idx < 0:
                raise RuntimeError(
                    f"Traceback failed at symbol {k}. "
                    "The trellis likely became disconnected."
                )

            decided_symbol_indices[k] = sym_idx
            phase_idx = prev_phase_idx
            mem_idx = prev_mem_idx

        decided_symbols = constellation_points[decided_symbol_indices]
        bits = self._coding_instance.generate_symbols_to_bits(
            decided_symbols,
            constellation_points
        )
        return bits