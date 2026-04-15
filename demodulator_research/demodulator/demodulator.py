from typing import Optional, Literal

import numpy as np
import pydantic

from synthetic_dataset_creation.coding.coding import Coding, CodingConfig
from synthetic_dataset_creation.constellation.constellation import Constellation, ConstellationConfig
from synthetic_dataset_creation.modulator.modulator import FMConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShape, PulseShapeConfig


class DemodulatorConfig(pydantic.BaseModel):
    pulse_shape_config: PulseShapeConfig
    constellation_config: ConstellationConfig
    fm_config: Optional[FMConfig] = None


class Demodulator:
    def __init__(self, config: DemodulatorConfig):
        self.config = config
        self._pulse_shape_instance = PulseShape(config.pulse_shape_config)
        self._constellation_instance = Constellation(config.constellation_config)
        self._coding_instance = Coding(CodingConfig(constellation=config.constellation_config))


    def demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float, apply_fm: bool = False,
                   method: str = "differentiate") -> np.ndarray:

        if method == "differentiate":
            return self.differentiate_demodulate(rx_signal, symbol_time, sample_rate, apply_fm)
        elif method == "coherent":
            return self.coherent_demodulate(rx_signal, symbol_time, sample_rate, apply_fm)
        elif method == "non_coherent":
            return self.non_coherent_demodulate(rx_signal, symbol_time, sample_rate, apply_fm)
        else:
            raise ValueError(f"Unknown demodulation method: {method}")

    def differentiate_demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float, apply_fm: bool = False) -> np.ndarray:

        sps = int(symbol_time * sample_rate)

        if apply_fm:
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

    def coherent_demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float, apply_fm: bool = False) -> np.ndarray:

        rx_signal = np.asarray(rx_signal)
        sps = int(symbol_time * sample_rate)

        constellation_points = self._constellation_instance.generate_constellation_points()
        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)

        if not apply_fm:
            matched_filter = pulse_shape[::-1].conj()
            mf_energy = np.sum(np.abs(matched_filter) ** 2)

            filtered_signal = np.convolve(rx_signal, matched_filter, mode="full") / mf_energy
            sampling_offset = len(pulse_shape) - 1
            detected_symbols = filtered_signal[sampling_offset::sps]

            n_expected_symbols = int(np.ceil((len(rx_signal) - len(pulse_shape) + 1) / sps))
            n_expected_symbols = max(n_expected_symbols, 0)
            detected_symbols = detected_symbols[:n_expected_symbols]

            bits = self._coding_instance.generate_symbols_to_bits(detected_symbols, constellation_points)
            return bits

        if self.config.fm_config is None:
            raise ValueError("FM config must be provided when apply_fm=True")

        h = self.config.fm_config.frequency_sensitivity
        dt = 1.0 / sample_rate

        n_expected_symbols = int(np.floor((len(rx_signal) - len(pulse_shape) + 1) / sps))

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
                    metric = np.real(np.vdot(s_ref, r_seg)) / denom

                if metric > best_metric:
                    best_metric = metric
                    best_symbol = a

            decided_symbols.append(best_symbol)

            m_best = best_symbol * one_symbol_pulse
            phase_state = phase_state + 2.0 * np.pi * h * np.sum(m_best) * dt
            phase_state = np.angle(np.exp(1j * phase_state))

        decided_symbols = np.asarray(decided_symbols, dtype=constellation_points.dtype)
        bits = self._coding_instance.generate_symbols_to_bits(decided_symbols, constellation_points)
        return bits

    def non_coherent_demodulate(
        self,
        rx_signal: np.ndarray,
        symbol_time: float,
        sample_rate: float,
        apply_fm: bool = False
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

        if not apply_fm:
            # For real linear PAM this is not the usual preferred detector,
            # but this gives a non-coherent-style comparison based on magnitude.
            matched_filter = pulse_shape[::-1].conj()
            mf_energy = np.sum(np.abs(matched_filter) ** 2)

            filtered_signal = np.convolve(rx_signal, matched_filter, mode="full") / mf_energy
            sampling_offset = len(pulse_shape) - 1
            detected_symbols = filtered_signal[sampling_offset::sps]

            n_expected_symbols = int(np.ceil((len(rx_signal) - len(pulse_shape) + 1) / sps))
            n_expected_symbols = max(n_expected_symbols, 0)
            detected_symbols = detected_symbols[:n_expected_symbols]

            # Magnitude-only decision:
            # compare |detected_symbols| to |constellation_points|, then restore sign from sample.
            abs_constellation = np.abs(constellation_points)
            detected_abs = np.abs(detected_symbols)

            detected_indices = np.argmin(
                np.abs(detected_abs[:, None] - abs_constellation[None, :]),
                axis=1
            )
            estimated_symbols = constellation_points[detected_indices]

            # crude sign restoration for real PAM
            estimated_symbols = np.sign(np.real(detected_symbols)) * np.abs(estimated_symbols)

            bits = self._coding_instance.generate_symbols_to_bits(estimated_symbols, constellation_points)
            return bits

        if self.config.fm_config is None:
            raise ValueError("FM config must be provided when apply_fm=True")

        h = self.config.fm_config.frequency_sensitivity
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