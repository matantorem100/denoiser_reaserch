import numpy as np
import pydantic

class PulseShapeConfig(pydantic.BaseModel):
    pulse_shape_type: str
    normalization_type: str
    rolloff: float = 0.25
    span_in_symbols: int = 1

class PulseShape:
    def __init__(self, config: PulseShapeConfig):
        self.config = config

    def generate_pulse_shape(self, sample_rate: float, symbol_time: float):
        if self.config.pulse_shape_type == "rect":
            rect_pulse = self._generate_rect_pulse(sample_rate, symbol_time)
            return rect_pulse
        elif self.config.pulse_shape_type == "rrc":
            rrc_pulse = self._generate_rrc_pulse(sample_rate, symbol_time)
            return rrc_pulse
        elif self.config.pulse_shape_type == "protocol":
            protocol_pulse = self._generate_protocol_pulse(sample_rate, symbol_time)
            return protocol_pulse
        else:
            raise ValueError(f"Unknown pulse shape type: {self.config.pulse_shape_type}")

    def _generate_rect_pulse(self, sample_rate: float, symbol_time: float) -> np.ndarray:
        rect_pulse = np.ones(round(sample_rate * symbol_time), dtype=np.float32)
        normalized_pulse = self._normalize_pulse(rect_pulse, sample_rate)
        return normalized_pulse

    def _generate_rrc_pulse(self, sample_rate: float, symbol_time: float) -> np.ndarray:
        beta = self.config.rolloff
        span = self.config.span_in_symbols

        if not (0 <= beta <= 1):
            raise ValueError("rolloff must satisfy 0 <= rolloff <= 1")

        sps = sample_rate * symbol_time
        if not np.isclose(sps, round(sps), atol=1e-10):
            raise ValueError("sample_rate * symbol_time must be an integer (samples per symbol)")

        sps = int(round(sps))
        if sps <= 0:
            raise ValueError("samples per symbol must be positive")
        if span <= 0:
            raise ValueError("span_in_symbols must be positive")

        # Build the usual centered RRC samples, then rotate them into a causal frequency pulse.
        # The semi-coherent detector expects chunk[0] to hold the current symbol's main contribution, not the weak left tail.
        num_samples = span * sps + 1
        center_index = num_samples // 2
        t = (np.arange(num_samples) - center_index) / sample_rate
        T = symbol_time

        pulse = np.zeros_like(t, dtype=np.float64)
        eps = 1e-12

        for i, ti in enumerate(t):
            # Special case: t = 0
            if abs(ti) < eps:
                pulse[i] = (1 / np.sqrt(T)) * (1 + beta * (4 / np.pi - 1))

            # Special case: t = ±T/(4β), only relevant when β > 0
            elif beta > 0 and abs(abs(ti) - T / (4 * beta)) < eps:
                pulse[i] = (beta / np.sqrt(2 * T)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                                      + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))

            else:
                numerator = (np.sin(np.pi * ti * (1 - beta) / T) + 4 * beta * ti / T * np.cos(np.pi * ti * (1 + beta) / T))
                denominator = (np.pi * ti / T * (1 - (4 * beta * ti / T) ** 2))
                pulse[i] = (1 / np.sqrt(T)) * (numerator / denominator)

        causal_pulse = np.roll(pulse, -center_index)

        return self._normalize_pulse(causal_pulse, sample_rate)

    def _generate_protocol_pulse(self, sample_rate: float, symbol_time: float) -> np.ndarray:
        beta = self.config.rolloff
        span = self.config.span_in_symbols
        sps = sample_rate * symbol_time

        if not (0 <= beta <= 1):
            raise ValueError("rolloff must satisfy 0 <= rolloff <= 1")
        if not np.isclose(sps, round(sps), atol=1e-10):
            raise ValueError("sample_rate * symbol_time must be an integer (samples per symbol)")

        sps = int(round(sps))
        if sps <= 0:
            raise ValueError("samples per symbol must be positive")
        if span <= 0:
            raise ValueError("span_in_symbols must be positive")

        n_taps = span * sps + 1
        fft_size = self._next_power_of_two(max(8 * n_taps, 1024))
        frequencies = np.fft.fftfreq(fft_size, d=1.0 / sample_rate)

        h1_time = self._frequency_response_to_time_filter(
            frequency_response=self._protocol_h1_response(frequencies, symbol_time, beta),
            n_taps=n_taps,
        )
        h2_time = self._frequency_response_to_time_filter(
            frequency_response=self._protocol_h2_response(frequencies, symbol_time),
            n_taps=n_taps,
        )

        pulse = np.convolve(h1_time, h2_time, mode="full")
        pulse = self._center_trim(pulse, n_taps)

        return self._normalize_pulse(pulse, sample_rate)

    @staticmethod
    def _protocol_h1_response(frequencies: np.ndarray, symbol_time: float, beta: float) -> np.ndarray:
        abs_f = np.abs(frequencies)
        lower = (1.0 - beta) / (2.0 * symbol_time)
        upper = (1.0 + beta) / (2.0 * symbol_time)

        response = np.zeros_like(abs_f, dtype=np.float64)
        response[abs_f <= lower] = 1.0

        if beta == 0:
            response[abs_f <= 1.0 / (2.0 * symbol_time)] = 1.0
            return response

        transition = (abs_f > lower) & (abs_f < upper)
        response[transition] = np.cos((symbol_time / (4.0 * beta)) * (2.0 * np.pi * abs_f[transition] - np.pi * (1.0 - beta) / symbol_time))
        return response

    @staticmethod
    def _protocol_h2_response(frequencies: np.ndarray, symbol_time: float) -> np.ndarray:
        x = frequencies * symbol_time
        return np.sinc(x)

    @staticmethod
    def _frequency_response_to_time_filter(frequency_response: np.ndarray, n_taps: int) -> np.ndarray:
        impulse_response = np.fft.fftshift(np.fft.ifft(frequency_response))
        impulse_response = np.real_if_close(impulse_response, tol=1000).real
        return PulseShape._center_trim(impulse_response, n_taps)

    @staticmethod
    def _center_trim(signal: np.ndarray, n_samples: int) -> np.ndarray:
        start = (len(signal) - n_samples) // 2
        stop = start + n_samples
        return signal[start:stop]

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 << (int(value) - 1).bit_length()

    def _normalize_pulse(self, pulse: np.ndarray, sample_rate: float) -> np.ndarray:
        if self.config.normalization_type == "None":
            return pulse
        elif self.config.normalization_type == "norm_2":
            normalized_pulse = pulse / np.linalg.norm(pulse, ord=2)
            return normalized_pulse
        elif self.config.normalization_type == "cpfsk":
            dt = 1 / sample_rate
            normalized_pulse = pulse * 0.5 / (np.sum(pulse) * dt)
            return normalized_pulse
        else:
            raise ValueError(f"Unknown normalization type: {self.config.normalization_type}")
