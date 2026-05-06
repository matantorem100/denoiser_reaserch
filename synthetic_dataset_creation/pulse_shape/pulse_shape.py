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
        elif self.config.pulse_shape_type == "rc":
            rc_pulse = self._generate_raised_cosine_frequency_pulse(sample_rate, symbol_time)
            return rc_pulse
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

        # Symmetric time axis: from -span*T/2 to +span*T/2
        num_samples = span * sps + 1
        t = (np.arange(num_samples) - num_samples // 2) / sample_rate
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

        return self._normalize_pulse(pulse, sample_rate)

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
