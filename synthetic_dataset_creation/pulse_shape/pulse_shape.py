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
        else:
            raise ValueError(f"Unknown pulse shape type: {self.config.pulse_shape_type}")

    def _generate_rect_pulse(self, sample_rate: float, symbol_time: float) -> np.ndarray:
        rect_pulse = np.ones(round(sample_rate * symbol_time * self.config.span_in_symbols), dtype=np.float32)
        normalized_pulse = self._normalize_pulse(rect_pulse, sample_rate).astype(np.float32)
        return normalized_pulse

    def _generate_rrc_pulse(self, sample_rate: float, symbol_time: float) -> np.ndarray:
        beta = self.config.rolloff
        span = self.config.span_in_symbols

        sps = int(round(sample_rate * symbol_time))
        N = span * sps

        # time axis centered around zero
        t = (np.arange(N) - N // 2) / sample_rate

        g = np.zeros_like(t)

        for i in range(len(t)):
            ti = t[i]

            if abs(ti) < 1e-12:
                g[i] = (1.0 + beta * (4 / np.pi - 1)) / symbol_time

            elif beta > 0 and abs(abs(ti) - symbol_time / (4 * beta)) < 1e-12:
                g[i] = (
                    beta
                    / (symbol_time * np.sqrt(2))
                    * (
                        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                    )
                )

            else:
                numerator = (
                    np.sin(np.pi * ti * (1 - beta) / symbol_time)
                    + 4 * beta * ti / symbol_time * np.cos(np.pi * ti * (1 + beta) / symbol_time)
                )

                denominator = (
                    np.pi * ti * (1 - (4 * beta * ti / symbol_time) ** 2)
                )

                g[i] = numerator / denominator

        # Make it causal (important for your demodulator)
        g = np.roll(g, N // 2)

        return self._normalize_pulse(g, sample_rate).astype(np.float32)

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
