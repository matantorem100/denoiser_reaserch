from typing import Literal

import numpy as np
import pydantic

class PulseShapeConfig(pydantic.BaseModel):
    pulse_shape_type: Literal["rect"]
    normalization_type: str
class PulseShape:
    def __init__(self, config: PulseShapeConfig):
        self.config = config

    def generate_pulse_shape(self, sample_rate: float, symbol_time: float):
        if self.config.pulse_shape_type == "rect":
            rect_pulse = self.generate_rect_pulse(sample_rate, symbol_time)
            return rect_pulse
        else:
            raise ValueError(f"Unknown pulse shape type: {self.config.pulse_shape_type}")

    def generate_rect_pulse(self, sample_rate: float, symbol_time: float) -> np.ndarray:
        rect_pulse = np.ones(round(sample_rate * symbol_time), dtype=np.float32)
        normalized_pulse = self.normalize_pulse(rect_pulse, sample_rate)
        return normalized_pulse

    def normalize_pulse(self, pulse: np.ndarray, sample_rate: float) -> np.ndarray:
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
