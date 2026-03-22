from typing import Literal

import numpy as np
import pydantic

class ConstellationConfig(pydantic.BaseModel):
    constellation_type: Literal["PAM"]
    constellation_order: int

    @pydantic.field_validator("constellation_order")
    @classmethod
    def validate_constellation_order(cls, v: int) -> int:
        if v < 2:
            raise ValueError("constellation_order must be at least 2")
        if v & (v - 1) != 0:
            raise ValueError("constellation_order must be a power of 2")
        return v

class Constellation:
    def __init__(self, config: ConstellationConfig):
        self.config = config

    def generate_constellation_points(self):
        if self.config.constellation_type == "PAM":
            constellation_points = self.generate_pam_constellation()
            return constellation_points
        else:
            raise ValueError(f"Unknown constellation type: {self.config.constellation_type}")

    def generate_pam_constellation(self):
        constellation_points = 2 * (np.arange(self.config.constellation_order) + 1) - 1 - self.config.constellation_order
        return constellation_points



