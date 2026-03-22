from typing import Literal

import numpy as np
import pydantic

class ConstellationConfig(pydantic.BaseModel):
    constellation_type: Literal["PAM"]

class Constellation:
    def __init__(self, config: ConstellationConfig):
        self.config = config

    def generate_constellation_points(self, constellation_order: int):
        if self.config.constellation_type == "PAM":
            constellation_points = self.generate_pam_constellation(constellation_order)
            return constellation_points
        else:
            raise ValueError(f"Unknown constellation type: {self.config.constellation_type}")

    @staticmethod
    def generate_pam_constellation(constellation_order: int):
        constellation_points = 2 * (np.arange(constellation_order) + 1) - 1 - constellation_order
        return constellation_points



