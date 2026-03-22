import math
from typing import Dict, Tuple, Union

import numpy as np
import pydantic

from constellation.constellation import Constellation, ConstellationConfig

BitTuple = Tuple[int, ...]


class CodingConfig(pydantic.BaseModel):
    constellation: ConstellationConfig


class Coding:
    def __init__(self, config: CodingConfig):
        self.config = config

    @staticmethod
    def _binary_to_gray(n: int) -> int:
        return n ^ (n >> 1)

    @staticmethod
    def _gray_to_binary(n: int) -> int:
        result = n
        while n > 0:
            n >>= 1
            result ^= n
        return result

    @staticmethod
    def _int_to_bits(value: int, bits_per_symbol: int) -> BitTuple:
        return tuple((value >> i) & 1 for i in reversed(range(bits_per_symbol)))

    @staticmethod
    def _bits_to_int(bits: BitTuple) -> int:
        value = 0
        for bit in bits:
            value = (value << 1) | bit
        return value

    def get_constellation_points(self) -> np.ndarray:
        return Constellation(self.config.constellation).generate_constellation_points()

    def build_bits_to_symbol_map(self, constellation_points: Union[np.ndarray, None] = None) -> Dict[BitTuple, float | int]:
        if constellation_points is None:
            constellation_points = self.get_constellation_points()

        M = constellation_points.size
        bits_per_symbol = int(math.log2(M))

        mapping: Dict[BitTuple, float | int] = {}

        for binary_index in range(M):
            gray_index = self._binary_to_gray(binary_index)
            gray_bits = self._int_to_bits(gray_index, bits_per_symbol)
            mapping[gray_bits] = constellation_points[binary_index].item()

        return mapping

    def build_symbol_to_bits_map(self, constellation_points: Union[np.ndarray, None] = None) -> Dict[float | int, BitTuple]:
        bits_to_symbol_map = self.build_bits_to_symbol_map(constellation_points)

        mapping: Dict[float | int, BitTuple] = {}
        for bits, symbol in bits_to_symbol_map.items():
            mapping[symbol] = bits

        return mapping

    def code_bits(self, constellation_points: Union[np.ndarray, None] = None) -> Dict[BitTuple, float | int]:
        return self.build_bits_to_symbol_map(constellation_points)

    def generate_bits_to_symbols(self, bits_array: np.ndarray, constellation_points: Union[np.ndarray, None] = None) -> np.ndarray:
        if constellation_points is None:
            constellation_points = self.get_constellation_points()

        if bits_array.ndim != 1:
            raise ValueError("bits_array must be a 1D numpy array")

        if not np.all((bits_array == 0) | (bits_array == 1)):
            raise ValueError("bits_array must contain only 0/1 values")

        M = constellation_points.size
        bits_per_symbol = int(math.log2(M))

        if bits_array.size % bits_per_symbol != 0:
            raise ValueError(f"bits_array length must be divisible by bits_per_symbol={bits_per_symbol}")

        bits_to_symbol_map = self.build_bits_to_symbol_map(constellation_points)
        bit_groups = bits_array.reshape(-1, bits_per_symbol)

        symbols = np.array([bits_to_symbol_map[tuple(int(b) for b in group)] for group in bit_groups],
                           dtype=constellation_points.dtype)
        return symbols

    def generate_symbols_to_bits(self, symbols_array: np.ndarray, constellation_points: Union[np.ndarray, None] = None) -> np.ndarray:
        if constellation_points is None:
            constellation_points = self.get_constellation_points()

        if symbols_array.ndim != 1:
            raise ValueError("symbols_array must be a 1D numpy array")

        symbol_to_bits_map = self.build_symbol_to_bits_map(constellation_points)

        detected_indices = np.argmin(np.abs(symbols_array[:, None] - constellation_points[None, :]), axis=1)

        detected_symbols = constellation_points[detected_indices]

        bit_list = []
        for symbol in detected_symbols:
            bits = symbol_to_bits_map[symbol.item()]
            bit_list.extend(bits)

        return np.array(bit_list, dtype=np.uint8)