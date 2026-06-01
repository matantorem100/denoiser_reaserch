from dataclasses import dataclass

import numpy as np
import scipy.signal as sp


@dataclass
class GridSearchResult:
    uw_positions: list[int]
    peak_positions: list[int]
    peak_values: list[float]
    score: float


def find_threshold_peaks(correlation: np.ndarray, threshold: float = 0.45, min_peak_distance: int = 1) -> tuple[np.ndarray, np.ndarray]:
    correlation = np.asarray(correlation, dtype=float).reshape(-1)

    peaks, properties = sp.find_peaks(correlation, height=threshold, distance=max(1, int(min_peak_distance)))

    peak_values = np.asarray(properties["peak_heights"], dtype=float)

    return peaks.astype(int), peak_values


def _nearest_unused_peak(peaks: np.ndarray, peak_values: np.ndarray, expected_position: int, spacing_tolerance: int,
                         used_peak_indices: set[int]) -> tuple[int, int, float] | None:
    candidates = []
    for peak_index, peak_position in enumerate(peaks):
        if peak_index in used_peak_indices:
            continue

        error = abs(int(peak_position) - int(expected_position))
        if error <= spacing_tolerance:
            candidates.append((error, -float(peak_values[peak_index]), peak_index))

    if not candidates:
        return None

    _, _, best_peak_index = min(candidates)
    return (
        best_peak_index,
        int(peaks[best_peak_index]),
        float(peak_values[best_peak_index]),
    )


def find_best_uw_grid(correlation: np.ndarray, uw_spacing: int, threshold: float = 0.45, n_uw: int | None = None,
                      spacing_tolerance: int = 2, min_peak_distance: int | None = None, min_n_uw: int = 1,
                      first_peak_weight: float = 1.0) -> GridSearchResult:
    """
    Find UW positions from a correlation vector.

    Returns positions in correlation-index units. If the correlation was produced
    by full convolution, convert these indices to true signal starts outside this
    function using the detector-specific offset.

    first_peak_weight:
        Extra score multiplier for the first peak in each candidate grid.
        Use 1.0 for normal sum scoring. Use values > 1.0 when the first UW is
        expected to be cleaner/more reliable than later UWs.
    """
    correlation = np.asarray(correlation, dtype=float).reshape(-1)

    if min_peak_distance is None:
        min_peak_distance = max(1, uw_spacing // 3)

    peaks, peak_values = find_threshold_peaks(correlation=correlation, threshold=threshold, min_peak_distance=min_peak_distance)

    if len(peaks) == 0:
        return GridSearchResult([], [], [], 0.0)

    if n_uw == 1:
        best_peak_index = int(np.argmax(peak_values))
        return GridSearchResult(
            uw_positions=[int(peaks[best_peak_index])],
            peak_positions=[int(peaks[best_peak_index])],
            peak_values=[float(peak_values[best_peak_index])],
            score=float(max(first_peak_weight, 1.0) * peak_values[best_peak_index]),
        )

    best_positions: list[int] = []
    best_values: list[float] = []
    best_score = -np.inf
    max_grid_len = n_uw if n_uw is not None else max(1, len(correlation) // uw_spacing + 1)

    for first_peak_index, first_peak_position in enumerate(peaks):
        used_peak_indices = {first_peak_index}
        grid_positions = [int(first_peak_position)]
        grid_values = [float(peak_values[first_peak_index])]

        for grid_index in range(1, max_grid_len):
            expected_position = int(first_peak_position) + grid_index * uw_spacing
            if expected_position >= len(correlation) + spacing_tolerance:
                break

            matched_peak = _nearest_unused_peak(
                peaks=peaks,
                peak_values=peak_values,
                expected_position=expected_position,
                spacing_tolerance=spacing_tolerance,
                used_peak_indices=used_peak_indices,
            )
            if matched_peak is None:
                if n_uw is not None:
                    grid_positions = []
                    grid_values = []
                break

            peak_index, peak_position, peak_value = matched_peak
            used_peak_indices.add(peak_index)
            grid_positions.append(peak_position)
            grid_values.append(peak_value)

        if len(grid_positions) < min_n_uw:
            continue
        if n_uw is not None and len(grid_positions) != n_uw:
            continue

        score = float(np.sum(grid_values))
        if grid_values:
            score += float(max(first_peak_weight, 1.0) - 1.0) * float(grid_values[0])
        if len(grid_positions) > 1:
            score += 0.1 * len(grid_positions)

        if score > best_score:
            best_score = score
            best_positions = grid_positions
            best_values = grid_values

    if not best_positions:
        return GridSearchResult([], [], [], 0.0)

    return GridSearchResult(
        uw_positions=best_positions,
        peak_positions=best_positions.copy(),
        peak_values=best_values,
        score=float(best_score),
    )


def find_uw_positions(correlation: np.ndarray, uw_spacing: int, threshold: float = 0.45, n_uw: int | None = None,
                      spacing_tolerance: int = 2, first_peak_weight: float = 1.0) -> list[int]:

    result = find_best_uw_grid(
        correlation=correlation,
        uw_spacing=uw_spacing,
        threshold=threshold,
        n_uw=n_uw,
        spacing_tolerance=spacing_tolerance,
        first_peak_weight=first_peak_weight,
    )
    return result.uw_positions
