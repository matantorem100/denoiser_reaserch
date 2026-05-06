import numpy as np
import matplotlib.pyplot as plt

from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig
from synthetic_dataset_creation.modulator.modulator import ModulatorConfig, Modulator


# =============================================================================
# Basic helpers
# =============================================================================

def add_awgn_by_snr(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128)

    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )

    return signal + noise


def apply_frequency_offset(
    signal: np.ndarray,
    frequency_offset_hz: float,
    sample_rate: float,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128)
    n = np.arange(len(signal))

    return signal * np.exp(1j * 2.0 * np.pi * frequency_offset_hz * n / sample_rate)


def normalized_sliding_complex_correlation(
    signal: np.ndarray,
    reference: np.ndarray,
    use_abs: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Sliding normalized correlation:

        corr[k] = <reference, signal[k:k+L]> / (||reference|| ||segment||)

    If use_abs=True, returns abs(corr[k]).
    If use_abs=False, returns real(corr[k]).

    For detection, abs is usually better.
    """

    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)

    L = len(reference)
    n_lags = len(signal) - L + 1

    if n_lags <= 0:
        raise ValueError("Signal must be longer than or equal to reference.")

    reference_norm = np.linalg.norm(reference)
    out = np.zeros(n_lags, dtype=np.float64)

    for k in range(n_lags):
        segment = signal[k:k + L]
        denom = reference_norm * np.linalg.norm(segment)

        if denom < eps:
            out[k] = 0.0
            continue

        c = np.vdot(reference, segment) / denom

        if use_abs:
            out[k] = np.abs(c)
        else:
            out[k] = np.real(c)

    return out


def normalized_sliding_real_correlation(
    signal: np.ndarray,
    reference: np.ndarray,
    use_abs: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Same as normalized complex correlation, but for real-valued sequences.

    Useful for:

        diff(unwrap(angle(signal)))
    """

    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)

    L = len(reference)
    n_lags = len(signal) - L + 1

    if n_lags <= 0:
        raise ValueError("Signal must be longer than or equal to reference.")

    reference_norm = np.linalg.norm(reference)
    out = np.zeros(n_lags, dtype=np.float64)

    for k in range(n_lags):
        segment = signal[k:k + L]
        denom = reference_norm * np.linalg.norm(segment)

        if denom < eps:
            out[k] = 0.0
            continue

        c = np.dot(reference, segment) / denom

        if use_abs:
            out[k] = np.abs(c)
        else:
            out[k] = c

    return out


# =============================================================================
# Three UW correlation methods
# =============================================================================

def phase_diff_feature(x: np.ndarray) -> np.ndarray:
    """
    Feature used by method 1:

        y[n] = diff(unwrap(angle(x[n])))

    This is approximately instantaneous phase increment in rad/sample.
    """

    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    return np.diff(np.unwrap(np.angle(x)))


def differential_complex_feature(x: np.ndarray) -> np.ndarray:
    """
    Feature used by method 3:

        y[n] = x[n] * conj(x[n-1])

    This keeps the differential phase as a complex phasor.
    """

    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    return x[1:] * np.conj(x[:-1])


def compute_uw_correlation_methods(
    rx_signal: np.ndarray,
    uw_reference_signal: np.ndarray,
) -> dict:
    """
    Computes three normalized UW correlation curves.

    Method 1:
        correlation after diff(unwrap(angle(signal)))

    Method 2:
        direct complex correlation between rx signal and UW signal

    Method 3:
        differential complex correlation
    """

    rx_signal = np.asarray(rx_signal, dtype=np.complex128).reshape(-1)
    uw_reference_signal = np.asarray(uw_reference_signal, dtype=np.complex128).reshape(-1)

    # -------------------------------------------------------------------------
    # 1. Correlation after diff(unwrap(angle(signal)))
    # -------------------------------------------------------------------------
    rx_phase_diff = phase_diff_feature(rx_signal)
    uw_phase_diff = phase_diff_feature(uw_reference_signal)

    corr_phase_diff = normalized_sliding_real_correlation(
        signal=rx_phase_diff,
        reference=uw_phase_diff,
        use_abs=True,
    )

    # -------------------------------------------------------------------------
    # 2. Direct complex correlation
    # -------------------------------------------------------------------------
    corr_complex = normalized_sliding_complex_correlation(
        signal=rx_signal,
        reference=uw_reference_signal,
        use_abs=True,
    )

    # -------------------------------------------------------------------------
    # 3. Differential complex correlation
    # -------------------------------------------------------------------------
    rx_diff_complex = differential_complex_feature(rx_signal)
    uw_diff_complex = differential_complex_feature(uw_reference_signal)

    corr_diff_complex = normalized_sliding_complex_correlation(
        signal=rx_diff_complex,
        reference=uw_diff_complex,
        use_abs=True,
    )

    return {
        "phase_diff": corr_phase_diff,
        "complex": corr_complex,
        "diff_complex": corr_diff_complex,
    }


def summarize_peak(corr: np.ndarray) -> dict:
    peak_index = int(np.argmax(corr))
    peak_value = float(corr[peak_index])

    return {
        "peak_index": peak_index,
        "peak_value": peak_value,
    }


# =============================================================================
# Signal generation
# =============================================================================

def build_modulator():
    sample_rate = 10_000
    symbol_time = 1e-3

    pulse_shape_type = "rect"
    span_in_symbols = 1
    pulse_normalization = "cpfsk"

    constellation_type = "PAM"
    constellation_order = 4

    pulse_shape_config = PulseShapeConfig(
        pulse_shape_type=pulse_shape_type,
        normalization_type=pulse_normalization,
        span_in_symbols=span_in_symbols,
    )

    constellation_config = ConstellationConfig(
        constellation_type=constellation_type,
        constellation_order=constellation_order,
    )

    modulator = Modulator(
        ModulatorConfig(
            pulse_shape_config=pulse_shape_config,
            constellation_config=constellation_config,
        )
    )

    return modulator, sample_rate, symbol_time, constellation_order


def generate_test_signal(
    rng: np.random.Generator,
    modulator: Modulator,
    sample_rate: float,
    symbol_time: float,
    constellation_order: int,
    n_payload_bits: int,
    uw_n_bits: int,
    frequency_sensitivity: float,
    frequency_offset_hz: float,
    snr_db: float | None,
):
    """
    Generates:

        rx_signal
        clean_tx_signal
        uw_reference_signal
        tx_bits_with_uw

    Important:
    The UW is generated separately as a known bit pattern.
    The full signal is generated using the same UW at the beginning.
    """

    bits_per_symbol = int(np.log2(constellation_order))

    n_payload_bits = (n_payload_bits // bits_per_symbol) * bits_per_symbol
    uw_n_bits = (uw_n_bits // bits_per_symbol) * bits_per_symbol

    payload_bits = rng.integers(0, 2, size=n_payload_bits, dtype=np.uint8)

    # A fixed UW pattern with good transitions.
    # You may replace this with your real UW.
    uw_bits = rng.integers(0, 2, size=uw_n_bits, dtype=np.uint8)

    # Full signal: UW + payload.
    # I assume your modulator prepends uw to bits, based on your previous script.
    tx_signal, tx_bits_with_uw = modulator.modulate(
        bits=payload_bits,
        symbol_time=symbol_time,
        sample_rate=sample_rate,
        uw=uw_bits,
        frequency_offset=0.0,
        frequency_sensitivity=frequency_sensitivity,
    )

    # UW-only reference, generated without CFO.
    uw_reference_signal, _ = modulator.modulate(
        bits=np.array([], dtype=np.uint8),
        symbol_time=symbol_time,
        sample_rate=sample_rate,
        uw=uw_bits,
        frequency_offset=0.0,
        frequency_sensitivity=frequency_sensitivity,
    )

    # Apply CFO externally, so the same clean signal can be reused.
    rx_signal = apply_frequency_offset(
        signal=tx_signal,
        frequency_offset_hz=frequency_offset_hz,
        sample_rate=sample_rate,
    )

    if snr_db is not None:
        rx_signal = add_awgn_by_snr(
            signal=rx_signal,
            snr_db=snr_db,
            rng=rng,
        )

    return {
        "rx_signal": rx_signal,
        "clean_tx_signal": tx_signal,
        "uw_reference_signal": uw_reference_signal,
        "uw_bits": uw_bits,
        "payload_bits": payload_bits,
        "tx_bits_with_uw": tx_bits_with_uw,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_correlation_curves(
    correlations: dict,
    title: str,
    true_uw_start_sample: int = 0,
):
    plt.figure(figsize=(11, 5))

    plt.plot(
        correlations["phase_diff"],
        label="1. corr after diff(unwrap(angle))",
    )

    plt.plot(
        correlations["complex"],
        label="2. complex signal correlation",
    )

    plt.plot(
        correlations["diff_complex"],
        label="3. differential complex correlation",
    )

    plt.axvline(
        true_uw_start_sample,
        linestyle="--",
        label="true UW start",
    )

    plt.grid(True)
    plt.xlabel("Lag [samples]")
    plt.ylabel("Normalized correlation")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_zoomed_correlation_curves(
    correlations: dict,
    title: str,
    max_lag_to_show: int,
    true_uw_start_sample: int = 0,
):
    plt.figure(figsize=(11, 5))

    for key, label in [
        ("phase_diff", "1. corr after diff(unwrap(angle))"),
        ("complex", "2. complex signal correlation"),
        ("diff_complex", "3. differential complex correlation"),
    ]:
        y = correlations[key]
        stop = min(max_lag_to_show, len(y))
        plt.plot(np.arange(stop), y[:stop], label=label)

    plt.axvline(
        true_uw_start_sample,
        linestyle="--",
        label="true UW start",
    )

    plt.grid(True)
    plt.xlabel("Lag [samples]")
    plt.ylabel("Normalized correlation")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_peak_summary(
    summary_rows: list[dict],
    title: str,
):
    labels = [row["method"] for row in summary_rows]
    values = [row["peak_value"] for row in summary_rows]

    x = np.arange(len(labels))

    plt.figure(figsize=(9, 5))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim([0.0, 1.05])
    plt.grid(True, axis="y")
    plt.ylabel("Peak normalized correlation")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def print_summary(
    scenario_name: str,
    correlations: dict,
):
    print()
    print("=" * 90)
    print(scenario_name)
    print("=" * 90)

    for key, label in [
        ("phase_diff", "1. corr after diff(unwrap(angle))"),
        ("complex", "2. complex signal correlation"),
        ("diff_complex", "3. differential complex correlation"),
    ]:
        peak = summarize_peak(correlations[key])
        print(
            f"{label:<45} | "
            f"peak lag = {peak['peak_index']:>6d} samples | "
            f"peak value = {peak['peak_value']:.6f}"
        )


# =============================================================================
# Main experiment
# =============================================================================

def run_scenario(
    scenario_name: str,
    frequency_offset_hz: float,
    snr_db: float | None,
    rng_seed: int = 0,
):
    rng = np.random.default_rng(rng_seed)

    modulator, sample_rate, symbol_time, constellation_order = build_modulator()

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    n_payload_bits = 4000
    uw_n_bits = 20

    frequency_sensitivity = 0.5

    sps = int(round(sample_rate * symbol_time))

    data = generate_test_signal(
        rng=rng,
        modulator=modulator,
        sample_rate=sample_rate,
        symbol_time=symbol_time,
        constellation_order=constellation_order,
        n_payload_bits=n_payload_bits,
        uw_n_bits=uw_n_bits,
        frequency_sensitivity=frequency_sensitivity,
        frequency_offset_hz=frequency_offset_hz,
        snr_db=snr_db,
    )

    rx_signal = data["rx_signal"]
    uw_reference_signal = data["uw_reference_signal"]

    correlations = compute_uw_correlation_methods(
        rx_signal=rx_signal,
        uw_reference_signal=uw_reference_signal,
    )

    print_summary(
        scenario_name=scenario_name,
        correlations=correlations,
    )

    title_suffix = (
        f"CFO = {frequency_offset_hz} Hz, "
        f"SNR = {'clean' if snr_db is None else str(snr_db) + ' dB'}"
    )

    plot_correlation_curves(
        correlations=correlations,
        title=f"UW normalized correlation comparison\n{title_suffix}",
        true_uw_start_sample=0,
    )

    plot_zoomed_correlation_curves(
        correlations=correlations,
        title=f"UW normalized correlation comparison, zoom near UW\n{title_suffix}",
        max_lag_to_show=25 * sps,
        true_uw_start_sample=0,
    )

    summary_rows = []

    for key, method_name in [
        ("phase_diff", "phase diff"),
        ("complex", "complex"),
        ("diff_complex", "diff complex"),
    ]:
        peak = summarize_peak(correlations[key])
        summary_rows.append(
            {
                "method": method_name,
                "peak_index": peak["peak_index"],
                "peak_value": peak["peak_value"],
            }
        )

    plot_peak_summary(
        summary_rows=summary_rows,
        title=f"Peak UW correlation value\n{title_suffix}",
    )

    return {
        "scenario_name": scenario_name,
        "frequency_offset_hz": frequency_offset_hz,
        "snr_db": snr_db,
        "correlations": correlations,
        "data": data,
        "sample_rate": sample_rate,
        "symbol_time": symbol_time,
    }


def compare_no_cfo_vs_cfo():
    # Clean or noisy?
    # Use None for no noise.
    # Use e.g. 10, 5, 0 dB to test realistic behavior.
    snr_db = 0

    result_no_cfo = run_scenario(
        scenario_name="Scenario 1: without frequency offset",
        frequency_offset_hz=0.0,
        snr_db=snr_db,
        rng_seed=0,
    )

    result_with_cfo = run_scenario(
        scenario_name="Scenario 2: with frequency offset",
        frequency_offset_hz=200.0,
        snr_db=snr_db,
        rng_seed=0,
    )

    # -------------------------------------------------------------------------
    # Side-by-side comparison per method
    # -------------------------------------------------------------------------
    methods = [
        ("phase_diff", "1. corr after diff(unwrap(angle))"),
        ("complex", "2. complex signal correlation"),
        ("diff_complex", "3. differential complex correlation"),
    ]

    for key, label in methods:
        corr_no_cfo = result_no_cfo["correlations"][key]
        corr_with_cfo = result_with_cfo["correlations"][key]

        plt.figure(figsize=(11, 5))
        plt.plot(corr_no_cfo, label="without CFO")
        plt.plot(corr_with_cfo, label="with CFO")
        plt.axvline(0, linestyle="--", label="true UW start")
        plt.grid(True)
        plt.xlabel("Lag [samples]")
        plt.ylabel("Normalized correlation")
        plt.title(f"{label}\nNo CFO vs CFO")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # Compact final summary table
    # -------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("Final comparison summary")
    print("=" * 100)
    print(
        f"{'Method':<45} | "
        f"{'Peak no CFO':>12} | "
        f"{'Lag no CFO':>10} | "
        f"{'Peak CFO':>12} | "
        f"{'Lag CFO':>10}"
    )
    print("-" * 100)

    for key, label in methods:
        peak_no_cfo = summarize_peak(result_no_cfo["correlations"][key])
        peak_with_cfo = summarize_peak(result_with_cfo["correlations"][key])

        print(
            f"{label:<45} | "
            f"{peak_no_cfo['peak_value']:>12.6f} | "
            f"{peak_no_cfo['peak_index']:>10d} | "
            f"{peak_with_cfo['peak_value']:>12.6f} | "
            f"{peak_with_cfo['peak_index']:>10d}"
        )


if __name__ == "__main__":
    compare_no_cfo_vs_cfo()