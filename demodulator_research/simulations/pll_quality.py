import numpy as np
import matplotlib.pyplot as plt

from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig

from demodulator_research.demodulator.demodulator import DemodulatorConfig, Demodulator
from synthetic_dataset_creation.modulator.modulator import ModulatorConfig, Modulator


def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return 1.0

    tx_bits = np.asarray(tx_bits[:n], dtype=np.uint8)
    rx_bits = np.asarray(rx_bits[:n], dtype=np.uint8)

    return float(np.mean(tx_bits != rx_bits))


def add_awgn_by_snr(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal = np.asarray(signal)

    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )

    return signal + noise


def normalize_to_unit_magnitude(
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize complex samples to unit magnitude while avoiding division by zero.

    This keeps only the phase:

        x_norm[n] = x[n] / |x[n]|

    For very small samples, we divide by eps instead of zero.
    """

    x = np.asarray(x, dtype=np.complex128)
    mag = np.abs(x)

    return x / np.maximum(mag, eps)


def estimate_cfo_from_diff_phase_mean(
    rx_signal: np.ndarray,
    sample_rate: float,
) -> float:
    """
    Coarse CFO estimator from the mean differentiated phase.

    For:

        rx[n] = s[n] * exp(j * 2*pi*df*n/Fs)

    we have:

        angle(rx[n] * conj(rx[n-1])) ~= 2*pi*df/Fs + modulation contribution

    If the modulation has approximately zero-mean instantaneous frequency,
    the mean phase increment gives a coarse CFO estimate.
    """

    rx_signal = np.asarray(rx_signal, dtype=np.complex128).reshape(-1)

    if len(rx_signal) < 2:
        return 0.0

    coarse_cfo_hz = np.mean(np.diff(np.unwrap(np.angle(rx_signal)))) * sample_rate / (2.0 * np.pi)

    return float(coarse_cfo_hz)


def estimate_frequency_offset_diff_blind_circular_mean_hz(
    rx_signal: np.ndarray,
    sample_rate: float,
    normalize_amplitude: bool = True,
) -> float:
    """
    Blind differential frequency offset estimator using circular mean.

    Computes:

        d[n] = rx[n] * conj(rx[n-1])

    Then estimates phase increments using:

        phase_increment[n] = angle(d[n])

    Then averages the phase increments directly:

        mean_phase_increment = mean(angle(d[n]))

    Finally converts to Hz:

        f_hat = mean_phase_increment * Fs / (2*pi)

    For CPFSK, this estimator is intuitive because angle(d[n])
    is basically the instantaneous frequency in radians/sample.
    """

    rx = np.asarray(rx_signal, dtype=np.complex128).reshape(-1)

    if len(rx) < 2:
        return 0.0

    if normalize_amplitude:
        rx = normalize_to_unit_magnitude(rx)

    differential_rx = rx[1:] * np.conj(rx[:-1])

    mean_differential_phasor = np.mean(differential_rx)

    phase_increment_rad = np.angle(mean_differential_phasor)

    estimated_offset_hz = phase_increment_rad * sample_rate / (2.0 * np.pi)

    return float(estimated_offset_hz)


def estimate_coarse_cfo_hz(
    rx_signal: np.ndarray,
    sample_rate: float,
    coarse_estimator_type: str,
) -> float:
    """
    Selectable coarse CFO estimator.

    Options:
        "none"
        "diff_phase_mean"
        "diff_blind_circular_mean"
    """

    if coarse_estimator_type == "none":
        return 0.0

    if coarse_estimator_type == "diff_phase_mean":
        return estimate_cfo_from_diff_phase_mean(
            rx_signal=rx_signal,
            sample_rate=sample_rate,
        )

    if coarse_estimator_type == "diff_blind_circular_mean":
        return estimate_frequency_offset_diff_blind_circular_mean_hz(
            rx_signal=rx_signal,
            sample_rate=sample_rate,
            normalize_amplitude=True,
        )

    raise ValueError(
        f"Unknown coarse_estimator_type={coarse_estimator_type!r}. "
        f"Valid options are: 'none', 'diff_phase_mean', 'diff_blind_circular_mean'."
    )


def correct_frequency_offset(
    rx_signal: np.ndarray,
    estimated_cfo_hz: float,
    sample_rate: float,
) -> np.ndarray:
    rx_signal = np.asarray(rx_signal)

    n = np.arange(len(rx_signal))
    correction = np.exp(-1j * 2.0 * np.pi * estimated_cfo_hz * n / sample_rate)

    return rx_signal * correction


def semi_coherent_demodulate_with_trace(
    demodulator: Demodulator,
    rx_signal: np.ndarray,
    symbol_time: float,
    sample_rate: float,
    frequency_sensitivity: float = 0.5,
    alpha: float = 0.15,
    beta: float = 0.01,
    use_abs_metric: bool = False,
):
    """
    Semi-coherent CPFSK/FM detector with fine residual CFO tracking.

    This assumes coarse CFO correction was already applied before calling it.
    Therefore:

        freq_state_hz = residual CFO estimate
    """

    rx_signal = np.asarray(rx_signal)
    sps = int(round(symbol_time * sample_rate))

    constellation_points = demodulator._constellation_instance.generate_constellation_points()
    pulse_shape = demodulator._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)

    h = frequency_sensitivity
    dt = 1.0 / sample_rate

    one_symbol_pulse = pulse_shape[:sps]
    n_symbols = len(rx_signal) // sps

    decided_symbols = []

    phase_state = 0.0

    # This is residual CFO after coarse correction.
    freq_state_hz = 0.0

    freq_state_history_hz = []
    phase_error_history_rad = []
    phase_state_history_rad = []
    metric_history = []
    decided_symbol_history = []

    for k in range(n_symbols):
        start = k * sps
        stop = start + sps

        r_seg = rx_signal[start:stop]
        if len(r_seg) != sps:
            break

        r_norm = np.linalg.norm(r_seg)

        best_metric = -np.inf
        best_symbol = constellation_points[0]
        best_ref = None
        best_symbol_phase_increment = 0.0

        for a in constellation_points:
            m = a * one_symbol_pulse

            symbol_phase = 2.0 * np.pi * h * np.cumsum(m) * dt

            n = np.arange(sps)
            residual_freq_phase = 2.0 * np.pi * freq_state_hz * n / sample_rate

            phi = phase_state + symbol_phase + residual_freq_phase
            s_ref = np.exp(1j * phi)

            denom = r_norm * np.linalg.norm(s_ref)

            if denom < 1e-12:
                metric = -np.inf
            else:
                corr = np.vdot(s_ref, r_seg)

                if use_abs_metric:
                    metric = np.abs(corr) / denom
                else:
                    metric = np.real(corr) / denom

            if metric > best_metric:
                best_metric = metric
                best_symbol = a
                best_ref = s_ref
                best_symbol_phase_increment = 2.0 * np.pi * h * np.sum(m) * dt

        decided_symbols.append(best_symbol)

        corr = np.vdot(best_ref, r_seg)
        phase_error = np.angle(corr)

        # Fine residual frequency tracking.
        freq_state_hz += beta * phase_error / (2.0 * np.pi * symbol_time)

        phase_state += best_symbol_phase_increment
        phase_state += 2.0 * np.pi * freq_state_hz * symbol_time
        phase_state += alpha * phase_error

        phase_state = np.angle(np.exp(1j * phase_state))

        freq_state_history_hz.append(freq_state_hz)
        phase_error_history_rad.append(phase_error)
        phase_state_history_rad.append(phase_state)
        metric_history.append(best_metric)
        decided_symbol_history.append(best_symbol)

    decided_symbols = np.asarray(decided_symbols, dtype=constellation_points.dtype)

    rx_bits = demodulator._coding_instance.generate_symbols_to_bits(
        decided_symbols,
        constellation_points,
    )

    trace = {
        "residual_freq_state_hz": np.asarray(freq_state_history_hz),
        "phase_error_rad": np.asarray(phase_error_history_rad),
        "phase_state_rad": np.asarray(phase_state_history_rad),
        "metric": np.asarray(metric_history),
        "decided_symbols": np.asarray(decided_symbol_history),
    }

    return rx_bits, trace


def run_single_pll_experiment(
    coarse_estimator_type: str,
    rng_seed: int = 0,
):
    rng = np.random.default_rng(rng_seed)

    # -------------------------
    # Simulation parameters
    # -------------------------
    sample_rate = 10_000
    symbol_time = 1e-3
    n_bits = 4000

    true_frequency_offset_hz = 500
    snr_db = 2

    frequency_sensitivity = 0.5

    alpha = 0.15
    beta = 0.01

    pulse_shape_type = "rect"
    span_in_symbols = 1
    pulse_normalization = "cpfsk"

    constellation_type = "PAM"
    constellation_order = 4

    # -------------------------
    # Build objects
    # -------------------------
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

    demodulator = Demodulator(
        DemodulatorConfig(
            pulse_shape_config=pulse_shape_config,
            constellation_config=constellation_config,
        )
    )

    # -------------------------
    # Generate and modulate
    # -------------------------
    bits_per_symbol = int(np.log2(constellation_order))
    n_bits = (n_bits // bits_per_symbol) * bits_per_symbol

    tx_bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)

    tx_signal, tx_bits_with_uw = modulator.modulate(
        bits=tx_bits,
        symbol_time=symbol_time,
        sample_rate=sample_rate,
        uw=None,
        frequency_offset=true_frequency_offset_hz,
        frequency_sensitivity=frequency_sensitivity,
    )

    rx_signal = add_awgn_by_snr(tx_signal, snr_db=snr_db, rng=rng)

    # -------------------------
    # 1. Selectable coarse CFO estimation
    # -------------------------
    coarse_cfo_hz = estimate_coarse_cfo_hz(
        rx_signal=rx_signal,
        sample_rate=sample_rate,
        coarse_estimator_type=coarse_estimator_type,
    )

    rx_signal_coarse_corrected = correct_frequency_offset(
        rx_signal=rx_signal,
        estimated_cfo_hz=coarse_cfo_hz,
        sample_rate=sample_rate,
    )

    initial_residual_cfo_hz = true_frequency_offset_hz - coarse_cfo_hz

    print("=" * 80)
    print(f"Coarse estimator type:       {coarse_estimator_type}")
    print(f"True CFO:                    {true_frequency_offset_hz:.6f} Hz")
    print(f"Coarse CFO estimate:          {coarse_cfo_hz:.6f} Hz")
    print(f"Initial residual CFO:         {initial_residual_cfo_hz:.6f} Hz")

    # -------------------------
    # 2. Fine PLL tracking after coarse correction
    # -------------------------
    rx_bits, trace = semi_coherent_demodulate_with_trace(
        demodulator=demodulator,
        rx_signal=rx_signal_coarse_corrected,
        symbol_time=symbol_time,
        sample_rate=sample_rate,
        frequency_sensitivity=frequency_sensitivity,
        alpha=alpha,
        beta=beta,
        use_abs_metric=False,
    )

    ber = calculate_ber(tx_bits_with_uw, rx_bits)

    if len(trace["residual_freq_state_hz"]) == 0:
        raise RuntimeError("PLL trace is empty. Check signal length and sps.")

    residual_pll_final_hz = trace["residual_freq_state_hz"][-1]
    total_final_cfo_estimate_hz = coarse_cfo_hz + residual_pll_final_hz

    print(f"Final PLL residual estimate: {residual_pll_final_hz:.6f} Hz")
    print(f"Total final CFO estimate:    {total_final_cfo_estimate_hz:.6f} Hz")
    print(f"Final CFO estimation error:  {true_frequency_offset_hz - total_final_cfo_estimate_hz:.6f} Hz")
    print(f"BER:                         {ber:.6e}")

    result = {
        "coarse_estimator_type": coarse_estimator_type,
        "true_frequency_offset_hz": true_frequency_offset_hz,
        "coarse_cfo_hz": coarse_cfo_hz,
        "initial_residual_cfo_hz": initial_residual_cfo_hz,
        "residual_pll_final_hz": residual_pll_final_hz,
        "total_final_cfo_estimate_hz": total_final_cfo_estimate_hz,
        "final_cfo_error_hz": true_frequency_offset_hz - total_final_cfo_estimate_hz,
        "ber": ber,
        "trace": trace,
        "symbol_time": symbol_time,
        "sample_rate": sample_rate,
        "snr_db": snr_db,
        "alpha": alpha,
        "beta": beta,
    }

    return result


def plot_single_result(result: dict):
    trace = result["trace"]

    symbol_time = result["symbol_time"]
    true_frequency_offset_hz = result["true_frequency_offset_hz"]
    coarse_cfo_hz = result["coarse_cfo_hz"]
    initial_residual_cfo_hz = result["initial_residual_cfo_hz"]
    snr_db = result["snr_db"]
    alpha = result["alpha"]
    beta = result["beta"]
    coarse_estimator_type = result["coarse_estimator_type"]

    symbol_index = np.arange(len(trace["residual_freq_state_hz"]))
    time_sec = symbol_index * symbol_time

    total_cfo_estimate_hz = coarse_cfo_hz + trace["residual_freq_state_hz"]

    plt.figure(figsize=(9, 5))
    plt.plot(symbol_index, total_cfo_estimate_hz, label="Coarse + PLL CFO estimate")
    plt.axhline(true_frequency_offset_hz, linestyle="--", label="True CFO")
    plt.axhline(coarse_cfo_hz, linestyle=":", label="Coarse CFO estimate")
    plt.grid(True)
    plt.xlabel("Symbol index")
    plt.ylabel("CFO estimate [Hz]")
    plt.title(
        f"CFO Lock: {coarse_estimator_type}\n"
        f"True CFO={true_frequency_offset_hz} Hz, SNR={snr_db} dB, alpha={alpha}, beta={beta}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(symbol_index, trace["residual_freq_state_hz"], label="PLL residual CFO estimate")
    plt.axhline(initial_residual_cfo_hz, linestyle="--", label="Initial residual CFO")
    plt.grid(True)
    plt.xlabel("Symbol index")
    plt.ylabel("Residual CFO estimate [Hz]")
    plt.title(f"PLL Residual CFO Lock after Coarse Correction: {coarse_estimator_type}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(time_sec, total_cfo_estimate_hz, label="Coarse + PLL CFO estimate")
    plt.axhline(true_frequency_offset_hz, linestyle="--", label="True CFO")
    plt.grid(True)
    plt.xlabel("Time [sec]")
    plt.ylabel("CFO estimate [Hz]")
    plt.title(f"CFO Lock versus Time: {coarse_estimator_type}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(symbol_index, trace["phase_error_rad"])
    plt.grid(True)
    plt.xlabel("Symbol index")
    plt.ylabel("Phase error [rad]")
    plt.title(f"PLL Phase Error: {coarse_estimator_type}")
    plt.tight_layout()
    plt.show()


def compare_coarse_estimators():
    coarse_estimator_types = [
        "none",
        "diff_phase_mean",
        "diff_blind_circular_mean",
    ]

    results = []

    for coarse_estimator_type in coarse_estimator_types:
        result = run_single_pll_experiment(
            coarse_estimator_type=coarse_estimator_type,
            rng_seed=0,
        )
        results.append(result)

    # -------------------------
    # Comparison plot
    # -------------------------
    plt.figure(figsize=(9, 5))

    for result in results:
        trace = result["trace"]
        symbol_index = np.arange(len(trace["residual_freq_state_hz"]))

        total_cfo_estimate_hz = (
            result["coarse_cfo_hz"] + trace["residual_freq_state_hz"]
        )

        plt.plot(
            symbol_index,
            total_cfo_estimate_hz,
            label=result["coarse_estimator_type"],
        )

    true_cfo = results[0]["true_frequency_offset_hz"]

    plt.axhline(true_cfo, linestyle="--", label="True CFO")
    plt.grid(True)
    plt.xlabel("Symbol index")
    plt.ylabel("Total CFO estimate [Hz]")
    plt.title("PLL CFO Lock Comparison for Different Coarse Estimators")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Print compact summary
    # -------------------------
    print("\nSummary:")
    print("-" * 100)
    print(
        f"{'Estimator':<28} | "
        f"{'Coarse CFO [Hz]':>16} | "
        f"{'Final CFO [Hz]':>16} | "
        f"{'Error [Hz]':>12} | "
        f"{'BER':>12}"
    )
    print("-" * 100)

    for result in results:
        print(
            f"{result['coarse_estimator_type']:<28} | "
            f"{result['coarse_cfo_hz']:>16.6f} | "
            f"{result['total_final_cfo_estimate_hz']:>16.6f} | "
            f"{result['final_cfo_error_hz']:>12.6f} | "
            f"{result['ber']:>12.6e}"
        )


def main():
    # Option 1:
    # Run only one estimator and plot detailed traces.
    coarse_estimator_type = "diff_blind_circular_mean"
    result = run_single_pll_experiment(
        coarse_estimator_type=coarse_estimator_type,
        rng_seed=0,
    )
    plot_single_result(result)

    # Option 2:
    # Compare all three options in one run.
    compare_coarse_estimators()


if __name__ == "__main__":
    main()