from typing import Any

import numpy as np

from synthetic_dataset_creation.dataset_creation import Dataset, DatasetConfig


def build_pipeline_test_dataset(params: dict[str, Any]) -> dict[float, dict[int, dict[str, Any]]]:
    """
    Build a dataset for testing the receiver pipeline.
    The output format is: dataset[snr_db][sample_index] = sample
    where each sample is the dictionary returned by Dataset.generate_dataset(), with a few extra fields added.

    Expected params
    ---------------
    sample_rate:
        Sampling rate in Hz.

    symbol_time:
        Symbol time in seconds.

    total_signal_duration_sec:
        Total received block duration. Example: 0.600 for 600 ms.

    leading_noise_duration_sec:
        Noise-only duration before the first UW/signal. Example: 0.100.

    uw_spacing_sec:
        Time spacing between UW starts. Example: 0.100 for one UW every 100 ms.

    n_uw:
        Number of UWs in a signal sample. If the sample is noise-only, this is
        forced to 0.

    signal_probability:
        Probability that a generated sample contains signal. Otherwise it is
        noise-only.

    snr_db_values:
        Iterable of SNR/EbN0 values to simulate.

    n_samples_per_snr:
        Number of random samples generated per SNR.

    min_frequency_offset_hz, max_frequency_offset_hz:
        Random CFO range for signal samples.

    min_phase_offset_rad, max_phase_offset_rad:
        Random phase offset range for signal samples.

    Notes
    -----
    If a sample contains signal:
        received block = leading noise + signal
        first UW is at the start of the signal, after the leading noise
        later UWs are spaced by uw_spacing_sec

    If a sample is noise-only:
        received block = noise with the same total block length

    This function intentionally keeps all derived values local so the caller can
    pass a short params dictionary without manually calculating sample counts.
    """

    rng = np.random.default_rng(params.get("random_seed", None))

    sample_rate = float(params["sample_rate"])
    symbol_time = float(params["symbol_time"])
    sps = int(round(sample_rate * symbol_time))
    constellation_order = int(params["constellation_order"])
    bits_per_symbol = int(np.log2(constellation_order))
    total_signal_samples = int(round(float(params["total_signal_duration_sec"]) * sample_rate))
    leading_noise_samples = int(round(float(params["leading_noise_duration_sec"]) * sample_rate))
    uw_spacing_samples = int(round(float(params["uw_spacing_sec"]) * sample_rate))
    uw_spacing_symbols = int(round(uw_spacing_samples / sps))
    uw_spacing_bits = uw_spacing_symbols * bits_per_symbol

    if total_signal_samples <= leading_noise_samples:
        raise ValueError("total_signal_duration_sec must be larger than leading_noise_duration_sec")

    n_signal_symbols = int((total_signal_samples - leading_noise_samples) // sps)
    n_bits_to_transmit = n_signal_symbols * bits_per_symbol

    dataset: dict[float, dict[int, dict[str, Any]]] = {}

    for snr_db in params["snr_db_values"]:
        snr_db = float(snr_db)
        dataset[snr_db] = {}

        for sample_index in range(int(params["n_samples_per_snr"])):
            has_signal = bool(rng.random() < float(params["signal_probability"]))
            n_uw = int(params["n_uw"]) if has_signal else 0

            frequency_offset_hz = float(rng.uniform(float(params["min_frequency_offset_hz"]), float(params["max_frequency_offset_hz"])))
            phase_offset_rad = float(rng.uniform(float(params["min_phase_offset_rad"]), float(params["max_phase_offset_rad"])))

            dataset_config = DatasetConfig(
                sample_rate=sample_rate,
                symbol_time=symbol_time,
                n_bits_to_transmit=n_bits_to_transmit,
                const_frequency_offset=frequency_offset_hz,
                const_phase_offset=phase_offset_rad,
                h=float(params.get("h", 0.5)),
                pulse_shape_type=str(params["pulse_shape_type"]),
                span_in_symbols=int(params["span_in_symbols"]),
                pulse_normalization=str(params["pulse_normalization"]),
                pulse_causal=bool(params.get("pulse_causal", False)),
                constellation_type=str(params["constellation_type"]),
                constellation_order=constellation_order,
                noise_type=str(params["noise_type"]),
                noise_db_values=[snr_db],
                n_signals_per_snr=1,
                use_uw=True,
                uw_probability=1.0,
                n_uw_values=[n_uw],
                uw_bits=list(params["uw_bits"]),
                uw_spacing_bits=uw_spacing_bits,
                first_uw_after_noise=True,
                leading_noise_samples=leading_noise_samples,
                noise_only_when_no_uw=True,
                random_uw_start=False,
                uw_mode="overwrite",
                random_seed=int(rng.integers(0, 2**31 - 1)),
            )

            sample = Dataset(dataset_config).generate_dataset()[snr_db][0]

            # Force exactly the requested block length. Dataset adds leading
            # noise before the signal, and convolution can make the signal a few
            # samples longer than the nominal symbol count.
            rx_signal = np.asarray(sample["rx_signal"], dtype=np.complex128)
            if len(rx_signal) >= total_signal_samples:
                sample["rx_signal"] = rx_signal[:total_signal_samples]
            else:
                sample["rx_signal"] = np.pad(rx_signal, (0, total_signal_samples - len(rx_signal)))

            channel_noise = np.asarray(sample["channel_noise"], dtype=np.complex128)
            if len(channel_noise) >= total_signal_samples:
                sample["channel_noise"] = channel_noise[:total_signal_samples]
            else:
                sample["channel_noise"] = np.pad(channel_noise, (0, total_signal_samples - len(channel_noise)))

            sample["sps"] = int(sps)
            sample["bits_per_symbol"] = int(bits_per_symbol)
            sample["total_signal_samples"] = int(total_signal_samples)
            sample["leading_noise_samples"] = int(leading_noise_samples)
            sample["uw_spacing_samples"] = int(uw_spacing_samples)
            sample["uw_spacing_bits"] = int(uw_spacing_bits)
            sample["has_signal"] = bool(has_signal)

            dataset[snr_db][sample_index] = sample

    return dataset
