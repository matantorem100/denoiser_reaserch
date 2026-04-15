import numpy as np
import matplotlib.pyplot as plt

from synthetic_dataset_creation.channel.channel import ChannelConfig, Channel
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig, Constellation
from demodulator_research.demodulator.demodulator import DemodulatorConfig, Demodulator
from synthetic_dataset_creation.modulator.modulator import ModulatorConfig, Modulator, FMConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig, PulseShape

# params:
sample_rate = 10000
symbol_time = 1e-3
sps = int(symbol_time * sample_rate)
n_bits_to_transmit = 100000
uw = None
apply_fm = True
frequency_offset = 0.0
h = 0.5

pulse_shape_type = "rect"
pulse_normalization = "cpfsk"

if apply_fm and pulse_normalization != "cpfsk":
    raise ValueError("the pulse shape should be cpfsk normalized in fm modulation")

if not apply_fm and pulse_normalization != "norm_2":
    raise ValueError("the pulse shape should be norm 2 normalized in linear modulation")

constellation_type = "PAM"
constellation_order = 4

pulse_shape_config = PulseShapeConfig(pulse_shape_type=pulse_shape_type, normalization_type=pulse_normalization)
pulse_shape_instance = PulseShape(pulse_shape_config)

constellation_config = ConstellationConfig(constellation_type=constellation_type, constellation_order=constellation_order)
constellation_instance = Constellation(constellation_config)

fm_config = FMConfig(frequency_offset=frequency_offset, frequency_sensitivity=h) if apply_fm else None

modulator_config = ModulatorConfig(pulse_shape_config=pulse_shape_config, constellation_config=constellation_config,
                           fm_config=fm_config)
modulator_instance = Modulator(modulator_config)

demodulator_config = DemodulatorConfig(pulse_shape_config=pulse_shape_config, constellation_config=constellation_config,
                                       fm_config=fm_config)
demodulator_instance = Demodulator(demodulator_config)

# modulate in different eb_n0_db
ber_list = []
noise_sizes = np.arange(0, 20, 2)
for noise_size in noise_sizes:
    bits = np.random.randint(0, 2, size=n_bits_to_transmit)

    modulated_signal_without_noise, tx_bits = modulator_instance.modulate(bits, symbol_time, sample_rate, apply_fm, uw)

    channel_config_in_current_eb_n0_db = ChannelConfig(channel_type="awgn",
                                                       bits_per_symbol=int(np.log2(constellation_order)),
                                                       sps=sps, noise_type="snr", noise_db=float(noise_size))
    channel_instance_in_current_eb_n0_db = Channel(channel_config_in_current_eb_n0_db)

    modulated_signal_with_noise = channel_instance_in_current_eb_n0_db.transmit(modulated_signal_without_noise)

    rx_bits = demodulator_instance.differentiate_demodulate(rx_signal=modulated_signal_with_noise, symbol_time=symbol_time,
                                              sample_rate=sample_rate, apply_fm=apply_fm)

    min_len = min(len(tx_bits), len(rx_bits))

    bit_errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
    ber = bit_errors / min_len
    ber_list.append(ber)

ber_array = np.array(ber_list)
ber_array = np.maximum(ber_array, 1e-5)

plt.figure(figsize=(8, 5))

plt.semilogy(noise_sizes, ber_array, marker="o", label=f"{constellation_order}-PAM CPFSK - {apply_fm}")

plt.grid(True, which="both")
plt.xlabel("snr [dB]")
plt.ylabel("BER")
plt.legend()
plt.tight_layout()
plt.show()


