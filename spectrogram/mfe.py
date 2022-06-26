
import jax.numpy as jnp

# this package ported from speechpy mfe

# @article{DBLP:journals/corr/abs-1803-01094,
#   author    = {Amirsina Torfi},
#   title     = {SpeechPy - {A} Library for Speech Processing and Recognition},
#   journal   = {CoRR},
#   volume    = {abs/1803.01094},
#   year      = {2018},
#   url       = {http://arxiv.org/abs/1803.01094},
#   eprinttype = {arXiv},
#   eprint    = {1803.01094},
#   timestamp = {Tue, 17 Sep 2019 14:15:09 +0200},
#   biburl    = {https://dblp.org/rec/journals/corr/abs-1803-01094.bib},
#   bibsource = {dblp computer science bibliography, https://dblp.org}
# }

from . import feature
from . import processing

def generate_features(signal, sampling_freq,
                        frame_length=0.02, frame_stride=0.01, num_filters=40, fft_length=256,
                        low_frequency=300, high_frequency=0, noise_floor_db=-52):

    if (num_filters < 2):
        raise Exception('num_filters should be at least 2')

    low_frequency = None if low_frequency == 0 else low_frequency
    high_frequency = None if high_frequency == 0 else high_frequency

    # Rescale to [-1, 1] and add preemphasis
    signal = processing.preemphasis(signal, cof=0.98, shift=1)

    # we calculate the filter banks in numpy since it involves some difficult
    # to port to jax pieces and isn't dependant on the signal
    filter_banks = feature.filterbanks(num_filters, coefficients=129,
        sampling_freq=sampling_freq, low_freq=low_frequency,
        high_freq=high_frequency)

    # calc mfe
    mfe = feature.mfe(signal, filter_banks, sampling_frequency=sampling_freq,
                      frame_length=frame_length, frame_stride=frame_stride,
                      fft_length=fft_length)

    # Clip to avoid zero values
    mfe = jnp.clip(mfe, 1e-30, None)
    # Convert to dB scale
    # log_mel_spec = 10 * log10(mel_spectrograms)
    mfe = 10 * jnp.log10(mfe)

    # Add power offset and clip values below 0 (hard filter)
    # log_mel_spec = (log_mel_spec + self._power_offset - 32 + 32.0) / 64.0
    # log_mel_spec = tf.clip_by_value(log_mel_spec, 0, 1)
    mfe = (mfe - noise_floor_db) / ((-1 * noise_floor_db) + 12)
    mfe = jnp.clip(mfe, 0, 1)

    # Quantize to 8 bits and dequantize back to float32
    mfe = jnp.uint8(jnp.around(mfe * 2**8))
    # clip to 2**8
    mfe = jnp.clip(mfe, 0, 255)
    mfe = jnp.float32(mfe / 2**8)

    return mfe.reshape(-1, num_filters)
