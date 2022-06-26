import numpy as np
import jax.numpy as jnp

from . import processing
from . import functions

def filterbanks(
        num_filter,
        coefficients,
        sampling_freq,
        low_freq,
        high_freq):

    high_freq = high_freq or sampling_freq / 2
    low_freq = low_freq or 300

    # Computing the Mel filterbank
    # converting the upper and lower frequencies to Mels.
    # num_filter + 2 is because for num_filter filterbanks we need
    # num_filter+2 point.
    mels = np.linspace(
        functions.frequency_to_mel(low_freq),
        functions.frequency_to_mel(high_freq),
        num_filter + 2)

    # we should convert Mels back to Hertz because the start and end-points
    # should be at the desired frequencies.
    hertz = functions.mel_to_frequency(mels)

    # The frequency resolution required to put filters at the
    # exact points calculated above should be extracted.
    #  So we should round those frequencies to the closest FFT bin.
    freq_index = (
        np.floor(
            (coefficients + 1) *
            hertz /
            sampling_freq)).astype(int)

    # Initial definition
    filterbank = np.zeros([num_filter, coefficients])

    # The triangular function for each filter
    for i in range(0, num_filter):
        left = int(freq_index[i])
        middle = int(freq_index[i + 1])
        right = int(freq_index[i + 2])
        z = np.linspace(left, right, num=right - left + 1)
        tri = functions.triangle(z, left, middle, right)
        filterbank[i, left:right + 1] = tri

    return filterbank


def mfe(signal, filter_banks, sampling_frequency, frame_length,
        frame_stride, fft_length):

    # Stack frames
    frames = processing.stack_frames(
        signal,
        sampling_frequency=sampling_frequency,
        frame_length=frame_length,
        frame_stride=frame_stride)

    # calculation of the power sprectum
    power_spectrum = processing.power_spectrum(frames, fft_length)

    # Filterbank energies
    features = jnp.dot(power_spectrum, filter_banks.T)
    features = functions.zero_handling(features)

    return features
