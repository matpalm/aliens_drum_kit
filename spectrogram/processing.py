import jax.numpy as jnp
import math

def preemphasis(signal, shift=1, cof=0.98):
    """preemphasising on the signal.

    Args:
        signal (array): The input signal.
        shift (int): The shift step.
        cof (float): The preemphasising coefficient. 0 equals to no filtering.

    Returns:
           array: The pre-emphasized signal.
    """

    rolled_signal = jnp.roll(signal, shift)
    return signal - cof * rolled_signal


def stack_frames(
        signal,
        sampling_frequency,
        frame_length,
        frame_stride):
    """Frame a signal into overlapping frames.

    Args:
        sig (array): The audio signal to frame of size (N,).
        sampling_frequency (int): The sampling frequency of the signal.
        frame_length (float): The length of the frame in second.
        frame_stride (float): The stride between frames.

    Returns:
            array: Stacked_frames-Array of frames of size (number_of_frames x frame_len).

    """

    # Initial necessary values
    length_signal = len(signal)
    frame_sample_length = math.ceil(sampling_frequency * frame_length)
    frame_stride = math.ceil(sampling_frequency * frame_stride)
    x = (length_signal - (frame_sample_length - frame_stride))
    numframes = math.floor(x / frame_stride)

    # Getting the indices of all frames.
    indices = (jnp.tile(jnp.arange(0, frame_sample_length),
                       (numframes, 1)) +
               jnp.tile(jnp.arange(0, numframes * frame_stride, frame_stride),
                       (frame_sample_length, 1)).T)
    indices = jnp.array(indices, dtype=jnp.int32)

    # Extracting the frames based on the allocated indices.
    frames = signal[indices]
    return frames


def fft_spectrum(frames, fft_points=512):
    """This function computes the one-dimensional n-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT). Please refer to
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html
    for further details.

    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.

    Returns:
            array: The fft spectrum.
            If frames is an num_frames x sample_per_frame matrix, output
            will be num_frames x FFT_LENGTH.
    """
    SPECTRUM_VECTOR = jnp.fft.rfft(frames, n=fft_points, axis=-1, norm=None)
    return jnp.absolute(SPECTRUM_VECTOR)


def power_spectrum(frames, fft_points=512):
    """Power spectrum of each frame.

    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.

    Returns:
            array: The power spectrum.
            If frames is an num_frames x sample_per_frame matrix, output
            will be num_frames x fft_length.
    """
    return 1.0 / fft_points * jnp.square(fft_spectrum(frames, fft_points))
