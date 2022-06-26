import numpy as np
import jax.numpy as jnp


def frequency_to_mel(f):
    """converting from frequency to Mel scale.

    :param f: The frequency values(or a single frequency) in Hz.
    :returns: The mel scale values(or a single mel).
    """
    return 1127 * np.log(1 + f / 700.)


def mel_to_frequency(mel):
    """converting from Mel scale to frequency.

    :param mel: The mel scale values(or a single mel).
    :returns: The frequency values(or a single frequency) in Hz.
    """
    return 700 * (np.exp(mel / 1127.0) - 1)


def triangle(x, left, middle, right):
    out = np.zeros(x.shape)
    first_half_idx = np.logical_and(left < x, x <= middle)
    first_half_val = (x[first_half_idx] - left) / (middle - left)
    out[first_half_idx] = first_half_val
    second_half_idx = np.logical_and(middle <= x, x < right)
    second_half_val = (right - x[second_half_idx]) / (right - middle)
    out[second_half_idx] = second_half_val
    return out


def zero_handling(x):
    """
    This function handle the issue with zero values if the are exposed
    to become an argument for any log function.
    :param x: The vector.
    :return: The vector with zeros substituted with epsilon values.
    """
    return jnp.where(x == 0, np.finfo(float).eps, x)
