import os
import jax.numpy as jnp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_dir_exists_for_file(fname):
    ensure_dir_exists(os.path.dirname(fname))

def fname_for(sample_id):    
    return f"{(sample_id%100):02d}/{sample_id:06d}.npy"

class CrossFader(object):
    """ constant power cross fader """

    def __init__(self, clip_len, a_amt):
        self.clip_len = clip_len
        a_amt = jnp.tile(a_amt, (self.clip_len,))
        # see https://github.com/electro-smith/DaisySP/blob/master/Source/Dynamics/crossfade.cpp#L21
        self.scalar_1 = jnp.sin(a_amt * jnp.pi / 2)
        self.scalar_2 = jnp.sin((1.0 - a_amt) * jnp.pi / 2)

    def process(self, a, b):
        if len(a) != len(b):
            raise Exception("expected same length input samples")
        if len(a) != self.clip_len:
            raise Exception(
                f"expected clips to be length {self.clip_len}")
        return (a * self.scalar_1) + (b * self.scalar_2)

def show_waveform_and_spectrogram(a):
    a = np.array(a)

    plt.clf()
    fig, axs = plt.subplots(2, figsize=(20, 10))
        
    librosa.display.waveshow(a, sr=SR, ax=axs[0])

    # make n_fft smaller than default 2056 if <1s clips
    X = librosa.stft(a, n_fft=256) 
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=SR, x_axis='time', y_axis='hz', ax=axs[1])

    plt.show()        

def peak_db(clip):
    X = librosa.stft(clip, n_fft=256) 
    Xdb = librosa.amplitude_to_db(abs(X))
    return np.max(Xdb)