import util
import numpy as np
import jax
from jax import vmap, jit
import jax.numpy as jnp
from spectrogram import mfe

# load a sample of clips
clip_fnames = [
    'clips/65/293165.npy',
    'clips/49/544749.npy',
    'clips/60/790960.npy',    
    'clips/40/875640.npy',
    'clips/26/394926.npy'
]

SR = 22050
CLIP_LEN_SEC = int(SR * 0.3)
clips = jnp.stack([np.load(f) for f in clip_fnames])
assert clips.shape[1] == CLIP_LEN_SEC

def calculate_spectrogram(clip):
    return mfe.generate_features(clip, SR)

spectrogram = calculate_spectrogram(clips[0])
print(spectrogram.shape)
exit()

# construct cross fader
x_fader = util.CrossFader(clip_len=CLIP_LEN_SEC, a_amt=0.75)

# define method to take three clips, ijk, and output a crossfaded pairs ij & ik

def pair(i, j, k):    
    clip_i_j = x_fader.process(a=clips[i], b=clips[j])
    clip_i_k = x_fader.process(a=clips[i], b=clips[k])
    return jnp.stack([clip_i_j, clip_i_k])

# check vectorised form
i_s = jnp.array([3, 1])
j_s = jnp.array([1, 5])
k_s = jnp.array([4, 9])
pairs = vmap(pair)
combos = pairs(i_s, j_s, k_s)
print("combos.shape", combos.shape)  # (2, 2, 6615)  (batch_size=2, |cij_cik|=2, CLIP_LEN=6615)

# TODO run through spectrogram
# make function take one audio sequence and return (T, F) output
# then vmap it twice

# (2, 2, time, frequencies)

# OR, could make the to_spectrogram function take two input
# recall: the model just needs to take one audio sequence -> spectrogram

# for inference, we want single clip -> embeddings
# for training we need to build ij, ik combos and do dot product etc for contrastive