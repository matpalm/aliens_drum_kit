import util
import numpy as np
import jax
from jax import vmap, jit
import jax.numpy as jnp

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

x_fader = util.CrossFader(clip_len=CLIP_LEN_SEC, a_amt=0.75)

def pair(i, j, k):    
    clip_i_j = x_fader.process(a=clips[i], b=clips[j])
    clip_i_k = x_fader.process(a=clips[i], b=clips[k])
    return jnp.stack([clip_i_j, clip_i_k])

pairs = jit(vmap(pair))

i_s = jnp.array([3, 1])
j_s = jnp.array([1, 5])
k_s = jnp.array([4, 9])
combos = pairs(i_s, j_s, k_s)
print("combos.shape", combos.shape)

