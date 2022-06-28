#!/usr/bin/env python3

# debug extract a sample of clips from full audio based on pre calculated peak_dbs

import argparse
import util
import tqdm
import numpy as np
import sample_db
import soundfile

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--full-audio-npy', type=str, required=True)
opts = parser.parse_args()

# load full audio
full_audio_npy = np.load(opts.full_audio_npy)
print("full_audio_npy.shape", full_audio_npy.shape)

# extract a sample of clips from db
print("loading clip information")
db = sample_db.SampleDB()
clips = db.sample_clips(n=10, min_peak_db=10, min_start=100000)

# populate output
for i, (clip_id, clip_start, clip_end) in tqdm.tqdm(enumerate(clips)):
    fname = f"clip_{clip_id:06d}_{clip_start}.wav"
    clip = full_audio_npy[clip_start : clip_end]
    print(fname, clip.min(), clip.max(), clip.sum())
    soundfile.write(fname, clip, samplerate=22050)

