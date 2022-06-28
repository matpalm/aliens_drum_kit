#!/usr/bin/env python3

import argparse
import glob
from sample_db import SampleDB
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--samples-input-dir', type=str, required=True)
parser.add_argument('--full-audio-npy', type=str, required=True)
parser.add_argument('--min-clip-peak-db', type=float, default=-20)
parser.add_argument('--output-npy', type=str, required=True)
parser.add_argument('--output-tsv', type=str, required=True)
parser.add_argument('--debug-max-samples-clips', type=int, default=None)
opts = parser.parse_args()

# output records for packing into tsv
output_tsv = open(opts.output_tsv, 'w')
print("id\ttype", file=output_tsv)

# collect all sample ids on disk 
sample_fnames = sorted(glob.glob(f"{opts.samples_input_dir}/*/*npy"))
if opts.debug_max_samples_clips is not None:
    sample_fnames = sample_fnames[:opts.debug_max_samples_clips]
print("|sample_fnames|", len(sample_fnames))

# collect all clip ids
db = SampleDB()
clips = db.sample_clips(min_peak_db=opts.min_clip_peak_db,                     
                        n=opts.debug_max_samples_clips)
_id, clip_start, clip_end = clips[0]
clip_len = clip_end - clip_start                    
print("|clips|", len(clips))

# load full audio
print("loading full audio")
full_audio_a = np.load(opts.full_audio_npy)

# allocate output array
N = len(sample_fnames) + len(clips)
output_a = np.empty(shape=(N, clip_len), dtype=np.float32)

# pack samples first ...
print("packing samples")
for i, sample_fname in enumerate(tqdm(sample_fnames)):
    output_a[i] = np.load(sample_fname)
    sample_id = int(sample_fname.split("/")[-1].replace(".npy", ""))  # clumsy
    print(f"{sample_id}\ts", file=output_tsv)

# ... then pack clips
offset = len(sample_fnames)
print("packing clips")
for i, (clip_id, clip_start, clip_end) in enumerate(tqdm(clips)):
    output_a[offset + i] = full_audio_a[clip_start : clip_end]
    print(f"{clip_id}\tc", file=output_tsv)
    print("clip", clip_id, clip_start, clip_end)

# write output and flush metadata
np.save(opts.output_npy, output_a)
output_tsv.close()




