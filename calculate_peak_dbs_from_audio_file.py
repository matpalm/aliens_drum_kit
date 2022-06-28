#!/usr/bin/env python3

# extract samples of fixed length from a long wav (e.g. the audio of a movie)

import argparse
import util
import librosa
import numpy as np
import sample_db

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input', type=str, required=True)    
parser.add_argument('--clip-len-sec', type=float, default=0.3)
parser.add_argument('--clip-stride-sec', type=float, default=0.05)
parser.add_argument('--sample-rate', type=int, default=22050)
opts = parser.parse_args()

clip_len = int(opts.sample_rate * opts.clip_len_sec)
clip_stride = int(opts.sample_rate * opts.clip_stride_sec)
print("output clip_len", clip_len, "clip_stride", clip_stride)

full_audio, _sr = librosa.load(opts.input, sr=opts.sample_rate)
print("|full_audio|", len(full_audio))

db = sample_db.SampleDB()

offset = 0
records = []
n_written = 0
while True:

    # extract subclip, exit when we've reach end
    clip_start = offset
    clip_end = offset + clip_len    
    sub_clip = full_audio[clip_start : clip_end]
    if len(sub_clip) != clip_len:
        break

    # derive peak_db
    peak_db = util.peak_db(sub_clip)
        
    # write record to db (batched)
    records.append((clip_start, clip_end, peak_db))
    if len(records) == 1024:
        db.create_clip_records(records)
        n_written += 1024
        print("n_written", n_written)
        records = []

    # step forward
    offset += clip_stride

# flush final records
db.create_clip_records(records)