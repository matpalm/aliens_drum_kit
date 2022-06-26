#!/usr/bin/env python3

# extract samples of fixed length from all samples
# for samples longer than fixed length, take start of samples
# for samples shorter than fixed length, pad with zeros on either side

import argparse
import util
import sample_db
import librosa
import numpy as np
import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--clips-dir', type=str, required=True)
parser.add_argument('--min-sample-len-sec', type=float, required=0.2)
parser.add_argument('--max-sample-len-sec', type=float, required=0.5)
parser.add_argument('--clip-len-sec', type=float, default=0.3)
parser.add_argument('--sample-rate', type=int, default=22050)
opts = parser.parse_args()

target_clip_len = int(opts.sample_rate * opts.clip_len_sec)

def ensure_target_length(a):
    len_a = len(a)    
    if len_a == target_clip_len:
        return a
    elif len_a < target_clip_len:
        pad_amount = (target_clip_len - len_a) // 2
        zeros = np.zeros(pad_amount + 1)
        return np.concatenate([zeros, a, zeros])[:target_clip_len]
    elif len_a > target_clip_len:
        return a[:target_clip_len]

def process(job):
    sid, fname = job
    # load wav and sanity check size
    wav, sr = librosa.load(fname, sr=opts.sample_rate)
    wav_len = len(wav) / sr
    assert wav_len >= opts.min_sample_len_sec, ( fname, wav_len )
    assert wav_len <= opts.max_sample_len_sec, ( fname, wav_len )
    # crop, or pad, depending on size
    wav = ensure_target_length(wav)
    # save in subdirs
    clip_fname = f"{opts.clips_dir}/{util.fname_for(sid)}"
    util.ensure_dir_exists_for_file(clip_fname)
    np.save(clip_fname, wav)

util.ensure_dir_exists(opts.clips_dir)

db = sample_db.SampleDB()

jobs = list(db.clips_between_lengths(opts.min_sample_len_sec, opts.max_sample_len_sec))

for job in tqdm.tqdm(jobs):
    print(">", job)
    process(job)

#p = Pool(1)
#for _ in tqdm.tqdm(p.imap(process, jobs), total=len(jobs)):
#    pass


