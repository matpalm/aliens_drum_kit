#!/usr/bin/env python3

# backfill the sample db with peak db values

import sample_db
import sys
import util
import librosa
import os
import numpy as np
import tqdm
from multiprocessing import Pool

def calculate_peak_db(sample_id):    
    clip_fname = f"clips/samples/{util.fname_for(int(sample_id))}"
    clip = np.load(clip_fname)
    X = librosa.stft(clip, n_fft=256) 
    Xdb = librosa.amplitude_to_db(abs(X))
    return np.max(Xdb)    

print("derive peak_dbs")
sample_ids = [int(l) for l in sys.stdin]
peak_dbs = []
p = Pool()
for result in tqdm.tqdm(p.imap(calculate_peak_db, sample_ids), total=len(sample_ids)):
    peak_dbs.append(result)

print("update database")
db = sample_db.SampleDB()
db.set_peak_dbs(sample_ids, peak_dbs)

# for sample_id in tqdm.tqdm(sample_ids):
#     try:
#         clip_fname = f"clips/samples/{util.fname_for(sample_id)}"
#         clip = np.load(clip_fname)
#         X = librosa.stft(clip, n_fft=256) 
#         Xdb = librosa.amplitude_to_db(abs(X))
#         peak_db = np.max(Xdb)
#         db.set_peak_db(sample_id, peak_db)
#         print(sample_id)
#     except Exception as e:
#         print(f"FAILED [{sid}] [{str(e)}]", file=sys.stderr)
