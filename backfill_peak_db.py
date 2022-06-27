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
    npy_fname = f"npys/samples/{util.fname_for(int(sample_id))}"
    array = np.load(npy_fname)
    return util.peak_db(array)

print("derive peak_dbs")
sample_ids = [int(l) for l in sys.stdin]
peak_dbs = []
p = Pool()
for result in tqdm.tqdm(p.imap(calculate_peak_db, sample_ids), total=len(sample_ids)):
    peak_dbs.append(result)

print("update database")
db = sample_db.SampleDB()
db.set_peak_dbs(sample_ids, peak_dbs)
