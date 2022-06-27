#!/usr/bin/env python3

import sample_db
import sys
import util
import librosa
import os
import numpy as np
import tqdm

db = sample_db.SampleDB()
db.create_if_required()

fnames = []
lengths = []
sample_rates = []

for record in sys.stdin:
    try:
        _sample_id, fname, length, sample_rate = record.split("\t")
        fnames.append(fname)
        lengths.append(float(length))
        sample_rates.append(int(sample_rate))
    except Exception as e:
        print(f"FAILED [{record.strip()}] [{str(e)}]", file=sys.stderr)

print(len(fnames))
db.create_records(fnames, lengths, sample_rates)
