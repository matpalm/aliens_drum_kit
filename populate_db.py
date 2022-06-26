#!/usr/bin/env python3

# run once to populate db from scratch

import sample_db
import librosa
import os
import sys

db = sample_db.SampleDB()
db.create_if_required()

for line in sys.stdin:
    try:
        fname = line.strip()
        wav, sr = librosa.load(fname, sr=None)
        clip_len_sec = len(wav) / sr
        db.create_record(fname, clip_len_sec, sr)
    except Exception as e:
        print(f"failed to load [{fname}] [{str(e)}]", file=sys.stderr)