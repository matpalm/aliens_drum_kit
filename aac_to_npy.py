import argparse
import librosa
import numpy as np

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
opts = parser.parse_args()

wav, _sr = librosa.load(opts.input, sr=22050)
print(wav.shape)
np.save(opts.output, wav)
