import argparse
import os
from glob import glob
from distutils.dir_util import copy_tree
from scipy.signal import resample_poly
import soundfile as sf

parser = argparse.ArgumentParser("Script for resampling a dataset")
parser.add_argument("source_dir", type=str)
parser.add_argument("out_dir", type=str)
parser.add_argument("original_sr", type=int)
parser.add_argument("target_sr", type=int)
parser.add_argument("extension", type=str, default="wav")


def main(out_dir, original_sr, target_sr, extension):
    assert original_sr >= target_sr, "Upsampling not supported"
    wavs = glob(os.path.join(out_dir, "*.{}".format(extension)))
    for wav in wavs:
        data, fs = sf.read(wav)
        assert fs == original_sr
        data = resample_poly(data, target_sr, fs)
        sf.write(wav, data)


if __name__ == "__main__":
    args = parser.add_argument()
    copy_tree(args.source_dir, args.out_dir)
    main(args.out_dir, args.original_sr, args.target_sr, args.extension)