#!/usr/bin/env python3

import argparse
import librosa
import soundfile
from .py8bits import core


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str, help='path of input audio')
    parser.add_argument("output_path", type=str, help='path of output file')
    parser.add_argument("-s", "--sample_rate", type=float, 
                            help='sample rate of input file', default=44100.)
    parser.add_argument("-c", help='Use SVS background instead of non-SVS one', action="store_false")
    parser.add_argument("--block_size", type=int, help='block size of pYIN', default=2048)
    parser.add_argument("--step_size", type=int, help='step size of pYIN', default=256)

    args = parser.parse_args()
    print('block_size : %d, step_size : %d' %(args.block_size, args.step_size))

    audio, fs = librosa.load(args.audio_path, args.sample_rate, mono=False)
    audio_8bit = core.convert(audio, fs=fs, v_centered=args.c, block_size=args.block_size, step_size=args.step_size)
    soundfile.write(args.output_path, audio_8bit, samplerate=int(fs))

if __name__ == '__main__':
    main()
