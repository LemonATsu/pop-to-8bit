#!/usr/bin/env python3
"""pop-8-bit"""

import argparse
import librosa
import soundfile
from .py8bits import core


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser()

    arguments = [
      [("audio_path", ),
       {"type": str,
        "help": 'path of input audio'}],
      [("output_path", ),
       {"type": str,
        "help": 'path of output file'}],
      [("-s", "--sample_rate"),
       {"type": float,
        "help": 'sample rate of input file (default: 44100)',
        "default": 44100.}],
      [("-c", ),
       {"help": 'Use SVS background instead of non-SVS one (default: False)',
       "action": "store_false"}],
      [("--block_size", ),
       {"type": int,
        "help": 'block size of pYIN (default: 2048)',
        "default": 2048}],
      [("--step_size", ),
       {"type": int,
        "help": 'step size of pYIN (default: 256)',
        "default": 256}],
      [("--kmax", ),
       {"type": int,
        "help": 'krylov iterations factor (default: 1)',
        "default": 1}]
    ]

    for argument in arguments:
        args, kwargs = argument
        parser.add_argument(*args, **kwargs)

    args = parser.parse_args()

    print(f'block_size : {args.block_size}, step_size : {args.step_size}')

    audio, fs = librosa.load(args.audio_path,
                             args.sample_rate,
                             mono=False)

    audio_8bit = core.convert(audio,
                              fs=fs,
                              v_centered=args.c,
                              block_size=args.block_size,
                              step_size=args.step_size,
                              kmax=args.kmax)

    soundfile.write(args.output_path,
                    audio_8bit,
                    samplerate=int(fs))


if __name__ == '__main__':
    main()
