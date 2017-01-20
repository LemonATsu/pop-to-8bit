# Automatic Conversion of Pop Music into Chiptunes for 8-Bit Pixel Art

This is a Python version implementaion of the [paper](https://lemonatsu.github.io/pdf/su17icassp.pdf).

Note that this version may generate slightly different result in compare to the original version, and the processing technique in [section 2.3](https://lemonatsu.github.io/pdf/su17icassp.pdf) of paper is omitted due to that fact that it can be achieve by tuning the parameter of [pYIN](https://code.soundsoftware.ac.uk/projects/pyin) plug-in.

## Prerequisites
- Python 3.4+
- [pYIN vamp plug-in](https://code.soundsoftware.ac.uk/projects/pyin)
- [LibROSA](http://librosa.github.io/librosa/)
- [Pypropack](https://github.com/jakevdp/pypropack)
- Numpy

## Usage 
`` You can simply convert your audio by:
usage: 8bits.py [-h] [-s SAMPLE_RATE] [--block_size BLOCK_SIZE]

                [--step_size STEP_SIZE]

                audio_path output_path


``

