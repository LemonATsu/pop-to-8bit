# pop-to-8bit

This is a Python version implementaion of the [paper](https://lemonatsu.github.io/pdf/su17icassp.pdf), and you can also see some informations in our [website](https://lemonatsu.github.io).

Note that this version may generate slightly different result in compare to the original version, and the processing technique in [section 2.3](https://lemonatsu.github.io/pdf/su17icassp.pdf) of paper is omitted due to that fact that it can be achieved by tuning the parameter of [pYIN](https://code.soundsoftware.ac.uk/projects/pyin) plug-in.

The NMF constraint is also not implemented in this version due to its ineffectiveness of improving the conversion result.

## Prerequisites
- Python 3.4+
- [pYIN vamp plug-in](https://code.soundsoftware.ac.uk/projects/pyin)
- [LibROSA](http://librosa.github.io/librosa/)
- [Pypropack](https://github.com/jakevdp/pypropack)
- Numpy

## Usage 
You can simply convert your audio by:
``` 
python 8bits.py [-h] [-s SAMPLE_RATE] [--block_size BLOCK_SIZE]

                [--step_size STEP_SIZE]

                audio_path output_path
```
Tuning the ``step_size`` and ``block_size`` can help reach a more accurate pitch result.

## Acknowledgement
- [pYIN vamp plug-in](https://code.soundsoftware.ac.uk/projects/pyin) : Matthias Mauch, Dixon, Simon
- [LibROSA: 0.4.1](http://librosa.github.io/librosa/) :
Brian McFee; Matt McVicar; Colin Raffel; Dawen Liang; Oriol Nieto; Eric Battenberg; Josh Moore; Dan Ellis; Ryuichi YAMAMOTO; Rachel Bittner; Douglas Repetto; Petr Viktorin; João Felipe Santos; Adrian Holovaty
- [Pypropack](https://github.com/jakevdp/pypropack) : Jake Vanderplas
- [robust-matrix-decomposition](https://kastnerkyle.github.io/posts/robust-matrix-decomposition/) : Kyle Kastner
- [RPCA](https://github.com/apapanico/RPCA) : Alex Pananicolaou



