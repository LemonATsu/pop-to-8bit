# pop-to-8bit

This is a Python version implementation of the [paper](https:///lemonatsu.github.io/files/su17icassp.pdf), and you can also see some information on our [website](https://lemonatsu.github.io).

Note that this version may generate slightly different results compared to the original version, and the processing technique in [section 2.3](https://lemonatsu.github.io/pdf/su17icassp.pdf) of the paper is omitted due to the fact that it can be achieved by tuning the pYIN parameters.

The NMF constraint is also not implemented in this version due to its ineffectiveness in improving the conversion result.

## Prerequisites
- Python 3.8+
- [LibROSA](http://librosa.github.io/librosa/)
- [SciPy](https://scipy.org/)
- NumPy

If you use Archlinux, there is an [AUR](https://aur.archlinux.org/packages/pop-to-8bit) package available.

## Installation

You can install the package with pip:

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

## Usage
You can simply convert your audio with:

```console
popto8bit [-h] [-s SAMPLE_RATE] [--block_size BLOCK_SIZE]
                [--step_size STEP_SIZE]
                audio_path output_path
```

### Example

```bash
popto8bit input.wav output_8bit.wav
```

Tuning the ``step_size`` and ``block_size`` can help reach a more accurate pitch result.

## Changes from Original

This fork includes the following updates for compatibility with modern Python and libraries:

- **Removed external pYIN vamp plugin dependency**: Now uses LibROSA's built-in `pyin` function for pitch detection
- **Removed pypropack dependency**: Replaced with `scipy.sparse.linalg.svds` for truncated SVD
- **Updated scipy imports**: Fixed deprecated `scipy.signal.hamming` to `scipy.signal.windows.hamming`
- **Updated librosa API**: Fixed `librosa.load()` to use keyword arguments
- **Simplified dependencies**: Only requires `librosa` and `scipy` (NumPy is installed as a dependency of these)

## Acknowledgement
- [pYIN algorithm](https://code.soundsoftware.ac.uk/projects/pyin) : Matthias Mauch, Dixon, Simon (now integrated via LibROSA)
- [LibROSA](http://librosa.github.io/librosa/) :
Brian McFee; Matt McVicar; Colin Raffel; Dawen Liang; Oriol Nieto; Eric Battenberg; Josh Moore; Dan Ellis; Ryuichi YAMAMOTO; Rachel Bittner; Douglas Repetto; Petr Viktorin; João Felipe Santos; Adrian Holovaty
- [SciPy](https://scipy.org/) : For sparse linear algebra (replacement for pypropack)
- [robust-matrix-decomposition](https://kastnerkyle.github.io/posts/robust-matrix-decomposition/) : Kyle Kastner
- [RPCA](https://github.com/apapanico/RPCA) : Alex Pananicolaou

## License

This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).



