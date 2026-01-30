import numpy as np
import librosa

def pYIN(audio, fs=44100., hop_size=1024, block_size=2048, step_size=1024, 
            lowampsuppression=.1, onsetsensitivity=.7, prunethresh=.09):
    """
    This function uses librosa's built-in pYIN to conduct pitch analysis,
    and convert the pitch estimates into an activation matrix.
    
    Tuning the parameters here can improve the resulting 8-bit music.

    Parameters
    ----------
    audio : ndarray
        Audio input.        
    fs : float
        Sample rate.
    hop_size : int
        Hop size for the resulting activation matrix.
    block_size : int
        Block size for pYIN (frame_length).
    step_size : int
        Step size for pYin (hop_length for pyin).
    lowampsuppression : float
        The threshold for pYIN to suppress pitches that have low amplitude.
    onsetsensitivity : float
        Onset sensitivity for pYIN (not used in librosa's pyin).
    prunethresh : float
        Prune threshold for pYIN (not used in librosa's pyin).

    Return
    ------
    actl : ndarray
        Activation matrix generated from the pYIN pitch result.

    """

    length = len(audio)
    audio = np.asarray(audio)

    # Use librosa's built-in pyin
    # fmin and fmax cover typical singing voice range
    fmin = librosa.note_to_hz('C2')  # ~65 Hz
    fmax = librosa.note_to_hz('C7')  # ~2093 Hz

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=fs,
        frame_length=block_size,
        hop_length=step_size
    )

    # Convert f0 to activation matrix format
    actl = proc_frame_librosa(f0, voiced_flag, length, fs=fs,
                               hop_size=hop_size, step_size=step_size)

    return actl

def proc_frame_librosa(f0, voiced_flag, length, fs=44100., hop_size=1024,
                        step_size=1024, offset=34-1):
    """
    Parse the librosa pYIN result and generate a corresponding activation matrix.

    Parameters
    ----------
    f0 : ndarray
        Fundamental frequency estimates from librosa.pyin.
    voiced_flag : ndarray
        Boolean array indicating voiced frames.
    length : int
        Length of the audio input. It will be used to calculate the size of 
        resulting activation matrix.
    fs : float
        Sample rate.
    hop_size : int
        Hop size of the activation matrix.
    step_size : int
        Step size used in pyin analysis.
    offset : int
        The offset is used to offset the note number in order to match
        the pre-recorded 8-bit template, due to the fact that the index
        of template is starting from 0.

    Return
    ------
    frames : ndarray
        Resulting activation matrix.

    """

    flen = int(length / hop_size) - 1
    frames = np.zeros(flen)
    samples = np.zeros(length, dtype=int)

    # Convert f0 to midi and fill samples array
    for i, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        if voiced and not np.isnan(freq) and freq > 0:
            st = i * step_size
            dur = step_size
            midi = int(np.round(librosa.hz_to_midi(freq) - offset))
            midi = max(0, midi)  # Ensure non-negative
            end_idx = min(st + dur, length)
            samples[st:end_idx] = midi

    # Convert samples to frames
    for i in range(0, flen):
        d = samples[i * hop_size : (i + 1) * hop_size]
        counts = np.bincount(d)
        maxcount = np.argmax(counts)
        frames[i] = maxcount

    return frames

