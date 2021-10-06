import numpy as np
import vamp
import librosa

def pYIN(audio, fs=44100., hop_size=1024, block_size=2048, step_size=1024, 
            lowampsuppression=.1, onsetsensitivity=.7, prunethresh=.09):
    """
    This function will call the pYIN vamp plug-in to conduct the pitch analysis,
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
        Block size for pYIN.
    step_size : int
        Step size for pYin.
    lowampsuppression : float
        The threshold for pYIN to suppress pithches that have low amplitude. 
    onsetsensitivity : float
        Onset sensitivity for pYIN.
    prunethresh : float
        Prune threshold for pYIN.

    Return
    ------
    actl : ndarray
        Activation matrix generated from the pYIN pitch result.

    """


    length = len(audio)
    audio = np.asarray(audio)

    parameters = {
        'prunethresh'       : prunethresh,
        'lowampsuppression' : lowampsuppression,
        'onsetsensitivity'  : onsetsensitivity,
    }

    data = vamp.collect(audio, fs, 'vamp-pyin-f0:pyin', 'notes', 
                        parameters=parameters, block_size=block_size, step_size=step_size)['list']
    actl  = proc_frame(data, length, fs=fs, hop_size=hop_size)

    return actl

def proc_frame(data, length, fs=44100., hop_size=1024, offset=34-1):
    """
    Parse the pYIN result and generate a corresponding activation matrix.

    Parameters
    ----------
    data : array
        Array of dictionary such that each dictionary contains the duration 
        and timestamp of a pitch.
    length : int
        Length of the audio input. It will be used to calculate the size of 
        resulting activation matrix.
    fs : float
        Sample rate.
    hop_size : int
        Hop size of the activation matix.
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
    samples = np.zeros(length, dtype=np.int)
    hz_samples = np.zeros(length, dtype=np.int)

    for d in data:
        dur = int(float(d['duration']) * fs)
        st = int(float(d['timestamp']) * fs)
        midi = int(np.round(librosa.hz_to_midi(float(d['values'])) - offset))
        samples[st : st+dur] = midi
        hz_samples[st : st+dur] = float(d['values'])

    for i in range(0, flen):
        d = samples[i * hop_size : (i + 1) * hop_size]
        counts = np.bincount(d)
        maxcount = np.argmax(counts)
        frames[i] = maxcount

    return frames

