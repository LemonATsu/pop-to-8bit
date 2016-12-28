import numpy as np
import vamp
import librosa

def pYIN(audio, fs=44100., hop_size=1024, block_size=2048, step_size=1024, 
            lowampsuppression=.1, onsetsensitivity=.7, prunethresh=.09):

    length = len(audio)
    audio = np.asarray(audio)

    parameters = {
             'prunethresh' : prunethresh,
        'lowampsuppression' : lowampsuppression,
        'onsetsensitivity' : onsetsensitivity,
    }

    data = vamp.collect(audio, fs, 'pyin:pyin', 'notes', 
                        parameters=parameters, block_size=block_size, step_size=step_size)['list']
    actl  = proc_frame(data, length, fs=fs, hop_size=hop_size)

    return actl

def proc_frame(data, length, fs=44100., hop_size=1024, offset=34-1):

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

