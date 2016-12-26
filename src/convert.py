import os
import numpy as np
import librosa
import scipy.io as spio
from nmf import nmf

t_path = os.path.dirname(__file__) + '/../templates/'

def convert_to_8bit(voice=None, accom=None, fs=44100., win_size=2048, hop_size=1024, 
                        voice_template='pulsenarrow', accom_template='spiky', max_iter=10):

    voice_8bit = None
    accom_8bit = None

    if voice is not None:
        voice_spec = librosa.core.stft(voice, n_fft=win_size, 
                                        win_length=win_size, hop_length=hop_size)
        voice_8bit = convert_8bit_voice(np.abs(voice_spec), voice_template, max_iter)

    if accom is not None:
        accom_spec = librosa.core.stft(accom, n_fft=win_size, 
                                        win_length=win_size, hop_length=hop_size)
        accom_8bit = convert_8bit_accom(np.abs(accom_spec), accom_template, max_iter)

    return voice_8bit, accom_8bit



def convert_8bit_accom(V, template, max_iter):

    W = load_mat(template, mat_type='d')
    H = nmf(V, W, max_iter)

    # keep the top 3 notes with the highest energy 
    # in each activation frame
    H = select_notes(H, n=3)
    
    # TODO : smooth activation matrix
    H = smooth_activation(H)

    # TODO : convert to time-domain

    accom_8bit = []

    return accom_8bit

def convert_8bit_voice(V, template, max_iter):

    W = load_mat(template, mat_type='d')
    H = nmf(V, W, max_iter)

    # TODO : pYin, and 'pick up' the notes

    # TODO : smooth activation

    # TODO : only leave one note in the activation
    #        to make the sound less noisy

    # TODO : covert to time-domain

    voice_8bit = []

    return voice_8bit

def load_mat(name, mat_type='d'):

    mat_name = ''
    full_path = ''

    if mat_type == 'd':
        mat_name = name + 'Temp'
        full_path = t_path + name + '_dic.mat'
    if mat_type == 't':
        mat_name = 'mat'
        full_path = t_path + name + '_td.mat'
        
    saved_mat = spio.loadmat(full_path)

    return saved_mat[mat_name]

def select_notes(H, n=3, t_len=3):

    selected = np.zeros(H.shape)

    for i in range(0, H.shape[1]):
        energy = sum_energy(H[:, i], t_len=t_len)
        _, indices = find_nlargest(energy, n=n)
        for j in indices:
            s = j * t_len
            selected[s:s+t_len, i] = H[s:s+t_len, i]

    return selected
    
def smooth_activation(H, t_len=3, hop_size=9):

    w = int(hop_size / 2)
    energy_mat = np.zeros((int(H.shape[0] / t_len), H.shape[1]))
    smoothed = np.zeros(H.shape)

    for i in range(0, H.shape[0]):
        energy_mat[:, i] = sum_energy(H[:, i], t_len=t_len).flatten()

    r, c = np.where(energy_mat == 0)
    time_len = energy_mat.shape[1]
    
    for i, j in zip(r, c):
        s = np.maximum(0, j - w)        # start
        e = np.minimum(time_len, j + w) # end
        
        indices = np.where(energy_mat[i, s:e] > 0)[0] 
        if (indices.shape[0] != 0) and (w > indices[0]) and (w < indices[-1]):
            offset = i * t_num
            # np.mean is also applicable
            smoothed[offset:offset+t_len, j] = np.median(H[offset:offset+t_len, j], axis=1)
            
    return smoothed

def convert_to_timedomain(H, hop_size, template, t_len=3):
    
    m, n = H.shape
    result = np.zeros((n * hop_size, 1))
    segments = []

    for i in range(0, m, t_len):
        note = int(i / t_len) 
    
        row = H[i, :]
        # pad 0 at each end, so the difference on both ends can be detected
        nonzero = np.concatenate(([0], np.greater(np.abs(row).view(np.int8), 0), [0]))
        # absdiff, so we will have value 1 at the begin and end points 
        # of the consecutive non-zero elements.
        # by doing so, we can mark the begin and end points of a segments
        absdiff = np.abs(np.diff(nonzero))
        indices = np.where(absdiff == 1)[0].reshape(-1, 2)
        
        for j in range(0, indices.shape[0]):
            # s and e represent the start and end point of a segment
            s = indices[j, 0]
            e = indices[j, 1]
            # s - e = 1 indicates the segment has only one frame
            rng = e - s
            if rng <= 1:
                continue

            signal_8bit = template[:, note]
            # turn into time-domain length
            rng = rng * hop_size
            # pad the signal if it's not long enough
            while np.shape[0] < rng : 
                signal_8bit = np.concatenate((signal_8bit, signal_8bit))
            energies = np.mean(H[i:i+t_len, s:e], axis=0) 
            # extend energy matrix to match time-domain length
            energies = np.repeat(energies, hop_size, axis=1)
            
            # turn into time-domain length
            s = s * hop_size
            e = e * hop_size
            result[s:e] = result[s:e] + signal_8bit[0:rng] * energies 
    
    return result

def sum_energy(v, t_len=3):

    c = np.abs(v).reshape(-1, 1)
    e_len = int(v.shape[0] / t_len)
    energy = np.zeros((e_len, 1))
    
    for i in range(0, e_len):
        s = i * t_len
        energy[i, 0] = np.max(c[s:s+t_len, 0])

    return energy

def find_nlargest(v, n=3):

    indices = []
    x = v.copy()

    for i in range(0, n):
        index = np.argmax(x)
        value = x[index]
        x[index] = -np.inf
        indices.append(index)

    return value, indices


if __name__ == '__main__':
    
    # example clips, already mono
    voice, fs = librosa.load('examples/c1_voice.wav', sr=44100.)
    accom, fs = librosa.load('examples/c1_accom.wav', sr=44100.)

    v8, a8 = convert_to_8bit(accom=accom)

