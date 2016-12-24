import numpy as np
import librosa
import scipy.io as spio
from nmf import nmf

t_path = '../templates/'


def convert_to_8bit(voice, accom, fs=44100., win_size=2048, hop_size=1024, 
                        voice_template='pulsenarrow', acomm_template='spiky', max_iter=10):

    voice_spect = librosa.core.stft(voice, n_fft=win_size, win_length=win_size, hop_length=hop_size)
    accom_spect = librosa.core.stft(accom, n_fft=win_size, win_length=win_size, hop_length=hop_size)

    convert_8bit_voice(np.abs(voice_spec), voice_template, max_iter)
    convert_8bit_accom(np.abs(accom_spec), accom_template, max_iter)



def convert_8bit_accom(V, template, max_iter):

    W = load_mat(template, mat_type='d')
    H = nmf(V, W, max_iter)

    # TODO : keep 3 notes with the highest energy 
    #        in each time frame


    # TODO : smooth activation matrix

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

