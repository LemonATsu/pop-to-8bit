from ialm_rpca import ialm_RPCA
import numpy as np
import librosa


def svs_RPCA(wave, fs=44100., l=1., n_fft=1024, 
                    win_size=1024, mask_type=1, gain=1, power=1, scf=2./3.):
            
    hop_size = int(win_size / 4.)
    S_mix = scf * librosa.core.stft(wave, n_fft=n_fft, 
                                    hop_length=hop_size, win_length=win_size)
    
    S_mix = S_mix.T
    length = np.max(S_mix.shape)
    A_mag, E_mag = ialm_RPCA(np.power(np.abs(S_mix), power), l / np.sqrt(length))
    phase = np.angle(S_mix.T)

    L = (A_mag * np.exp(1j * phase).T)
    S = (E_mag * np.exp(1j * phase).T)

    if mask_type == 1:
        mask = np.greater(np.abs(S), (gain * np.abs(L)))
        S = S * mask
        L = S_mix - S
    
    voice = librosa.core.istft(S.T, hop_length=hop_size, win_length=win_size)
    accom = librosa.core.istft(L.T, hop_length=hop_size, win_length=win_size)
    voice = voice / np.max(np.abs(voice))
    accom = accom / np.max(np.abs(accom))

    return voice, accom


if __name__ == '__main__':
    clip, fs = librosa.load('examples/c1.wav', mono=False,sr=44100)
    clip = clip.T
    clip = clip[:,1] + clip[:,0]
    voice, accom = svs_RPCA(clip)
    
    librosa.output.write_wav('voice.wav', voice, sr=44100)
    librosa.output.write_wav('accom.wav', accom, sr=44100)

