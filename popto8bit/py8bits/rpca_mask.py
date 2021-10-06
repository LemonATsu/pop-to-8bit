from .ialm_rpca import ialm_RPCA
import numpy as np
import librosa


def svs_RPCA(wave,
             fs=44100.,
             l=1.,
             n_fft=1024,
             win_size=1024,
             mask_type=1,
             gain=1,
             power=1,
             scf=2./3.,
             kmax=1):
    """
    Use robust principal components analysis(RPCA) to conduct the SVS task.

    Parameters
    ----------
    wave : ndarray
        Audio input.
    fs : float
        Sample rate.
    l : float
        Lambda parameter for RPCA.
    n_fft : int
        Same as the window size.
    win_size : int
        Window size for FFT.
    mask_type : int
        Type of mask that will be applied to the RPCA result.
        If equal to 1, it will apply median filter and binary mask
        to the RPCA results;otherwise it will use no mask.
    gain : float
        Gain for the A matrix.
    power : float
        Use (input_signal)^power as the input of RPCA.
    scf : float
        Scaling factor of the spectrogram of input wave.
    kmax : integer
        Scaling factor which controls the number of Krylov iterations
        in pypropack svdp.

    Returns
    -------
    voice : ndarray
        Voice separated from the input wave.
    accom : ndarray
        Instrumental accompaniment separated from the input wave.

    """
            
    hop_size = int(win_size / 4.)
    S_mix = scf * librosa.core.stft(wave,
                                    n_fft=n_fft,
                                    hop_length=hop_size,
                                    win_length=win_size)
    
    S_mix = S_mix.T
    length = np.max(S_mix.shape)
    A_mag, E_mag = ialm_RPCA(np.power(np.abs(S_mix), power),
                             l / np.sqrt(length),
                             kmax=kmax)
    # A_mag, E_mag, _ = rpca_alm(np.power(np.abs(S_mix), power),
    #                            l / np.sqrt(length))
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

