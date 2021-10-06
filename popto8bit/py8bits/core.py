import os
import numpy as np
import librosa
import scipy.io as spio
from scipy.signal import hamming
from .nmf import nmf
from .pyin import pYIN
from .svs import svs

t_path = os.path.dirname(__file__) + '/../templates/'

def convert(wave, fs=44100, voice_scale=.3, accom_scale=.4, v_centered=True, verbose=True, **kwargs):
    """
    The top function of the whole converting process. 
    It will first perform singing voice separation, and then process the 
    separated signal, and synthesize them in the time domain.

    Parameters
    ----------
    wave : ndarray
        Audio input.
    fs : int
        Sample rate.
    voice_scale : float
        The magnitude scale of converted voice signal.
    accom_scale : float
        The magnitude scale of converted accompaniment signal.
    v_centered : bool
        If the voice in the wave is centered or not.
        If so, we can obtain a clear accompaniment signal by simply subtracting 
        the left and right channel.
    verbose : bool
        To print the debug log.
    **kwargs :
        Keyword arguments.

    Return
    ------
        The converted 8bits signal.

    """


    if verbose == True :
        print('Start separating audio ...')
    voice, accom = svs(wave, fs, v_centered=v_centered)
    if verbose == True :
        print('Done. \nStart converting to 8-bit ...')
    voice_8bit, accom_8bit = convert_to_8bit(voice, accom, fs, **kwargs)
    if verbose == True :
        print('Done.')

    return voice_8bit * voice_scale + accom_8bit * accom_scale 

def convert_to_8bit(voice=None, accom=None, fs=44100., win_size=2048, hop_size=1024, 
                        voice_template='pulsenarrow', accom_template='spiky', max_iter=10, **kwargs):
    """
    Convert signal to 8bit.
    
    Parameters
    ----------
    voice : ndarray
        Time domain signal of voice that you want to convert to 8bit version.
    accom : ndarray
        Time domain signal of accompaniment that you want to convert to 8bit version.
    fs : int
        Sample rate.
    win_size : int
        Window size for STFT.
    hop_size : int
        Hop size for STFT.
    voice_template : str
        Name of 8bits template you want to use to convert your voice signal.
    accom_template : str
        Name of 8bits template you want to use to convert your accompaniment signal.
    max_iter : int
        Number of iterations for NMF.
    **kwargs : 
        Keyword arguments for pYIN.

    Returns
    -------
    voice_8bit : ndarray
        Converted 8bits version of voice signal.
    accom_8bit : ndarray
        Converted 8bits version of accompaniment signal.

    """

    voice_8bit = None
    accom_8bit = None
    window = hamming(win_size, sym=False)

    if voice is not None:
        voice_spec = librosa.core.stft(voice, n_fft=win_size, 
                                           win_length=win_size, window=window, hop_length=hop_size)
        voice_8bit = convert_8bit_voice(voice, np.abs(voice_spec), voice_template, fs=fs, 
                                            hop_size=hop_size, max_iter=max_iter, **kwargs)

    if accom is not None:
        accom_spec = librosa.core.stft(accom, n_fft=win_size, 
                                           win_length=win_size, window=window, hop_length=hop_size)
        accom_8bit = convert_8bit_accom(np.abs(accom_spec), accom_template, 
                                            hop_size=hop_size, max_iter=max_iter)

    return voice_8bit, accom_8bit

def convert_8bit_voice(voice, V, template, fs=44100, energy=None,
                          hop_size=1024, max_iter=10, **kwargs):
    """
    Convert voice signal to 8bit version.

    Parameters
    ----------
    voice : ndarray
        Time domain signal of voice.
    V : ndarray
        Magnitude spectrogram of voice.
    template : str
        Name of 8bits template.
    fs : int
        Sample rate.
    energy : float
        Energy you want to assign to the 8bits version signal.
        Default value is np.max(H) / 4, where H is the activaion matrix
        of voice from NMF.
    hop_size : int
        Hop size of spectrogram.
    max_iter : int
        Maximum number of iterations. 
    **kwargs :
        Keyword arguments for pYIN.

    Return
    ------
    voice_8bit : ndarray
        Time domain signal of 8bits voice.
        
    """

    W = load_mat(template, mat_type='d')
    T = load_mat(template, mat_type='t')
    H = nmf(V, W, max_iter)

    # perfrom pYIN to get pitch information
    midi = pYIN(voice, fs=fs, hop_size=hop_size, **kwargs)
    # generate activation
    if energy is None:
        energy = np.max(H)
    H = generate_activation(midi, energy, H.shape)
    # smooth activations
    H = smooth_activation(H)
    # covert to time-domain
    voice_8bit = synthesize_in_timedomain(H, hop_size, T)


    return voice_8bit

def convert_8bit_accom(V, template, hop_size=1024, max_iter=10):
    """
    Convert accompaniment signal to 8bits music.

    Parameters
    ----------
    V : ndarray
        Magnitude spectrogram of accompaniment signal.
    template : str
        Name of 8bits template you want to use.
    hop_size : int
        Hop size of spectrogram.
    max_iter : int
        Maximum number of iterations.

    Return
    ------
    accom_8bit : ndarray
        Time domain signal of 8bits accompaniment.

    """

    W = load_mat(template, mat_type='d')
    T = load_mat(template, mat_type='t')
    H = nmf(V, W, max_iter)

    # keep the top 3 notes with the highest energy 
    # in each activation frame
    H = select_notes(H, n=3)
    # smooth activation matrix
    H = smooth_activation(H)
    # convert to time-domain
    accom_8bit = synthesize_in_timedomain(H, hop_size, T)

    return accom_8bit

def load_mat(name, mat_type='d'):
    """
    Helper function for loading 8bits template files.
    The templates we provided is generated in MATLAB,
    and thus we have to use scipy to parse it.

    Parameters
    ----------
    name : str
        Name of template file.
    mat_type : str
        Type of template. 'd' is for the template in 
        frequency domain, while 't' is in time-domain.
        
        The variable stored in _dic.mat (frequency domain templates)
        is usually named as 'template name'+Temp, while the one in
        time domain is mat.

    Return
    ------
    saved_mat : ndarray
        Template matrix.

    """

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
        value, _ = find_nlargest(energy, n=n)
        energy[energy < value] = 0
        indices = np.where(energy != 0)[0]
        for j in indices:
            s = j * t_len
            selected[s:s+t_len, i] = H[s:s+t_len, i]

    return selected
    
def smooth_activation(H, t_len=3, hop_size=9):
    """
    Smoothing the activation matrix by applying median filter to
    the 'gap' between activated frame.
    
    We only interested in the unactivated frames that are between
    the activated one.

    Parameters
    ----------
    H : ndarray
        Activation matrix.
    t_len : int
        The number of components in a template. t_len=3 by default.
    hop_size : int
        The width of the median filter.

    Return
    ------
    smoothed : ndarray
        Smoothed activation matrix.

    """

    w = int(hop_size / 2)
    energy_mat = np.zeros((int(H.shape[0] / t_len), H.shape[1]))
    smoothed = np.copy(H)

    for i in range(0, H.shape[1]):
        energy_mat[:, i] = sum_energy(H[:, i], t_len=t_len).flatten()

    r, c = np.where(energy_mat == 0)
    time_len = energy_mat.shape[1]
    
    for i, j in zip(r, c):
        s = np.maximum(0, j - w)        # start
        e = np.minimum(time_len, j + w + 1) # end
        
        # find the activated frames near the center.
        indices = np.where(energy_mat[i, s:e] != 0)[0] 

        if (indices.shape[0] != 0) and (w > indices[0]) and (w < indices[-1]):
            offset = i * t_len
            # np.mean is also applicable
            smoothed[offset:offset+t_len, j] = np.median(H[offset:offset+t_len, s:e], axis=1)
            
    return smoothed

def synthesize_in_timedomain(H, hop_size, template, t_len=3):
    """
    Synthesize the 8bits signal in time-domain.

    Parameters
    ----------
    H : ndarray
        Activation matrix obtained from either NMF or activation generation.
    hop_size : ndarray
        Hop size of the activation matrix.
    template : ndarray
        Time domain 8bits template.
    t_len :
        The number of components in the frequency domain 8bits templates.

    Return
    ------
    result : ndarray
        Synthesized 8bits signal.
    
    """

    m, n = H.shape
    result = np.zeros((n * hop_size, 1))
    segments = []

    for i in range(0, m, t_len):
        note = int(i / t_len) 
    
        row = H[i, :]
        # pad 0 at each end, so the difference on both ends can be detected
        nonzero = np.concatenate(([0], np.greater(np.abs(row), 0).view(np.int8), [0]))
        # absdiff, so we will have value 1 at the begin and end points 
        # of the consecutive non-zero elements.
        # by doing so, we can mark the begin and end points of a segments
        absdiff = np.abs(np.diff(nonzero))
        indices = np.where(absdiff == 1)[0].reshape(-1, 2)
        signal_8bit = template[:, note]
        
        for j in range(0, indices.shape[0]):
            # s and e represent the start and end point of a segment
            s = indices[j, 0]
            e = indices[j, 1]
            # s - e = 1 indicates the segment has only one frame
            rng = e - s
            if rng <= 1:
                continue

            # turn into time-domain length
            rng = rng * hop_size
            # pad the signal if it's not long enough
            while signal_8bit.shape[0] < rng : 
                signal_8bit = np.concatenate((signal_8bit, signal_8bit))
            energies = np.mean(H[i:i+t_len, s:e], axis=0) 
            # extend energy matrix to match time-domain length
            energies = np.repeat(energies, hop_size)
            
            # turn into time-domain length
            s = s * hop_size
            e = e * hop_size
            result[s:e, 0] = result[s:e, 0] + signal_8bit[0:rng] * energies 
    try:
        result = result / (np.max(result) - np.min(result))
    except ValueError:
        print('Warning : invalid value occurs, try changing step_size/block_size to avoid this problem.')
        
    return result

def generate_activation(midi, energy, shape, t_len=3):
    """
    Create an activation matrix by converting a given midi vector.
    
    Parameters
    ----------
    midi : ndarray
           Vector of midis, obtained from pYIN.
    energy : float
           The energy of each activated frame you want to set.
    shape : tuple
           Shape of activation matrix.
    t_len : int
           The number of components in a template. t_len=3 by default.

    Return
    ------
    H : ndarray
        Activation martrix with shape=shape.

    """

    H = np.zeros(shape)
    indices = np.where(midi != 0)[0]

    for i in range(0, indices.shape[0]):
        index = indices[i]
        s = int((midi[index]-1) * t_len)
        H[s:s+t_len, index] = energy

    return H

def sum_energy(v, t_len=3):
    """
    Compute the energy of templates in a given time frame,
    by finding the max value in templates' component.

    Parameters
    ----------
    v : ndarray
        Time frame as a vector.
    t_len : int
        The number of components in a template. t_len=3 by default.

    Returns
    -------
    energy : ndarray
        Vectors of activation energies.

    """

    c = np.abs(v).reshape(-1, 1)
    e_len = int(v.shape[0] / t_len)
    energy = np.zeros((e_len, 1))
    
    for i in range(0, e_len):
        s = i * t_len
        energy[i, 0] = np.max(c[s:s+t_len, 0])

    return energy

def find_nlargest(v, n=3):
    """
    Find the top n candidates with the strongest activavtion in a given time frame.

    Parameters
    ----------
    v : ndarray
        Time frame as a vector.
    n : int
        Number of candidates that we have to find.

    Returns
    -------
    value : float
        The n-th strongest activation.
    indices : list
        Indices of candidates.

    """

    indices = []
    x = v.copy()

    for i in range(0, n):
        index = np.argmax(x)
        value = x[index].copy()
        x[index] = -np.inf
        indices.append(index)

    return value, indices

