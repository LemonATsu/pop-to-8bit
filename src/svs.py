from rpca_mask import svs_RPCA

def svs(wave, fs=44100, v_centered=True):
    """
        assume vocal is centered.
    """
    X = wave

    if wave.shape[0] < wave.shape[1]:
        X = X.T
    if X.ndim > 1:
        rpca_wave = X[:, 1] + X[:, 0]

    voice, accom = svs_RPCA(rpca_wave, fs)

    if v_centered == True:
        accom = (X[:, 1] - X[:, 0]) / 2 

    return voice, accom

