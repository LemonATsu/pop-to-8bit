import numpy as np

def nmf(V, W, max_iter=10, random_state=0):
    """
    Parameters
    ----------
    V : ndarray
        Source matrix of V = WH.
    W : ndarray
        'Dictionary' matrix of V = WH.
    max_iter : int
        Maximum number of iterations.
    random_state : int
        Random seed for numpy.

    Returns
    -------
    H : ndarray
        Activation matrix of V = WH

    """

    np.random.seed(random_state)

    F, T = V.shape
    K = W.shape[1]
    ones = np.ones((F, T))
    H = np.rand(K, T)
    eps = np.finfo(float).eps

    for i in range(0, max_iter):
        H = H * (np.dot(W.T, V / (np.dot(W, H) + eps))) / (np.dot(W.T, ones))

    return H
    
