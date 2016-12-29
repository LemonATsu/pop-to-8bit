import numpy as np
from numpy.linalg import svd, norm
from pypropack import svdp

def ialm_RPCA(D, l=None, tol=1e-7, max_iter=1000, mu=1.25, rho=1.5):
    """
    Parameters
    ----------
    D : ndarray
        Input matrix, with size (m, n).
    l : float
        lamda, will be set to 1.0 / np.sqrt(m) if not specified.
    tol : float
        Tolerance for stopping criterion.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    A_hat : ndarray
            Low-rank array.
    E_hat : ndarray
            Sparse array.

    Copy Rights
    -----------
    This is a Python version of implementation based on :
    http://perception.csl.illinois.edu/matrix-rank/sample_code.html
    I do not own the copy right of this.

    
    Minming Chen, October 2009. Questions? v-minmch@microsoft.com 
    Arvind Ganesh (abalasu2@illinois.edu)

    Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
    Microsoft Research Asia, Beijing

    """

    m, n = D.shape
    print(D.shape)

    if l == None :
        l = 1. / np.sqrt(m)
    
    Y = D.copy()
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / l
    dual_norm = np.maximum(norm_two, norm_inf)
    Y = Y / dual_norm

    A_hat = np.zeros((m, n))
    E_hat = np.zeros((m, n))

    u = mu / norm_two
    u_bar = u * 1e7
    d_norm = norm(D, 'fro')
    
    i = 0
    converged = False
    stop_criterion = 1.
    sv = 10
    while not converged:
        i += 1        
        T = D - A_hat + (1. / u) * Y
        E_hat = np.maximum(T - (l / u), 0) + np.minimum(T + (l / u), 0)
        if choosvd(n, sv):
            U, S, V = svdp(D - E_hat + (1. / u) *Y, sv)
            print('svdp')
        else:
            U, S, V = svd(D-E_hat + (1. / u) * Y, full_matrices=False)
            print('svd-')

        
        # in np, S is a vector of 'diagonal value', 
        # so we don't need to do np.diag like the code in matlab
        svp = np.where(S > (1. / u))[0].shape[0]

        if svp < sv:
            sv = np.minimum(svp + 1, m)
        else:
            sv = np.minimum(svp + round(.05 * m), m)

        A_hat = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - (1. /u))), V[:svp, :])

        Z = D - A_hat - E_hat
        Y = Y + u * Z
        u = np.minimum(u * rho, u_bar)
        stop_criterion = norm(Z, 'fro') / d_norm

    return A_hat, E_hat

def choosvd(n_, d_):

    n = float(n_)
    d = float(d_)

    if n <= 100:
        if d / n <= 0.02:
            return True
        else:
            return False
    elif n <= 200:
        if d / n <= 0.06:
            return True
        else:
            return False
    elif n <= 300:
        if d / n <= 0.26:
            return True
        else:
            return False
    elif n <= 400:
        if d / n <= 0.28:
            return True
        else:
            return False
    elif n <= 500:
        if d / n <= 0.34:
            return True
        else:
            return False
    else:
        if d / n <= 0.38:
            return True
        else:
            return False

