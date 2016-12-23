import numpy as np
from numpy.linalg import svd, norm

def ialm_RPCA(D, l=None, tol=5e-6, max_iter=1000, mu=1.25, rho=1.5):
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

    if l == None :
        l = 1. / np.sqrt(m)
    
    Y = D
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
    sv = 10.
    while not converged:
        i += 1        
        T = D - A_hat + (1. / u) * Y
        E_hat = np.maximum(T - (l / u), 0) + np.minimum(T + (l / u), 0)
        U, S, V = svd(D - E_hat + (1. / u) * Y, full_matrices=False)

        
        svp = (S > 1 / u).shape[0]
        #diag_S = np.diag(S)
        #svp = np.count_nonzero((diag_S > 1 / u) == 1)

        if svp < sv:
            sv = np.minimum(svp + 1, m)
        else:
            sv = np.minimum(svp + round(.05 * m), m)

        A_hat = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - (1. /u))), V[:svp, :])

        Z = D - A_hat - E_hat
        Y = Y + u * Z
        u = np.minimum(u * rho, u_bar)
        stop_criterion = norm(Z, 'fro') / d_norm
        if stop_criterion < tol or i >= max_iter:
            converged = True

    return A_hat, E_hat

