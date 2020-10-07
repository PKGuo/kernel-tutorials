import numpy as np
from scipy.sparse.linalg import svds as svd
from scipy.sparse.linalg import eigs as speig
import scipy

from tqdm import tqdm as tqdm

def compute_P(A_c, S, A_r, rcond=1e-12):
    """ Computes the latent-space projector for the feature matrix """
    SA = np.matmul(S, A_r)
    SA = np.matmul(SA, SA.T)

    v_SA, U_SA = np.linalg.eigh(SA)
    v_SA[v_SA < rcond] = 0

    return np.matmul(U_SA, np.diagflat(np.sqrt(v_SA)))

def get_Ct(X, Y, alpha=0.5, rcond=1e-6):
    """
        Creates the PCovR modified covariance
        ~C = (alpha) * X^T X +
             (1-alpha) * (X^T X)^(-1/2) ~Y ~Y^T (X^T X)^(-1/2)

        where ~Y is the properties obtained by linear regression.
    """

    cov = X.T @ X

    # changing these next two lines can cause a LARGE error
    Cinv = np.linalg.pinv(cov)
    Cisqrt = scipy.linalg.sqrtm(Cinv)

    # parentheses speed up calculation greatly
    Y_hat = Cisqrt @ (X.T @ Y)

    if(len(Y_hat.shape) < 2):
        Y_hat = Y_hat.reshape((-1, 1))

    C_lr = Y_hat @ Y_hat.T

    C_pca = cov
    C = alpha*C_pca + (1.0-alpha)*C_lr

    return C

def get_Kt(X, Y, alpha=0.5):
    """
        Creates the PCovR modified kernel distances
        ~K = (alpha) * X X^T +
             (1-alpha) * Y Y^T

    """

    K = np.zeros((X.shape[0], X.shape[0]))
    K += (1 - alpha) * Y @ Y.T
    K += (alpha) * X @ X.T

    return K

def svd_feature_select(A, n, k=1, idxs=None, sps=False, **kwargs):
    """
        Selection function which computes the CUR
        indices using the SVD Decomposition
    """

    if(idxs is None):
        ref_idx = []
    else:
        ref_idx = list(idxs)

    idxs = []

    Acopy = A.copy()

    for nn in range(n):
        if(len(ref_idx) <= n):
            if(not sps):
                (S, v, D) = np.linalg.svd(Acopy)
            else:
                (S, v, D) = svd(Acopy, k)
                D = D[np.flip(np.argsort(v))]
            pi = (D[:k]**2.0).sum(axis=0)
            pi[idxs] = 0  # eliminate possibility of selecting same column twice
            i = pi.argmax()
            idxs.append(i)
        else:
            idxs.append(ref_idx[nn])

        v = Acopy[:, idxs[-1]] / \
            np.sqrt(np.matmul(Acopy[:, idxs[-1]], Acopy[:, idxs[-1]]))

        for i in range(Acopy.shape[1]):
            Acopy[:, i] -= v * np.dot(v, Acopy[:, i])

    return list(idxs)

def pcovr_sample_select(A, n, Y, alpha, k=1, idxs=None, iterative=True, rcond = 1e-12, **kwargs):
    """
        Selection function which computes the CUR
        indices using the PCovR `Covariance` matrix
    """

    Acopy = A.copy()
    Ycopy = Y.copy()

    K = get_Kt(Acopy, Ycopy, alpha)

    if(idxs is None):
        ref_idx = []
    else:
        ref_idx = list(idxs)

    idxs = []

    try:
        for nn in tqdm(range(n)):
            if(len(ref_idx) <= nn):

                v_Kt, U_Kt = speig(K, k)
                U_Kt = U_Kt[:, np.flip(np.argsort(v_Kt))]

                pi = (np.real(U_Kt)[:, :k]**2.0).sum(axis=1)
                if(not iterative):
                    return list(reversed(np.argsort(pi)))

                pi[idxs] = 0.0
                j = pi.argmax()
            else:
                j = ref_idx[nn]

            idxs.append(j)


            if(alpha < 1):
                Ycopy -= Acopy @ (np.linalg.pinv(Acopy[idxs].T @ Acopy[idxs], rcond=rcond) @ Acopy[idxs].T) @ Ycopy[idxs]

            Ajnorm = np.dot(Acopy[j], Acopy[j])
            for i in range(Acopy.shape[0]):
                Acopy[i] -= (np.dot(Acopy[i], Acopy[j]) / Ajnorm)  * Acopy[j]

            K = get_Kt(Acopy, Ycopy, alpha)


    except (ValueError, KeyboardInterrupt):
        print("INCOMPLETE AT {}/{}".format(len(idxs), n))
        return list(idxs)

    return list(idxs)

def pcovr_feature_select(A, n, Y, alpha, k=1, idxs=None, iterative=True, rcond=1E-12, **kwargs):
    """
        Selection function which computes the CUR
        indices using the PCovR `Covariance` matrix
    """

    Acopy = A.copy()
    Ycopy = Y.copy()

    if(idxs is None):
        ref_idx = []
    else:
        ref_idx = list(idxs)

    idxs = []
    Ct = get_Ct(Acopy, Ycopy, alpha=alpha)

    try:
        for nn in tqdm(range(n)):
            if(len(ref_idx) <= nn):

                Ct = get_Ct(Acopy, Ycopy, alpha=alpha)

                v_Ct, U_Ct = speig(Ct, k)
                U_Ct = U_Ct[:, np.flip(np.argsort(v_Ct))]

                pi = (np.real(U_Ct)[:, :k]**2.0).sum(axis=1)

                if(not iterative):
                    return list(reversed(np.argsort(pi)))

                pi[idxs] = 0  # eliminate possibility of selecting same column twice
                j = pi.argmax()
                idxs.append(j)
            else:
                idxs.append(ref_idx[nn])

            v = np.linalg.pinv(
                np.matmul(Acopy[:, idxs].T, Acopy[:, idxs]), rcond=rcond)
            v = np.matmul(Acopy[:, idxs], v)
            v = np.matmul(v, Acopy[:, idxs].T)

            Ycopy -= np.matmul(v, Ycopy)

            v = Acopy[:, idxs[-1]] / \
                np.sqrt(np.matmul(Acopy[:, idxs[-1]], Acopy[:, idxs[-1]]))


            for i in range(Acopy.shape[1]):
                Acopy[:, i] -= v * np.dot(v, Acopy[:, i])


    except (ValueError, KeyboardInterrupt):
        print("INCOMPLETE AT {}/{}".format(len(idxs), n))
        return list(idxs)

    return list(idxs)

class CUR:
    """
        Super class for sampleCUR and featureCUR
        Performs CUR Decomposition on a Supplied Matrix for feature or sample seletion

        ---Arguments---
        matrix: matrix to be decomposed
        precompute: (int, tuple, Nonetype) number of columns, rows to be computed
                    upon instantiation. Defaults to None.
        pi_function: (<func> | string) Importance metric and selection for the matrix
        rcond: (float) regularization for pseudo-inverses and decompositions
        params: (dict) Dictionary of additional parameters to be passed to the
                pi function

        ---References---
        1.  G.  Imbalzano,  A.  Anelli,  D.  Giofre,  S.  Klees,  J.  Behler,
            and M. Ceriotti, J. Chem. Phys.148, 241730 (2018)
    """
    def __init__(self, matrix,
                 precompute=None,
                 pi_function='svd',
                 rcond=1E-12,
                 params={}
                 ):
        self.A = matrix

        if(isinstance(pi_function, str)):
            if(pi_function == 'pcovr'):
                self.select = self.pcovr_select
            elif(pi_function == 'svd'):
                self.select = self.svd_select
        else:
            self.select = pi_function

        self.rcond = rcond

        self.params = params

        if(pi_function.startswith('pcovr')):
            try:
                assert all([x in params for x in ['Y', 'alpha']])
            except:
                print(
                    "For column selection with PCovR, `Y` and `alpha` must be entries in `params`")

        self.idx = None

        if(precompute is not None):
            self.idx = self.compute(precompute)

    def compute(self, n):
        self.idx = self.select(self.A, n=n, rcond=self.rcond, idxs=self.idx, **self.params)
        return self.idx

class sampleCUR(CUR):

    def __init__(self, matrix,
                 precompute=None,
                 pi_function='svd',
                 rcond=1E-12,
                 params={}
                 ):
        self.pcovr_select = pcovr_sample_select
        super().__init__(matrix=matrix, precompute=precompute,
                         pi_function=pi_function,
                         rcond=1E-12,
                         params=params)

    def svd_select(A, **kwargs):
        return svd_feature_select(A.T, **kwargs)

class featureCUR(CUR):
    def __init__(self, matrix,
                 precompute=None,
                 pi_function='svd',
                 rcond=1E-12,
                 params={}
                 ):
        self.pcovr_select = pcovr_feature_select
        self.svd_select = svd_feature_select
        super().__init__(matrix=matrix, precompute=precompute,
                         pi_function=pi_function,
                         rcond=1E-12,
                         params=params)
