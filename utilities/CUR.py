import numpy as np
from scipy.sparse.linalg import svds as svd
from scipy.sparse.linalg import eigs as speig
import scipy

from tqdm import tqdm as tqdm


def sorted_eig(mat, thresh=0.0, n=None, sps=True):
    """
        Returns n eigenvalues and vectors sorted
        from largest to smallest eigenvalue
    """
    if(sps):
        from scipy.sparse.linalg import eigs as speig
        if(n is None):
            n = mat.shape[0] - 1
        val, vec = speig(mat, k=n, tol=thresh)
        val = np.real(val)
        vec = np.real(vec)

        idx = sorted(range(len(val)), key=lambda i: -val[i])
        val = val[idx]
        vec = vec[:, idx]

    else:
        val, vec = np.linalg.eigh(mat)
        val = np.flip(val, axis=0)
        vec = np.flip(vec, axis=1)

    vec[:, val < thresh] = 0
    val[val < thresh] = 0

    return val[:n], vec[:, :n]



def compute_P(A_c, S, A_r, thresh=1e-12):
    """ Computes the latent-space projector for the feature matrix """
    SA = np.matmul(S, A_r)
    SA = np.matmul(SA, SA.T)

    v_SA, U_SA = np.linalg.eigh(SA)
    v_SA[v_SA < thresh] = 0

    return np.matmul(U_SA, np.diagflat(np.sqrt(v_SA)))


def get_Ct(X, Y, alpha=0.5, regularization=1e-6):
    """
        Creates the PCovR modified covariance
        ~C = (alpha) * X^T X +
             (1-alpha) * (X^T X)^(-1/2) ~Y ~Y^T (X^T X)^(-1/2)

        where ~Y is the properties obtained by linear regression.
    """

    cov = X.T @ X

    if(alpha < 1.0):
        # changing these next two lines can cause a LARGE error
        Cinv = np.linalg.pinv(cov, hermitian=True)
        try:
            Csqrt = scipy.linalg.sqrtm(cov)
        except:
            v, U = sorted_eig(cov)
            Csqrt = U @ np.diagflat(np.sqrt(v)) @ U.T

        # parentheses speed up calculation greatly
        Y_hat = Csqrt @ Cinv @ (X.T @ Y)

        if(len(Y_hat.shape) < 2):
            Y_hat = Y_hat.reshape((-1, 1))

        C_lr = Y_hat @ Y_hat.T

    else:
        C_lr = np.zeros(cov.shape)

    C_pca = cov
    C = alpha*C_pca + (1.0-alpha)*C_lr

    return C


def svd_select(A, n, k=1, idxs=None, sps=False, **kwargs):
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


def pcovr_sample_select(A, n, Y, alpha, k=1, idxs=None, sps=False, thresh = 1e-8, **kwargs):
    """
        Selection function which computes the CUR
        indices using the PCovR `Covariance` matrix
    """

    Acopy = A.T.copy()
    Ycopy = Y.copy()

    K = alpha * Acopy @ Acopy.T + (1-alpha) * Ycopy @ Ycopy.T

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

                pi[idxs] = 0.0
                j = pi.argmax()
            else:
                j = ref_idx[nn]

            idxs.append(j)


            Ycopy -= Acopy @ (np.linalg.pinv(Acopy[idxs].T @ Acopy[idxs]) @ Acopy[idxs].T) @ Ycopy[idxs]

            if(np.sqrt(np.matmul(Acopy[idxs[-1]], Acopy[idxs[-1]])) > thresh): 
                v = Acopy[idxs[-1]] / \
                    np.sqrt(np.matmul(Acopy[idxs[-1]], Acopy[idxs[-1]]))


                for i in range(Acopy.shape[0]):
                    Acopy[i] -= v * np.dot(v, Acopy[i])

                K = alpha * Acopy@Acopy.T + (1-alpha)*Ycopy @ Ycopy.T
            else:
                return list(idxs)

    except (ValueError, KeyboardInterrupt):
        print("INCOMPLETE AT {}/{}".format(len(idxs), n))
        return list(idxs)

    return list(idxs)


def pcovr_feature_select(A, n, Y, alpha, k=1, idxs=None, sps=False, **kwargs):
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

    try:
        for nn in tqdm(range(n)):
            if(len(ref_idx) <= nn):

                Ct = get_Ct(Acopy, Ycopy, alpha=alpha)

                if(not sps):
                    v_Ct, U_Ct = sorted_eig(Ct, n=k)
                else:
                    v_Ct, U_Ct = speig(Ct, k)
                    U_Ct = U_Ct[:, np.flip(np.argsort(v_Ct))]

                pi = (np.real(U_Ct)[:, :k]**2.0).sum(axis=1)
                pi[idxs] = 0  # eliminate possibility of selecting same column twice
                j = pi.argmax()
                idxs.append(j)
            else:
                idxs.append(ref_idx[nn])

            v = np.linalg.pinv(
                np.matmul(Acopy[:, idxs].T, Acopy[:, idxs]))
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
selections = dict(svd=svd_select, pcovr=pcovr_feature_select,
                  pcovr_features=pcovr_feature_select,
                  pcovr_samples=pcovr_sample_select,
            )


class CUR:
    """
        Performs CUR Decomposition on a Supplied Matrix

        ---Arguments---
        matrix: matrix to be decomposed
        precompute: (int, tuple, Nonetype) number of columns, rows to be computed
                    upon instantiation. Defaults to None.
        select: (None, "feature", "sample")
        pi_function: (<func>) Importance metric and selection for the matrix
        symmetry_tolerance: (float) Tolerance by which a matrix is symmetric
        params: (dict) Dictionary of additional parameters to be passed to the
                pi function

        ---References---
        1.  G.  Imbalzano,  A.  Anelli,  D.  Giofre,  S.  Klees,  J.  Behler,
            and M. Ceriotti, J. Chem. Phys.148, 241730 (2018)
    """

    def __init__(self, matrix,
                 precompute=None,
                 select=None,
                 feature_select=False,
                 pi_function='svd',
                 symmetry_tolerance=1e-4,
                 params={}
                 ):
        self.A = matrix
        self.symmetric = self.A.shape == self.A.T.shape and \
            np.all(np.abs(self.A-self.A.T)) < symmetry_tolerance

        assert select in ['feature','sample', None]
        self.fs = (select=='feature') or feature_select
        self.ss = (select=='sample')

        print(self.fs, self.ss)
        assert not (self.fs and self.ss)

        if(isinstance(pi_function, str)):
            if(pi_function == 'pcovr'):
                if(self.fs):
                    self.select = selections['pcovr_features']
                else:
                    self.select = selections['pcovr_samples']
            else:
                self.select = selections.get(pi_function, None)
        else:
            self.select = pi_function
        self.params = params

        if(pi_function.startswith('pcovr')):
            try:
                assert all([x in params for x in ['Y', 'alpha']])
            except:
                print(
                    "For column selection with PCovR, `Y` and `alpha` must be entries in `params`")

        self.idx_c, self.idx_r = None, None
        if(precompute is not None):
            if(isinstance(precompute, int)):
                self.idx_c, self.idx_r = self.compute_idx(
                    precompute, precompute)
            else:
                self.idx_c, self.idx_r = self.compute_idx(*precompute)

    def compute_idx_r(self, n_r):
        if(not self.fs):
            if(not self.symmetric):
                idx_r = self.select(self.A.T, n_r, idxs=self.idx_r, **self.params)
            else:
                idx_r = compute_idx_c(n_r)
        else:
            idx_r = np.asarray(range(self.A.shape[0]))[:n_r]
        return idx_r

    def compute_idx_c(self, n_c):
        if(not self.ss):
            idx_c = self.select(self.A, n_c, idxs=self.idx_c, **self.params)
        else:
            idx_c = np.asarray(range(self.A.shape[1]))[:n_c]
        return idx_c

    def compute_idx(self, n_c, n_r):
        return self.compute_idx_c(n_c), self.compute_idx_r(n_r)

    def compute(self, n_c=None, n_r=None):
        """
           Compute the n_c selected columns and n_r selected rows
        """
        if(self.fs):
            n_r = self.A.shape[0]
        elif(self.symmetric and n_r is None):
            n_r = n_c
        elif(n_r is None):
            print("You must specify a n_r for non-symmetric matrices.")

        if(self.ss):
            n_c = self.A.shape[1]
        elif(self.symmetric and n_c is None):
            n_c = n_r
        elif(n_c is None):
            print("You must specify a n_r for non-symmetric matrices.")

        print(n_c)

        if(self.idx_c is None or len(self.idx_c) < n_c):
            idx_c = self.compute_idx_c(n_c)
            self.idx_c = idx_c
        else:
            idx_c = self.idx_c[:n_c]
        if(self.idx_r is None or len(self.idx_r) < n_r):
            idx_r = self.compute_idx_r(n_r)
            self.idx_r = idx_r
        else:
            idx_r = self.idx_r[:n_r]
        # The CUR Algorithm
        A_c = self.A[:, idx_c]
        A_r = self.A[idx_r, :]

        # Compute S.
        S = np.matmul(np.matmul(np.linalg.pinv(A_c), self.A),
                      np.linalg.pinv(A_r))
        return A_c, S, A_r

    def compute_P(self, n_c):
        """
           Computes the projector into latent-space for ML models
        """

        A_c, S, A_r = self.compute(n_c)
        self.P = compute_P(A_c, S, A_r)

    def loss(self, n_c, n_r=None):
        """
            Error between approximated matrix and target
        """
        A_c, S, A_r = self.compute(n_c, n_r)
        return np.linalg.norm(self.A - np.matmul(A_c, np.matmul(S, A_r)))/np.linalg.norm(self.A)
