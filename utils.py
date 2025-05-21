
def throwlow(p):
    """
    Remove low-activity segments from probability matrix `p` by zeroing values
    that are not preceded or followed by strong speaker activity.

    Args:
        p (np.ndarray): [L, Q] matrix of speaker probabilities.

    Returns:
        np.ndarray: Smoothed probability matrix.
    """
    L, Q = p.shape
    seg = 10

    pnew = np.copy(p)

    for q in range(Q):
        for l in range(1, seg+1):
            p_segr = np.max(p[l-1:l+seg-1, q])
            if p_segr < 0.25:
                pnew[l-1, q] = 0

        for l in range(seg+1, L-seg+1):
            p_segl = np.max(p[l-seg-1:l-1, q])
            p_segr = np.max(p[l-1:l+seg-1, q])
            if p_segl < 0.25 and p_segr < 0.25:
                pnew[l-1, q] = 0

        for l in range(L-seg+1, L+1):
            p_segl = np.max(p[l-seg-1:l-1, q])
            if p_segl < 0.25:
                pnew[l-1, q] = 0

    return pnew

import numpy as np

def FeatureExtr(S, F, d):
    """
        Extract normalized cross-channel STFT ratios for each microphone pair.

        Args:
            S (np.ndarray): STFT [F, T, M].
            F (list): Frequency bin indices.
            d (int): Temporal smoothing window size.

        Returns:
            Hl (np.ndarray): [lenF*(M-1), T] reshaped features.
            Hlm (np.ndarray): [lenF, T, M-1] local features per mic.
    """
    N_frames = S.shape[1]
    Mics = S.shape[2]
    lenF = len(F)

    Hlm = np.zeros((lenF, N_frames, Mics - 1), dtype=complex)
    Hl = np.zeros((lenF * (Mics - 1), N_frames), dtype=complex)

    for m in range(2, Mics + 1):
        for l in range(1, 1 + d // 2):
            Sl1 = S[F, l, 0]
            Slm = S[F, l, m - 1]
            Hlm[:, l, m - 2] = np.sum(Slm * np.conj(Sl1)) / (np.sum(Sl1 * np.conj(Sl1)) + 1e-8)

        for l in range(1 + d // 2, N_frames - d // 2):
            Sl1 = S[F, l - d // 2:l + d // 2 + 1, 0]
            Slm = S[F, l - d // 2:l + d // 2 + 1, m - 1]
            Hlm[:, l, m - 2] = np.sum(Slm * np.conj(Sl1), axis=1) / (np.sum(Sl1 * np.conj(Sl1), axis=1) + 1e-8)
            # Hlm[:, l, m - 2] = np.log10(np.abs(np.sum(Slm * np.conj(Sl1), axis=1) + 1e-8) -
            #                              np.log10(np.abs(np.sum(Sl1 * np.conj(Sl1), axis=1) + 1e-8))

        for l in range(N_frames - d // 2, N_frames):
            Sl1 = S[F, l, 0]
            Slm = S[F, l, m - 1]
            Hlm[:, l, m - 2] = np.sum(Slm * np.conj(Sl1)) / (np.sum(Sl1 * np.conj(Sl1)) + 1e-8)

        Hl[(m - 2) * lenF:(m-1) * lenF, :] = Hlm[:, :, m - 2]
    return Hl, Hlm

def findextremeSPA(E, QQ):
    """
        Run Successive Projection Algorithm (SPA) to find QQ extreme columns in E.

        Args:
            E (np.ndarray): [N, D] matrix of points.
            QQ (int): Number of extremes to extract.

        Returns:
            ext (np.ndarray): Indices of selected extremes.
            Qcov (int): Number of extremes with norm > threshold.
    """
    Q = E.shape[1] + 1
    normE = np.sum(E ** 2, axis=1)
    ext = np.zeros(QQ, dtype=int)

    maxq = np.zeros(Q)
    maxq[0], ext[0] = np.max(normE), np.argmax(normE)
    Pq = np.eye(Q - 1)

    for q in range(1, QQ):
        aq = E[ext[q - 1], :].reshape(-1, 1)
        Pq = np.dot(np.eye(Q - 1) - np.dot(Pq, aq) * np.dot(Pq, aq).T / np.linalg.norm(np.dot(Pq, aq)) ** 2, Pq)
        normP = np.sum((np.dot(E, Pq) ** 2), axis=1)
        maxq[q], ext[q] = np.max(normP), np.argmax(normP)

    Qcov = np.where(maxq < 5)[0][0] if np.any(maxq < 5) else QQ
    return ext, Qcov    



def smoother(C):
    L, Q = C.shape
    seg = 10
    Cnew = np.copy(C)

    for q in range(Q):
        for l in range(0, seg):
            S_segr = np.sum(Cnew[l:l+seg, q])
            if S_segr > 1:
                Cnew[l, q] = 1

        for l in range(seg+1, L-seg+1):
            S_segl = np.sum(C[l-seg-1:l, q])
            S_segr = np.sum(C[l+1:l+seg-1, q])
            if S_segl == 0 and S_segr == 0:
                Cnew[l, q] = 0
            elif S_segl > 0 and S_segr > 0:
                 Cnew[l, q] = 1     

        for l in range(L-seg, L):
            S_segl = np.sum(C[l-seg-1:l, q])
            if S_segl > 1:
                Cnew[l, q] = 1

    return Cnew



