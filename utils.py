# import numpy as np

def throwlow(p):
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

def MaskErr(Tmask, Emask, Q):
    MDq = np.zeros(Q)
    FAq = np.zeros(Q)


    N_frames = Tmask.shape[1]
    NFFT = Tmask.shape[0]
    non_noise_num = np.sum(Tmask != Q)
    for q in range(Q):
        MDq[q] = np.sum(np.sum((Tmask == q) & (Emask != q))) / non_noise_num
        FAq[q] = np.sum(np.sum((Tmask != q) & (Emask == q))) / non_noise_num

    MD = np.mean(MDq)
    FA = np.mean(FAq)
    Acc = np.sum((Tmask == Emask) & (Tmask != Q) & (Emask != Q)) / non_noise_num
    Err = 1 - Acc
    return MD, FA, Err

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


def VCA(Y, J, seed, snr_input=0):
    snr_input = 0

    L, F = Y.shape
    generator = np.random.default_rng(seed=15)
    if snr_input == 0:
        y_m = np.mean(Y, axis=1, keepdims=True)
        Y_o = Y - y_m
        YYt_svd = np.linalg.svd(np.dot(Y_o, Y_o.T) / float(L))
        Ud = YYt_svd[0][:, :J]
        x_p = np.dot(Ud.T, Y_o)

        P_y = np.sum(Y ** 2) / float(L)
        P_x = np.sum(x_p ** 2) / float(L) + np.sum(y_m ** 2)
        SNR = 10 * np.log10((P_x - J) / L * P_y) / (P_y - P_x)
    else:
        SNR = snr_input
    SNR_th = 15 + 10 * np.log10(J)

    if SNR < SNR_th:
        d = J - 1
        if snr_input == 0:
            Ud = Ud[:, :d]
        else:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m
            YYt_svd = np.linalg.svd(np.dot(Y_o, Y_o.T) / float(L))
            Ud = YYt_svd[0][:, :J]
            x_p = np.dot(Ud.T, Y_o)
        Yp = np.dot(Ud, x_p[:d, :]) + y_m

        x = x_p[:d, :]
        c = np.amax(np.sum(x ** 2, axis=0)) ** 0.5
        y = np.vstack((x, c * np.ones((1, L))))
    else:
        d = J
        YYt = np.linalg.svd(np.dot(Y, Y.T) / float(L))
        Ud = YYt[0][:, :d]
        x_p = np.dot(Ud.T, Y)
        Yp = np.dot(Ud, x_p[:d, :])
        x = np.dot(Ud.T, Y)
        u = np.mean(x, axis=1, keepdims=True)
        y = x / np.dot(u.T, x)
    y[np.isnan(y)] = 0
    indices = np.zeros((J), dtype=int)
    A = np.zeros((J, J))
    A[-1, 0] = 1

    for i in range(J):
        w = generator.random(size=(J, 1))
        f = w - np.dot(A, np.dot(np.linalg.pinv(A), w))
        f = f / np.linalg.norm(f)

        v = np.dot(f.T, y)

        indices[i] = np.argmax(np.absolute(v))
        A[:, i] = y[:, indices[i]]  # same as x(:,indice(i))

    E = Yp[:, indices]
    a = 5



