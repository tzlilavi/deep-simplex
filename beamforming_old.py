import numpy as np
from scipy.signal import istft
from scipy.linalg import eigh
from mir_eval.separation import bss_eval_sources

def beamformer(Xt, mask, xq, Q, fh, olap, lens, a, b, apply_mask, fs, att=0.3):
    N_frames = Xt.shape[1]
    NFFT = 2 * (Xt.shape[0] - 1)

    H, R, _ = RTFsmaskandNoiseCovEst2(Xt, Xt, mask, Q, fh)
    y, Y = lcmv_noise(Xt, H, R, olap, lens, a, b, fs)

    leni = int((N_frames - 1) * (1-olap) * NFFT)
    if apply_mask:
        ym = np.zeros_like(y)
        for q in range(Q):
            #Ym = Y[:, :, q] * ((mask == q) + (mask != q) * np.vstack([0.3 * np.ones(NFFT // 4), 0.3 * np.ones(NFFT // 4 + 1)] * N_frames))
            Ym = Y[:, :, q] * ((mask == q) + (mask != q) * att)
            ym[:, q] = istft(Ym, nperseg=NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1][:leni]
    else:
        ym = y

    SDRq, SIRq, _, _ = bss_eval_sources(np.real(ym).T, np.squeeze(xq[:, 0, :]).T)
    SDR = np.mean(SDRq)
    SIR = np.mean(SIRq)

    return SDR, SIR, ym




def beamformer_C(Xt, mask, xq, Q, fh, olap, lens, a, b, apply_mask, fs, att, C):
    N_frames = Xt.shape[1]
    NFFT = 2 * (Xt.shape[0] - 1)

    H, R, _ = RTFsmaskandNoiseCovEst2(Xt, Xt, mask, Q, fh)
    y, Y = lcmv_noise(Xt, H, R, olap, lens, a, b, fs)

    leni = int((N_frames - 1) * (1-olap) * NFFT)
    if apply_mask:
        ym = np.zeros_like(y)
        for q in range(Q):
            #Ym = Y[:, :, q] * ((mask == q) + (mask != q) * np.vstack([0.3 * np.ones(NFFT // 4), 0.3 * np.ones(NFFT // 4 + 1)] * N_frames))
            Ym = Y[:, :, q] * ((mask == q) + (mask != q) * att)
            ym[:, q] = istft(Ym, nperseg=NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1][:leni]
    else:
        ym = y

    Yf  = Ym  
    yf = np.zeros_like(y)
    for q in range(Q):
        Yf  = Y[:, :, q] * ((mask == q) + (mask != q) * att)
        onlyq  = np.logical_and(C[:,q]==1,(np.sum(C[:,:q],axis=1)+np.sum(C[:,q+1:Q],axis=1))==0)
        noq = C[:,q]==0
        Yf[:,onlyq] = Xt[:,onlyq,0] 
        Yf[:,noq] = 0
        yf[:, q] = istft(Yf, nperseg=NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1][:leni]



    SDRq, SIRq, _, _ = bss_eval_sources(np.real(yf).T, np.squeeze(xq[:, 0, :]).T)
    SDR = np.mean(SDRq)
    SIR = np.mean(SIRq)

    return SDR, SIR, yf    

def lcmv_noise(Xt, H, R, olap, lens, a, b, fs):
    NFFT = 2 * (Xt.shape[0] - 1)
    L = Xt.shape[1]
    Mics = Xt.shape[2]
    Q = H.shape[2]
    
    Yo = np.zeros((NFFT // 2 + 1, L, Q), dtype=np.complex128)
    
    for q in range(Q):
        W = np.zeros((Mics, NFFT // 2 + 1), dtype=np.complex128)
        C = np.zeros((Mics, Q), dtype=np.complex128)
        g = np.zeros(Q)
        g[q] = 1
        
        for k in range(25, NFFT // 2 + 1):
            C = np.squeeze(H[k, :, :])
            invR = np.linalg.inv(R[:, :, k] + b * np.linalg.norm(R[:, :, k]) * np.eye(Mics))
            
            if np.sum(C) == 0:
                W[:, k] = np.zeros(Mics)
            elif Q>1:
                temp1 = np.dot(C.conj().T, np.dot(invR, C))  # C' * invR * C
                temp2 = a * np.linalg.norm(temp1) * np.eye(temp1.shape[0])  # a * norm(C' * invR * C) * eye(Q)
                temp3 = np.linalg.inv(temp1 + temp2)  # inverse of (C' * invR * C + a * norm(C' * invR * C) * eye(Q))
                W[:, k] = np.dot(np.dot(np.dot(invR, C), temp3), g)
            else:
                W[:, k] = np.dot(invR, C)/np.dot(C.conj().T, np.dot(invR, C))

            Yo[k, :, q] = np.dot(Xt[k, :, :], W[:, k].conj())
    
    yo = np.zeros((lens, Q), dtype=np.complex128)
    leni = (L - 1) * olap * NFFT + NFFT
    for q in range(Q):
        yo[:, q] = istft(Yo[:, :, q], nperseg=NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1]
    
    return yo, Yo

def RTFsmaskandNoiseCovEst2(X, N, mask, Q, fk):
    NFFT = (X.shape[0] - 1) * 2
    Mics = X.shape[2]
    
    Rn = np.zeros((Mics, Mics, NFFT // 2 + 1), dtype=np.complex128)
    for k in range(1, NFFT // 2 + 2):
        fkq = np.where(mask[k - 1, :] == Q)[0]
        Nfkq = len(fkq)
        if Nfkq == 0:
            fkq = fk[Q]
            Nfkq = len(fkq)    
        Rn[:, :, k - 1] = np.dot(np.transpose(N[k - 1, fkq, :]), np.conj(N[k - 1, fkq, :])) / Nfkq

    H = np.zeros((NFFT // 2 + 1, Mics, Q), dtype=np.complex128)
    Hu = np.zeros((NFFT // 2 + 1, Mics, Q), dtype=np.complex128)

    for q in range(1, Q + 1):
        for k in range(1, NFFT // 2 + 2):
            fkq = np.where(mask[k - 1, :] == q-1)[0]
            Nfkq = len(fkq)
            if Nfkq < Q:
                fkq = fk[q]
                Nfkq = len(fkq)
            Rxk = np.dot(np.transpose(X[k - 1, fkq, :]), np.conj(X[k - 1, fkq, :])) / Nfkq
            ee, Uk = eigh(Rxk, Rn[:, :, k - 1])
            ee_sort = np.sort(ee)[::-1]
            ee_ord = np.argsort(ee)[::-1]
            Hk = np.dot(Rn[:, :, k - 1], Uk[:, ee_ord[0]])
            H[k - 1, :, q - 1] = Hk / Hk[0]
            Hu[k - 1, :, q - 1] = Hk

    return H, Rn, Hu
