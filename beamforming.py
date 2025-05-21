import numpy as np
from scipy.signal import istft, get_window, resample_poly, resample, decimate, firwin, filtfilt
from scipy.linalg import eigh, solve
from mir_eval.separation import bss_eval_sources
import CFG
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as t_audio_SI_SDR
from torchmetrics.functional.audio import signal_distortion_ratio as t_audio_SDR
import torch
import itertools
import matplotlib.pyplot as plt


def beamformer_C(Xt, mask, xq, Q, fh, olap, lens, a, b, apply_mask, fs, att, C):
    N_frames = Xt.shape[1]
    NFFT = 2 * (Xt.shape[0] - 1)

    H, R, _ = RTFsmaskandNoiseCovEst2(Xt, Xt, mask, Q, fh)
    y, Y = lcmv_noise(Xt, H, R, olap, lens, a, b, fs)

    leni = int((N_frames - 1) * (1-olap) * NFFT)
    if apply_mask:
        ym = np.zeros_like(y)
        for q in range(Q):
            Ym = Y[:, :, q] * ((mask == q) + (mask != q) * att)
            ym[:, q] = istft(Ym, nperseg=NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1][:ym[:, q].shape[0]]
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
        yo[:, q] = istft(Yo[:, :, q], nperseg=NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1][:yo[:, q].shape[0]]
    
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

        # Regularize covariance matrix
    for i in range(Mics):
        Rn[i, i, :] += 1e-6

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

def compute_R_xt(Xt):
    """
    Xt: F × T × M (STFT of mixture)
    Returns: R (M × M × F) spatial covariance
    """
    F, T, M = Xt.shape
    R = np.zeros((M, M, F), dtype=np.complex128)
    for f in range(F):
        Xf = Xt[f, :, :]  # shape: T × M
        R[:, :, f] = Xf.conj().T @ Xf / T  # (M × T) @ (T × M) → (M × M)
        R[:, :, f] += 1e-6 * np.eye(M)     # regularization
    return R

def beamformer(Xt, mask, xq, Q, fh, olap, lens, a, apply_mask, fs, att=0.3, Xq=None, Hq=None, Hl=None, C=None,
               b=0.01):
    """
        Perform LCMV beamforming to separate sources from multichannel input.

        Args:
            Xt (np.ndarray): [F, T, M] STFT of the mixture.
            mask (np.ndarray): [F, T] time-frequency mask (e.g., from NN or oracle).
            xq (np.ndarray): [T, M, Q] ground truth time-domain sources.
            Q (int): Number of sources.
            fh (list): Top time-frame indices per source (used for RTF estimation).
            olap (float): STFT overlap fraction.
            lens (int): Length of output waveform.
            a (float): LCMV regularization parameter.
            apply_mask (bool): Whether to apply TF masks after beamforming.
            fs (int): Sampling rate.
            att (float): Attenuation factor for non-target bins when masking.
            Xq, Hq, Hl, C, b: Optional overrides for beamforming.

        Returns:
            Tuple[float, float, np.ndarray]: SDR, SI-SDR, separated waveform [T, Q]
        """
    N_frames = Xt.shape[1]
    NFFT = 2 * (Xt.shape[0] - 1)
    if Hq is not None:
        H = Hq
    elif apply_mask:
        if CFG.add_noise:
            H, R, _ = RTFsmaskandNoiseCovEst2(Xt, Xt, mask, Q, fh)
        else:
            H = RTFsmaskEst2(Xt, mask, Q, fh)
            R = compute_R_xt(Xt)

    else:
        H = RTFsmaskEst2(Xq, None, Q, fh)

    #
    if CFG.add_noise:
        y, Y = lcmv_noise(Xt, H, R, olap, lens, a, b, fs)
    else:
        y, Y = lcmv_nonoise(Xt, H, olap, lens, a, fs)
    # y, Y = lcmv_try(Xt, H, olap, lens, fs)

    if apply_mask:
        ym = np.zeros_like(y)

        for q in range(Q):
            if C is None:
                Ym = Y[:, :, q] * ((mask == q) + (mask != q) * att)
                ym[:, q] = istft(Ym, nperseg=NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1][:ym[:, q].shape[0]]
            else:
                Ym = Y[:, :, q] * ((mask == q) + (mask != q) * att)
                onlyq = np.logical_and(C[:, q], (np.sum(C[:, :q], axis=1) + np.sum(C[:, q + 1:Q], axis=1)) == 0)
                noq = C[:, q] == 0
                Ym[:, onlyq] = Xt[:, onlyq, 0]
                Ym[:, noq] = 0
                ym[:, q] = istft(Ym, nperseg=NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1][:ym[:, q].shape[0]]
    else:
        ym = y


    if CFG.resample_flag:
        ym = decimate(ym, 1, axis=0, zero_phase=True)
        ym = resample(ym, int(CFG.lens / CFG.sample_factor), axis=0)

    if (ym.sum(axis=0) == 0).any():
        ym[:, np.where(ym.sum(axis=0) == 0)[0]] += 1e-8

    permutations = list(itertools.permutations(range(Q)))

    max_sisdr = float('-inf')
    ym_torch = torch.from_numpy(np.real(ym)).T
    y_torch = torch.from_numpy(np.real(xq[:, 0, :])).T
    for perm in permutations:
        perm_speakers_ym = ym_torch[perm, :]
        sisdr = t_audio_SI_SDR(perm_speakers_ym, y_torch).mean().item()

        if sisdr > max_sisdr:
            max_sisdr = sisdr
            best_y = perm_speakers_ym
            best_perm = perm

    SDR = t_audio_SDR(best_y, y_torch).mean().item()
    sisdr = max_sisdr

    return SDR, sisdr, ym[:, best_perm]


def lcmv_nonoise(Xt, H, olap, lens, a, fs):
    NFFT = 2 * (Xt.shape[0] - 1)
    L = Xt.shape[1]
    Mics = Xt.shape[2]
    Q = H.shape[2]

    Yo = np.zeros((NFFT // 2 + 1, L, Q), dtype=np.complex128)

    for q in range(Q):
        W = np.zeros((Mics, NFFT // 2 + 1), dtype=np.complex128)
        g = np.zeros(Q)
        g[q] = 1

        for k in range(25, NFFT // 2 + 1):
            C = np.squeeze(H[k, :, :])
            if np.sum(C) == 0:
                W[:, k] = np.zeros(Mics)
            else:
                temp1 = np.dot(C.conj().T, C)  # C' * C
                temp2 = a * np.linalg.norm(temp1) * np.eye(temp1.shape[0])  # a * norm(C' * C) * eye(Q)
                temp3 = np.linalg.pinv(temp1 + temp2)  # inverse of (C' * C + a * norm(C' * C) * eye(Q))
                W[:, k] = np.dot(C, temp3).dot(g)

            Yo[k, :, q] = np.dot(Xt[k, :, :], W[:, k].conj())

    yo = np.zeros((lens, Q), dtype=np.complex128)
    leni = (L - 1) * olap * NFFT + NFFT
    for q in range(Q):
        yo[:, q] = istft(Yo[:, :, q], nperseg=NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1][:yo.shape[0]]

    return yo, Yo


def RTFsmaskEst2(X, mask, Q, fk):
    NFFT = (X.shape[0] - 1) * 2
    Mics = X.shape[2]

    # Initialize the RTF matrix for speakers
    H = np.zeros((NFFT // 2 + 1, Mics, Q), dtype=np.complex128)

    # Loop through each speaker to compute the RTFs
    for q in range(Q):
        for k in range(1, NFFT // 2 + 2):
            if mask is not None:
                fkq = np.where(mask[k - 1, :] == q)[0]
                Nfkq = len(fkq)

                # If no bins are assigned to the speaker, fallback to fk[q]
                if Nfkq == 0:
                    fkq = fk[q]
                    Nfkq = len(fkq)

                # Estimate the covariance matrix for the speaker

                Rxk = np.dot(np.transpose(X[k - 1, fkq, :]), np.conj(X[k - 1, fkq, :])) / Nfkq

            else:
                # Rxk = np.cov(X[k - 1, :, :, q].T)
                Rxk = np.dot(X[k - 1, :, :, q].T, np.conj(X[k - 1, :, :, q])) / X.shape[1]
            # Perform eigenvalue decomposition to get the dominant eigenvector
            ee, Uk = eigh(Rxk)
            ee_ord = np.argsort(ee)[::-1]

            # Use the eigenvector corresponding to the largest eigenvalue as the RTF estimate
            Hk = Uk[:, ee_ord[0]]
            H[k - 1, :, q] = Hk / Hk[0]  # Normalize by the first microphone

    return H


def MVDR_over_speakers(mixture_stft, target_stft, target_signal, fs, olap, Q=CFG.Q, ref_mic=0, give_target=False):

    y = np.zeros((CFG.lens, Q), dtype=np.complex128)
    for q in range(Q):
        if give_target:
            y[:, q] = MVDR(mixture_stft, None, target_stft[:, :, :, q], fs, olap)
        else:
            noise = mixture_stft - target_stft[:, :, :, q]
            y[:, q] = MVDR(mixture_stft, noise, None, fs, olap)

    SDR = t_audio_SDR(torch.from_numpy(np.real(y)).T, torch.from_numpy(np.squeeze(target_signal[:, 0, :]).T)).mean().item()
    sisdr = t_audio_SI_SDR(torch.from_numpy(np.real(y)).T, torch.from_numpy(np.squeeze(target_signal[:, 0, :]).T)).mean().item()

    return SDR, sisdr, y

def MVDR(mixture_stft, noise_spec, target_stft, fs, olap, ref_mic=0):
    N_frames = mixture_stft.shape[0]
    NFFT = 2 * (mixture_stft.shape[1] - 1)

    # estimate steering vector for desired speaker (depending if target is available)
    if target_stft is not None:
        h = estimate_steering_vector(target_stft=target_stft)
    else:

        h = estimate_steering_vector(mixture_stft=mixture_stft, noise_stft=noise_spec)

    # calculate weights
    if target_stft is not None:
        w = mvdr_weights(target_stft, h)
        w = w / np.linalg.norm(w, axis=1, keepdims=True) + 1e-6
        sep_spec = apply_beamforming_weights(target_stft, w)
    else:
        w = mvdr_weights(mixture_stft, h)

        # apply weights
        sep_spec = apply_beamforming_weights(mixture_stft, w)

    # reconstruct wav
    recon = istft(sep_spec, nperseg=NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[0]

    return recon

def estimate_steering_vector(target_stft=None, mixture_stft=None, noise_stft=None):
    if target_stft is None:
        if mixture_stft is None or noise_stft is None:
            raise ValueError("If no target recordings are provided, both mixture and noise are needed.")

        C, F, T = mixture_stft.shape  # (mics, freq, time)
    else:
        C, F, T = target_stft.shape  # (mics, freq, time)

    eigen_vec, eigen_val, h = [], [], []

    for f in range(F):
        if target_stft is None:
            # _R0 = mixture_stft[:, f] @ np.conj(mixture_stft[:, f].T)
            _R0 = np.cov(mixture_stft[:, f])
            # _R1 = noise_stft[:, f] @ np.conj(noise_stft[:, f].T)
            _R1 = np.cov(noise_stft[:, f])
            _Rxx = _R0 - _R1
            # _Rxx += np.eye(C) * 1e-6  # Regularization

        else:
            # _Rxx = target_stft[:, f] @ np.conj(target_stft[:, f].T)
            _Rxx = np.cov(target_stft[:, f, :])

            _Rxx += np.eye(C) * 1e-6  # Regularization

            # _d, _v = np.linalg.eig(_Rxx)
            _d, _v = np.linalg.eigh(_Rxx)

        _d, _v = np.linalg.eigh(_Rxx)
        idx = np.argmax(_d.real)
        eigen_vec.append(_v[:, idx])
        eigen_val.append(_d[idx])

    for vec, val in zip(eigen_vec, eigen_val):
        if val != 0.0:
            # the part is modified from the MVDR implementation https://github.com/Enny1991/beamformers
            # vec = vec * val / np.abs(val)
            vec = vec / vec[0]  # normalized to the first channel
            h.append(vec)
        else:
            h.append(np.ones_like(vec))

    return np.vstack(h)

def apply_beamforming_weights(signals, weights):
    return np.einsum('fa, fbt -> at', np.conj(weights.T), signals)


def mvdr_weights(mixture_stft, h):
    C, F, T = mixture_stft.shape  # (mics, freq, time)
    W = np.zeros((F, C), dtype='complex128')
    for f in range(F):
        mult = np.linalg.pinv(np.cov(mixture_stft[:, f])) @ h[f]
        W[f, :] = mult / (h[f].conj().T @ mult)
    return W
