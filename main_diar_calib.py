from beamforming import beamformer, beamformer_C
from utils import throwlow, findextremeSPA, FeatureExtr, MaskErr, smoother
from conformal_risk_contorl import get_lhat


import os
import json
import glob
import numpy as np
from scipy.signal import decimate, stft
from scipy.io import wavfile
import scipy.signal as signal
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from tqdm import trange

# Set seed for reproducibility
seed0 = 88#45#32#66#78
np.random.seed(seed0)

# Paths
db_base_path0 = '/Users/blaufer/Documents/TAU/research/seperation/Elisheva/DB/Seperate'
db_alg_path = '/Users/blaufer/Documents/TAU/research/seperation/Enhancment'

# Parameters
seconds = 24
fs_h = 48000
lens_h = seconds * fs_h
fs = 16000
spk_olap = 3*fs
lens = seconds * fs
sile = int(0.5 * fs)
sile_h = int(0.5 * fs_h)
NFFT = 2 ** 11
len_win = 2 ** 11
olap = 0.75
N_frames = int(lens/ ((1-olap) * NFFT) + 1)
revs = ['RT60_Low']  # ,'RT60_High'};
SNRs = [30]  # [0,5,10,15,20];
Noise_type = 'Air_conditioner'  # 'Babble_noise';
wav_folder = 'wav_examples'

F0 = np.arange(NFFT // 2 + 1)
lenF0 = len(F0)

f1 = 1000
f2 = 2000
sv = 343
k = np.arange(0, NFFT // 2 + 1)
Q = 2
Iter = 10  # 50

mics = np.arange(24, 31)  # [7:8 10:11 13:14 16:17 19:20 22:23];%1:6;%[7:8 10:11 13:14 16:17 19:20 22:23];
M = len(mics)

all_speakers = 'ABCDEFHIJKLMNOPQRST'
speakers = 'ABCDEFHIJ'  
chairs = np.arange(1, 7)

highpass = signal.firwin(199, 150, fs=fs, pass_zero="highpass")

# Generate test positions
MD = np.zeros((Iter, len(SNRs), len(revs)))
FA = np.zeros((Iter, len(SNRs), len(revs)))

SDR = np.zeros((Iter, len(SNRs), len(revs)))
SIR = np.zeros((Iter, len(SNRs), len(revs)))

SDRC = np.zeros((Iter, len(SNRs), len(revs)))
SIRC = np.zeros((Iter, len(SNRs), len(revs)))

SDRi = np.zeros((Iter, len(SNRs), len(revs)))
SIRi = np.zeros((Iter, len(SNRs), len(revs)))

SDRiva = np.zeros((Iter, len(SNRs), len(revs)))
SIRiva = np.zeros((Iter, len(SNRs), len(revs)))

SDRp = np.zeros((Iter, len(SNRs), len(revs)))
SIRp = np.zeros((Iter, len(SNRs), len(revs)))

SNRin = np.zeros((Iter, len(revs)))

den = np.zeros((5, Iter, len(SNRs), len(revs)))
sumin = np.zeros((5, Iter, len(SNRs), len(revs)))

spk = np.empty((Iter, Q), dtype=int)
chr = np.empty((Iter, Q), dtype=int)

loss_list = [ ]

for ii in range(Iter):
    rand_spk = np.random.permutation(len(speakers))[:Q]
    spk[ii, :] = rand_spk 
    rand_chr = np.random.permutation(len(chairs))[:Q]
    chr[ii, :] = rand_chr 


# Loop over reverberation conditions
for rr, rev in enumerate(revs):
    db_base_path = os.path.join(db_base_path0, rev)
    
    # Loop over iterations
    for ii in range(Iter):
        # Load Signals
        xq = np.zeros((lens, M, Q))
        for q in range(Q):
            db_speaker_path = os.path.join(db_base_path, speakers[spk[ii, q]])
            dp_speaker_chair = os.path.join(db_speaker_path, f'*Chair_{chairs[chr[ii, q]]}*.wav')
            dir_spk = sorted(glob.glob(dp_speaker_chair))[0]
            spkdir = os.path.join(db_speaker_path, dir_spk)
            fs_sk, sk = wavfile.read(spkdir)
            info = {'name': speakers[spk[ii, q]], 'chair': str(chairs[chr[ii, q]]), 'first_sentence_number': dir_spk.split('S_')[1].split('_')[0]}
            randint = 0*fs_h #if q == 0 else 5*fs_h#np.random.randint(0, 3 * fs_h)
            deci = decimate(sk[randint:(randint + (lens_h - sile_h)//2), mics], 3, axis=0)
            if q==0:
                xq[sile:sile+len(deci), :, q] = deci
            else:
                xq[sile+len(deci)-spk_olap:sile+2*len(deci)-spk_olap, :, q] = deci

        # Apply high-pass to reduce noise
        xqf = np.zeros((lens, M, Q))
        for q in range(Q):
            for m in range(M):
                xqf[:, m, q] = signal.filtfilt(highpass, 1, xq[:, m, q])
                
        x = np.sum(xqf, axis=2)

        # Load Noise
        db_noise_path = os.path.join(db_base_path0, rev, f'Noises/In/{Noise_type}_in_{rev}.wav')
        fs_nk, nk = wavfile.read(db_noise_path)
        noise = decimate(nk[:lens_h, mics], 3, axis=0)

        # STFT
        Xq = np.empty((NFFT // 2 + 1, N_frames, M, Q), dtype=complex)
        N = np.empty((NFFT // 2 + 1, N_frames, M), dtype=complex)
        for m in range(M):
            N[:, :, m] = stft(noise[:, m], nperseg=NFFT, noverlap=olap * NFFT, fs=fs)[2]
            for q in range(Q):
                Xq[:, :, m, q] = stft(xqf[:, m, q], nperseg=NFFT, noverlap=olap * NFFT, fs=fs)[2]
        absql = np.abs(Xq[F0, :, 0, :])
        absNl = np.abs(N[F0, :, 0])

        # Loop over SNR levels
        for ss, SNR in enumerate(SNRs):
            # Add Noise
            Gn = np.sqrt(np.mean(np.var(xqf[:, 0, :],axis=0)) / np.var(noise[:, 0]) * 10 ** (-SNR / 10))
            xt = x + Gn * noise

            # STFT for mixed signals
            Xt = np.empty((NFFT // 2 + 1, N_frames, M), dtype=complex)
            for m in range(M):
                f, t, Xt[:, :, m] = stft(xt[:, m], nperseg=NFFT, noverlap=olap * NFFT, fs=fs)

            # Ideal mask
            Tmask = np.argmax(np.concatenate((absql, Gn * absNl[:,:,None]), axis=2),axis=-1)
            #Tmask[20 * np.log10(np.abs(Xt[:, :, 0])) <= -35] = Q
            Tmask[20 * np.log10(np.abs(Xt[:, :, 0])) <= 0] = Q
            Tmask[:, np.sum(Tmask == Q, axis=0) / (NFFT // 2 + 1) > 0.85] = Q
     
            
            plt.figure()
            RdB = 60
            Sig = 20*np.log10(np.abs(Xt[:,:,0]))
            maxval = Sig.max()
            Sig[Sig<maxval-RdB] = maxval-RdB
            plt.pcolormesh(t, f, Sig)#np.abs(Tmask))
            plt.colorbar()
            plt.savefig('mix_signal.png')

            plt.figure()
            plt.pcolormesh(t, f, np.abs(Tmask))
            plt.savefig('true_mask.png')

            # Global Simplex
            d = 2
            Hl = FeatureExtr(Xt, F0, d)
            F = np.arange(int(np.ceil(f1 * NFFT / fs)), int(np.floor(f2 * NFFT / fs)) + 1)  # frequency band of interest
            lenF = len(F)
            Fall = (np.tile(np.arange(0, lenF0 * (M - 2) + 1, lenF0), (lenF, 1)) + np.tile(F.reshape(-1, 1), (1, M - 1))).flatten()

            pr2 = np.zeros((N_frames, Q + 1))
            for q in range(Q + 1):
                pr2[:, q] = np.sum(Tmask[F, :] == q, axis=0) / lenF

            Hlf = np.concatenate((np.real(Hl[Fall, :]), np.imag(Hl[Fall, :])), axis=0)
            mnorm = np.sqrt(np.sum(Hlf ** 2, axis=0))
            Hln = Hlf / (np.tile(mnorm + 1e-12, ((M - 1) * lenF * 2, 1)))
            KK = np.dot(Hln.T, Hln)

            de, E0 = np.linalg.eig(KK)
      

            d_sorted = np.sort(de)[::-1]
            den[:, ii, ss, rr] = d_sorted[:5]
            E0 = E0[:, np.argsort(de)[::-1]]

            ext0, _ = findextremeSPA(E0[:, :Q], Q)
            print(ext0)
            max3 = np.max(pr2[ext0, :], axis=0)
            print(max3)
            id0 = np.argmax(pr2[ext0, :Q], axis=0)

            pe = np.dot(E0[:, :Q], np.linalg.inv(E0[ext0, :Q]))
            pe = pe * (pe > 0)
            pe[pe.sum(1) > 1, :] = pe[pe.sum(1) > 1, :] / pe[pe.sum(1) > 1, :].sum(1, keepdims=True)
            pe = throwlow(pe)
            pe = np.hstack((pe, 1 - pe.sum(1, keepdims=True)))
            pe[:, :Q] = pe[:, id0]
            
            lambdas = np.arange(0,0.2,0.01).tolist()
            loss = np.zeros((N_frames, len(lambdas)))
            for j,lam in enumerate(lambdas):    
                C = pe > lam
                C_true = pr2 > 0

                FN = np.mean(1-np.sum(C*C_true,axis=1)/C_true.sum(axis=1))
                
                FP = np.mean(np.sum(C*(~C_true),axis=1)/np.maximum(1,(~C_true).sum(axis=1)))
                
                loss[:,j] = np.mean(1-np.sum(C*C_true,axis=1)/C_true.sum(axis=1))
                print(fr'$\lambda$ = {lam}, FN = {FN}, FP = {FP}')
            
            loss_list.append(loss)


alpha_list = [0.05, 0.1, 0.15]
calib_loss_table = np.concatenate(loss_list,axis=0)
for j,lam in enumerate(lambdas):
    print(lam, calib_loss_table[:,j].mean())

lhats = np.zeros(len(alpha_list))
for a, alpha in enumerate(alpha_list):
    lhats[a], lhat_idx = get_lhat(calib_loss_table, np.array(lambdas), alpha, B=1)
    print(alpha, lhats[a])   

np.savez(os.path.join(wav_folder,'calib.npz'),lhats=lhats,alphas=np.array(alpha_list))
# test_loss_table = np.concatenate(loss_list[6:],axis=1)  
#print('test loss', np.mean(test_loss_table[lhat_idx]))       