from beamforming import beamformer, beamformer_C
from utils import throwlow, findextremeSPA, FeatureExtr, MaskErr, smoother
from conformal_risk_contorl import get_lhat
from mir_eval.separation import bss_eval_sources


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
db_base_path0 = '/Users/blaufer/Library/CloudStorage/OneDrive-Tel-AvivUniversity/TAU/research/seperation/Elisheva/DB/Seperate'
db_alg_path = '/Users/blaufer/Library/CloudStorage/OneDrive-Tel-AvivUniversity/TAU/research/seperation/Enhancment'

# Parameters
seconds = 24
fs_h = 48000
lens_h = seconds * fs_h
fs = 16000
spk_olap = 3*fs#3*fs
lens = seconds * fs
sile = int(0.5 * fs)
sile_h = int(0.5 * fs_h)
NFFT = 2 ** 11
len_win = 2 ** 11
olap = 0.75
N_frames = int(lens/ ((1-olap) * NFFT) + 1)
revs = ['RT60_Low']  # ,'RT60_High'};
SNRs = [30]  # [0,5,10,15,20];
Noise_type = 'Air_conditioner'  # 'Babble_noise';%'Air_conditioner'; %'Babble_noise';
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
speakers = 'KLMNOPQRST' 
chairs = np.arange(1, 7)

calib = np.load(os.path.join(wav_folder,'calib.npz'))
alpha = 0.1
lhat = calib['lhats'][1]
print(f'lhat {lhat}')

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
        if not os.path.exists(os.path.join(wav_folder,f'Iter_{ii}')):
           os.makedirs(os.path.join(wav_folder,f'Iter_{ii}'))
        # Load Signals
        xq = np.zeros((lens, M, Q))
        for q in range(Q):
            db_speaker_path = os.path.join(db_base_path, speakers[spk[ii, q]])
            dp_speaker_chair = os.path.join(db_speaker_path, f'*Chair_{chairs[chr[ii, q]]}*.wav')
            dir_spk = sorted(glob.glob(dp_speaker_chair))[0]
            spkdir = os.path.join(db_speaker_path, dir_spk)
            fs_sk, sk = wavfile.read(spkdir)
            info = {'name': speakers[spk[ii, q]], 'chair': str(chairs[chr[ii, q]]), 'first_sentence_number': dir_spk.split('S_')[1].split('_')[0]}
            with open(os.path.join(wav_folder,f'speaker_{q}.json'), "w") as json_file:
                 json.dump(info, json_file)
            randint = 0*fs_h #if q == 0 else 5*fs_h#np.random.randint(0, 3 * fs_h)
            deci = decimate(sk[randint:(randint + (lens_h - sile_h)//2), mics], 3, axis=0)
            if q==0:
                xq[sile:sile+len(deci), :, q] = deci#decimate(sk[randint:(randint + (lens_h - sile_h)), mics], 3, axis=0)
            else:
                xq[sile+len(deci)-spk_olap:sile+2*len(deci)-spk_olap, :, q] = deci

        # Apply high-pass to reduce noise
        xqf = np.zeros((lens, M, Q))
        for q in range(Q):
            for m in range(M):
                xqf[:, m, q] = signal.filtfilt(highpass, 1, xq[:, m, q])
            wavfile.write(os.path.join(wav_folder,f'Iter_{ii}',f'true_speaker_{q}.wav'),fs,xqf[:, 0, q].astype(np.int16))
                
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

            SDRmix, SIRmix, _, _ = bss_eval_sources(np.tile(xt[:,0][:,None],(1,Q)).T, np.squeeze(xqf[:,0,:]).T)
            print(SDRmix, SIRmix)

            wavfile.write(os.path.join(wav_folder,f'Iter_{ii}',f'mix.wav'),fs,xt[:, 0].astype(np.int16))

            # STFT for mixed signals
            Xt = np.empty((NFFT // 2 + 1, N_frames, M), dtype=complex)
            for m in range(M):
                f, t, Xt[:, :, m] = stft(xt[:, m], nperseg=NFFT, noverlap=olap * NFFT, fs=fs)

            # Ideal mask
            Tmask = np.argmax(np.concatenate((absql, Gn * absNl[:,:,None]), axis=2),axis=-1)
            #Tmask = np.argmax(Fmask, axis=3)
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

            #Fall_ext = Fall + lenF0
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
            
            # lambdas = np.arange(0,0.2,0.01).tolist()
            # alpha = 0.1
            # loss = np.zeros((N_frames, len(lambdas)))
            # for j,lam in enumerate(lambdas):    
            #     C = pe > lam
            #     C_true = pr2 > 0

            #     FN = np.mean(1-np.sum(C*C_true,axis=1)/C_true.sum(axis=1))
                
            #     FP = np.mean(np.sum(C*(~C_true),axis=1)/np.maximum(1,(~C_true).sum(axis=1)))
                
            #     loss[:,j] = np.mean(1-np.sum(C*C_true,axis=1)/C_true.sum(axis=1))
            #     print(fr'$\lambda$ = {lam}, FN = {FN}, FP = {FP}')
            #     loss_list.append(loss)

            C = pe > lhat
            C_true = pr2 > 0
            FN = np.mean(1-np.sum(C*C_true,axis=1)/C_true.sum(axis=1))
            print(f'FN = {FN}')
            C_smooth = smoother(C)

            Ne = 10
            fh2 = []
            for q in range(Q + 1):
                srow = np.argsort(pe[:, q])[::-1]
                fh2.append(srow[:Ne])


            # Local Mapping
            Emask = np.zeros((NFFT // 2 + 1, N_frames))
            for kk in trange(NFFT // 2 + 1):
                Lb = 1
                Fk = np.array([kk])
                Fall = np.tile(np.arange(0, lenF0 * (M - 1), lenF0)[:, np.newaxis], (1, Lb)) + np.tile(Fk[np.newaxis, :], (M - 1, 1))
                Fall = Fall.flatten()

                Hlf = np.concatenate((np.real(Hl[Fall, :]), np.imag(Hl[Fall, :])), axis=0)
                mnorm = np.sqrt(np.sum(Hlf ** 2, axis=0))
                Hln = Hlf / np.tile((mnorm + 1e-12),((M - 1) * Lb * 2, 1))
                #KK = np.dot(Hln.T, Hln)

                #De, E = np.linalg.eig(KK)
                #pdistEE = distance_matrix(E[:, :Q] ,E[:, :Q])#np.sqrt(np.sum((E[:, :Q] - E0[:, :Q]) ** 2, axis=1))
                pdistEE = distance_matrix(Hln.T ,Hln.T)
                p_dist_local = np.exp(-pdistEE)
                decide = np.dot(p_dist_local, pe)/(np.tile(pe.sum(axis=0), (N_frames, 1)))
                idk = np.argmax(decide, axis=1)
                Emask[kk, :] = idk

            
            Emask[20 * np.log10(np.abs(Xt[:, :, 0])) <= -10] = Q
            Emask[:,pe[:, Q] > 0.85] = Q
            # for nn in range(N_frames):
            #     q_domin = np.argmax(pe[nn,:])
            #     for q in range(Q):
            #         if pe[nn,q]<0.25:
            #             Emask[Emask[:,nn]==q,nn] = q_domin
            

            
            plt.figure()
            plt.pcolormesh(t, f, np.abs(Emask))
            plt.savefig('estimated_mask.png')

            Np = 10
            Pmask = (Q + 2) * np.ones((NFFT // 2 + 1, N_frames))
            fq = []
            for q in range(Q + 1):
                srow = np.argsort(pe[:, q])[::-1]
                fq.append(srow[:Np])
                Pmask[:, fq[q]] = q

            MD[ii, ss, rr], FA[ii, ss, rr] = MaskErr(Tmask, Emask, Q)
            print(f'MD and FA: {MD[ii, ss, rr]:.2f} {FA[ii, ss, rr]:.2f}\n')

            SDRp[ii, ss, rr], SIRp[ii, ss, rr], _ = beamformer(Xt, Pmask, xqf, Q, fq, olap, lens, 0.01, 0.01, 0, fs)
            print(f'SDRp and SIRp: {SDRp[ii, ss, rr]:.2f} {SIRp[ii, ss, rr]:.2f}\n')

            att = 0.3 #np.arange(0.0,0.6,0.1).tolist()
            
            SDRi[ii, ss, rr], SIRi[ii, ss, rr], yi = beamformer(Xt, Tmask, xqf, Q, fh2, olap, lens, 0.0001, 0.0001, 1, fs, att)
            SDR[ii, ss, rr], SIR[ii, ss, rr], ym = beamformer(Xt, Emask, xqf, Q, fh2, olap, lens, 0.01, 0.01, 1, fs, att)
            SDRC[ii, ss, rr], SIRC[ii, ss, rr], yc = beamformer_C(Xt, Emask, xqf, Q, fh2, olap, lens, 0.01, 0.01, 1, fs, att, C)
            print(f'SDRi and SIRi: {SDRi[ii, ss, rr]:.2f} {SIRi[ii, ss, rr]:.2f}')    
            print(f'SDR and SIR: {SDR[ii, ss, rr]:.2f} {SIR[ii, ss, rr]:.2f}')
            print(f'SDRC and SIRC: {SDRC[ii, ss, rr]:.2f} {SIRC[ii, ss, rr]:.2f}')


            for q in range(Q):
                wavfile.write(os.path.join(wav_folder,f'Iter_{ii}',f'ideal_speaker_{q}.wav'),fs,np.real(yi[:, q]).astype(np.int16))
                wavfile.write(os.path.join(wav_folder,f'Iter_{ii}',f'est_speaker_{q}.wav'),fs,np.real(ym[:, q]).astype(np.int16))
                wavfile.write(os.path.join(wav_folder,f'Iter_{ii}',f'est_speaker_{q}_c.wav'),fs,np.real(yc[:, q]).astype(np.int16))


print(f'SDRi and SIRi: {SDRi.mean():.2f} {SIRi.mean():.2f}')    
print(f'SDR and SIR: {SDR.mean():.2f} {SIR.mean():.2f}')
print(f'SDRC and SIRC: {SDRC.mean():.2f} {SIRC.mean():.2f}')

