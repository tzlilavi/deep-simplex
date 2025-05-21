from beamforming import beamformer
from utils import throwlow, findextremeSPA, FeatureExtr, MaskErr

import os
import json
import glob
import numpy as np
from scipy.signal import decimate, stft
from scipy.io import wavfile
import scipy.signal as signal
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange
from sklearn.metrics import mean_squared_error
import pandas as pd

from deep_unmixing import deep_unmixing
from deep_unmixing import plot_heat_mat
from deep_unmixing import plot_P_speakers
from deep_unmixing import plot_simplex
from deep_unmixing import deep_unmixing_param_search

# Set seed for reproducibility
seed0 = 15#32#66#78
np.random.seed(seed0)

# Paths
db_base_path0 = '../DB/Seperate'
# db_alg_path = '/Users/blaufer/Documents/TAU/research/seperation/Enhancment'

# Parameters
seconds = 20 # 4, 8, 12, 16, 20
Q = 2
SNRs = [20]  # [0,5,10,15,20];

epochs = 200
deep_test = True
param_search_flag = False

fs_h = 48000
lens_h = seconds * fs_h
fs = 16000
lens = seconds * fs
sile = int(0.5 * fs)
sile_h = int(0.5 * fs_h)
NFFT = 2 ** 11
len_win = 2 ** 11
olap = 0.75
N_frames = int(lens/ ((1-olap) * NFFT) + 1)
revs = ['RT60_Low']  # ,'RT60_High'};

Noise_type = 'Air_conditioner'  # 'Babble_noise';%'Air_conditioner'; %'Babble_noise';
wav_folder = 'wav_examples'

F0 = np.arange(NFFT // 2 + 1)
lenF0 = len(F0)

f1 = 1000
f2 = 2000
sv = 343
k = np.arange(0, NFFT // 2 + 1)

Iter = 1  # 50

att = 0.2

mics = np.arange(24, 31)  # [7:8 10:11 13:14 16:17 19:20 22:23];%1:6;%[7:8 10:11 13:14 16:17 19:20 22:23];
M = len(mics)

speakers = 'ABCDEFHIJKLMNOPQST' #R
chairs = np.arange(1, 7)

figs_dir = 'figures'
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)
highpass = signal.firwin(199, 150, fs=fs, pass_zero="highpass")

# Generate test positions
MD = np.zeros((Iter, len(SNRs), len(revs)))
FA = np.zeros((Iter, len(SNRs), len(revs)))

SDR = np.zeros((Iter, len(SNRs), len(revs)))
SIR = np.zeros((Iter, len(SNRs), len(revs)))

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
            spkdir = sorted(glob.glob(dp_speaker_chair))[0]
            fs_sk, sk = wavfile.read(spkdir)
            info = {'name': speakers[spk[ii, q]], 'chair': str(chairs[chr[ii, q]]), 'first_sentence_number': spkdir.split('S_')[1].split('_')[0]}
            # with open(os.path.join(wav_folder,f'speaker_{q}.json'), "w") as json_file:
            #      json.dump(info, json_file)
            randint = np.random.randint(0, 3 * fs_h)
            deci = decimate(sk[randint:(randint + (lens_h - sile_h)), mics], 3, axis=0)
            xq[sile:sile+len(deci), :, q] = decimate(sk[randint:(randint + (lens_h - sile_h)), mics], 3, axis=0)

        # Apply high-pass to reduce noise
        xqf = np.zeros((lens, M, Q))
        for q in range(Q):
            for m in range(M):
                xqf[:, m, q] = signal.filtfilt(highpass, 1, xq[:, m, q])
            # wavfile.write(os.path.join(wav_folder,f'true_speaker_{q}.wav'),fs,xqf[:, 0, q].astype(np.int16))

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

            # wavfile.write(os.path.join(wav_folder,f'mix.wav'),fs,xt[:, 0].astype(np.int16))

            # STFT for mixed signals
            Xt = np.empty((NFFT // 2 + 1, N_frames, M), dtype=complex)
            for m in range(M):
                f, t, Xt[:, :, m] = stft(xt[:, m], nperseg=NFFT, noverlap=olap * NFFT, fs=fs)

            # Ideal mask
            Tmask = np.argmax(np.concatenate((absql, Gn * absNl[:,:,None]), axis=2),axis=-1)
            Tmask[20 * np.log10(np.abs(Xt[:, :, 0])) <= -10] = Q
            Tmask[:, np.sum(Tmask == Q, axis=0) / (NFFT // 2 + 1) > 0.85] = Q

        
            
            plt.figure()
            RdB = 60
            Sig = 20*np.log10(np.abs(Xt[:,:,0]))
            maxval = Sig.max()
            Sig[Sig<maxval-RdB] = maxval-RdB
            plt.pcolormesh(t, f, Sig)
            plt.savefig(os.path.join(figs_dir, 'mix_signal.png'))

            plt.figure()
            plt.pcolormesh(t, f, np.abs(Tmask))
            plt.savefig(os.path.join(figs_dir, 'true_mask.png'))

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

            ## Create SPA_P
            # ext0, _ = findextremeSPA(E0[:, :Q], Q)
            #
            # plot_simplex(E0[:, :Q].T, ext0, Q=Q, SNR=SNRs[0], seconds=seconds)
            # Q_mat = E0[ext0,:Q].T
            #
            # deep_dict = deep_unmixing(KK, E0[:, :Q], Q_mat)
            # Create SPA_P with different Qs


            losses_J=[]
            losses_diff_J = []
            losses_model1_J = []
            losses_model2_center_J = []

            losses_J_N = []
            losses_diff_J_N = []
            losses_model1_J_N = []
            losses_model2_center_J_N = []
            Js = [Q]
            Ps = []


            if  not param_search_flag:
                for J in Js:

                    ext0, _ = findextremeSPA(E0[:, :J], J)

                    plot_simplex(E0[:, :J].T, ext0, Q=J, SNR=SNRs[0], seconds=seconds)
                    Q_mat = E0[ext0, :J].T

                    deep_dict = deep_unmixing(KK, E0[:, :J], Q_mat, epochs, run_two_models_flag=False)
                    P = np.array(deep_dict['output_mat'])
                    a=5

                #     total_losses = deep_dict['total_losses']
                #     diff_losses = deep_dict['models_diff_losses']
                #     model1_losses = deep_dict['model1_losses']
                #     model2_center_losses = deep_dict['model2_center_losses']
                #     print(f'For real J = {Q}, For J = {J}: \n'
                #           f'model_total_loss = {total_losses[-1]} \n'
                #           f'diff_losses = {diff_losses[-1]} \n'
                #           f'model1_loss = {model1_losses[-1]}'
                #           f'\nmodel2_center_losses = {model2_center_losses[-1]} ')
                #
                #     losses_J.append(total_losses[-1])
                #     losses_diff_J.append(diff_losses[-1])
                #     losses_model1_J.append(model1_losses[-1])
                #     losses_model2_center_J.append(model2_center_losses[-1])
                #
                #     losses_J_N.append(total_losses[-1]/J)
                #     losses_diff_J_N.append(diff_losses[-1]/J)
                #     losses_model1_J_N.append(model1_losses[-1]/J)
                #     losses_model2_center_J_N.append(model2_center_losses[-1]/(J**2))
                #
                #     if J==Q:
                #         id0 = np.argmax(pr2[ext0, :J], axis=0)
                #         P[:, :J] = P[:, id0]
                #     Ps.append(P)
                # print(f'Smallest loss for J = {Js[np.argmin(losses_J)]} \nSmallest diff loss for J = {Js[np.argmin(losses_diff_J)]}, \n'
                #       f'Smallest model1 loss for J = {Js[np.argmin(losses_model1_J)]}, \n'
                #       f'Smallest model2 center loss for J = {Js[np.argmin(losses_model2_center_J)]}')
                #
                # print(
                #     f'Smallest normalized by J loss for J = {Js[np.argmin(losses_J_N)]} \nSmallest normalized by J diff loss for J = {Js[np.argmin(losses_diff_J_N)]}, \n'
                #     f'Smallest normalized by J model1 loss for J = {Js[np.argmin(losses_model1_J_N)]}, \n'
                #     f'Smallest normalized by J^2 center loss for J = {Js[np.argmin(losses_model2_center_J_N)]}')
                #
                # speaker_data = [
                #     (pr2[:, :Q], 'real_P_speakers', 'real P speakers over L', pr2[:, -1]),
                #     (Ps[0][:, :2], '2_speakers', '2_speakers', Ps[0][:, :-1]),
                #     (Ps[1][:, :3], '3_speakers', '3_speakers', Ps[0][:, :-1]),
                #     (Ps[2][:, :4], '4_speakers', '4_speakers', Ps[0][:, :-1]),
                #     (Ps[3][:, :5], '5_speakers', '5_speakers', Ps[0][:, :-1]),
                # ]
                #
                # # Create a new figure for subplots
                # plt.figure(figsize=(18, 6))
                #
                # # Loop through each speaker data and plot as subplot
                # for i, (speakers, plot_name, title, noise) in enumerate(speaker_data, 1):
                #     plt.subplot(1, 5, i)
                #     plot_P_speakers(speakers, plot_name, figs_dir, title=title, noise=noise, save_flag=False,
                #                     show_flag=False, need_fig=False)
                # plt.tight_layout()
                #
                # # Show the subplots
                # plt.show()
                # a=5

#######################
            ####### Parameters search
            else:
                for J in Js:
                    ext0, _ = findextremeSPA(E0[:, :J], J)

                    plot_simplex(E0[:, :J].T, ext0, Q=J, SNR=SNRs[0], seconds=seconds)
                    Q_mat = E0[ext0, :J].T

                    id0 = np.argmax(pr2[ext0, :Q], axis=0)


                    deep_unmixing_param_search(KK, E0[:, :J], Q_mat, id0,pr2, seconds, SNRs[0], run_two_models_flag=True)



            # print(f'Loss for 2 speakers: {losses[0]} \n Loss for 3 speakers: {losses[1]} \n Loss for 4 speakers: {losses[2]}')

            # np.save('W_correlation_matrix', KK)
            # np.save('SPA_Q', Q_mat)
            # np.save('U_matrix', E0[:, :Q])



            P = np.array(deep_dict['output_mat'])
            save_model_plots = False

            model_loss_lr = deep_dict['model_name'] + '_' + deep_dict['loss'] + '_' + str(deep_dict['lr']).replace('.', '')

            W_fig = plot_heat_mat(KK, 'W_mat', figs_dir, title='W heatmap', save_flag=True, show_flag=True)

            model_PPt = plot_heat_mat(np.matmul(P,P.T),  model_loss_lr + '_PPt', figs_dir, title="Model output's P * P.T", save_flag=save_model_plots, show_flag=False, d=deep_dict)

            real_PPt = plot_heat_mat(np.matmul(pr2[:,:Q],pr2[:,:Q].T), 'real_PPt', figs_dir, title="real (J cols) P * P.T", save_flag=False, show_flag=False)

            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(np.matmul(P, P.T), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("Model output's P * P.T")

            # Add the second heatmap as the second subplot
            plt.subplot(1, 2, 2)
            plt.imshow(np.matmul(pr2[:, :Q], pr2[:, :Q].T), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("Real (2 cols) P * P.T")
            plt.tight_layout()
            plt.show()
            #
            plot_P_speakers(P[:, :Q], model_loss_lr + '_P_speakers', figs_dir, title=model_loss_lr + ' P speakers over L', noise=P[:, -1], save_flag=save_model_plots, show_flag=False)
            plot_P_speakers(pr2[:, :Q], 'real_P_speakers', figs_dir, title='real P speakers over L', noise=pr2[:, 2],save_flag=False, show_flag=False)



            max3 = np.max(pr2[ext0, :], axis=0)
            print(max3)
            id0 = np.argmax(pr2[ext0, :Q], axis=0)

            pe = np.dot(E0[:, :Q], np.linalg.inv(E0[ext0, :Q]))
            pe = pe * (pe > 0)
            pe[pe.sum(1) > 1, :] = pe[pe.sum(1) > 1, :] / pe[pe.sum(1) > 1, :].sum(1, keepdims=True)
            pe = throwlow(pe)
            pe = np.hstack((pe, 1 - pe.sum(1, keepdims=True)))


            pe[:, :Q] = pe[:, id0]

            P[:, :Q] = P[:, id0]


            plot_P_speakers(pe[:, :Q], 'SPA_P_speakers', figs_dir, title='SPA P speakers over L',
                            noise=pe[:,-1], save_flag=True, show_flag=False)
            speaker_data = [
                (pr2[:, :Q], 'real_P_speakers', 'real P speakers over L', pr2[:, -1]),
                (P[:, :Q], model_loss_lr + '_P_speakers', model_loss_lr + ' P speakers over L', P[:, -1]),
                (pe[:, :Q], 'SPA_model_P_speakers', 'SPA P speakers over L', pe[:, -1])
            ]

            # Create a new figure for subplots
            plt.figure(figsize=(18, 6))

            # Loop through each speaker data and plot as subplot
            for i, (speakers, plot_name, title, noise) in enumerate(speaker_data, 1):
                plt.subplot(1, 3, i)
                plot_P_speakers(speakers, plot_name, figs_dir, title=title, noise=noise, save_flag=False,
                                show_flag=False, need_fig=False)

            # Adjust layout to prevent overlapping
            plt.tight_layout()

            # Show the subplots
            plt.show()


            deep_L2 = np.sum((pr2[:, :Q] - P[:, :Q]) ** 2)
            deep_mse = mean_squared_error(pr2[:, :Q], P[:, :Q])

            SPA_L2 = np.sum((pr2[:, :Q] - pe[:, :Q]) ** 2)
            SPA_mse = mean_squared_error(pr2[:, :Q], pe[:, :Q])

            print(f'L2(deep model, real): {deep_L2:.4f}, MSE(deep model, real): {deep_mse:.4f}')
            print(f'L2(SPA, real): {SPA_L2:.4f}, MSE(SPA, real): {SPA_mse:.4f}')


            deep_dict['deep_L2'] = deep_L2
            deep_dict['deep_MSE'] = deep_mse
            deep_dict['SPA_L2'] = SPA_L2
            deep_dict['SPA_mse'] = SPA_mse
            deep_dict.pop('output_mat')
            excel_file = 'save_runs_values.xlsx'

            existing_df = pd.read_excel(excel_file, engine='openpyxl')  # Specify the engine if needed
            existing_df = pd.concat([existing_df, pd.DataFrame([deep_dict])], ignore_index=True)
            existing_df.to_excel(excel_file, index=False, engine='openpyxl')  # Specify the engine if needed


            if deep_test:
                model_tested = deep_dict['model_name']
                pe = P
            else:
                model_tested = 'SPA'


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
            
            
            plt.figure()
            plt.pcolormesh(t, f, np.abs(Emask))
            plt.savefig(os.path.join(figs_dir, f'estimated_mask_{model_tested}'))

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
            
            
            SDRi[ii, ss, rr], SIRi[ii, ss, rr], yi = beamformer(Xt, Tmask, xqf, Q, fh2, olap, lens, 0.0001, 0.0001, 1, fs, att)
            SDR[ii, ss, rr], SIR[ii, ss, rr], ym = beamformer(Xt, Emask, xqf, Q, fh2, olap, lens, 0.01, 0.01, 1, fs, att)
            print(f'SDRi and SIRi: {SDRi[ii, ss, rr]:.2f} {SIRi[ii, ss, rr]:.2f}')    
            print(f'SDR and SIR: {SDR[ii, ss, rr]:.2f} {SIR[ii, ss, rr]:.2f}')


            for q in range(Q):
                wavfile.write(os.path.join(wav_folder,f'ideal_speaker_{q}_{att}_{model_tested}.wav'),fs,np.real(yi[:, q]).astype(np.int16))
                wavfile.write(os.path.join(wav_folder,f'est_speaker_{q}_{att}_{model_tested}.wav'),fs,np.real(ym[:, q]).astype(np.int16))

            
            np.savez(os.path.join(wav_folder,f'prob_{model_tested}.npz'), p_est=pe, p_true=pr2)


