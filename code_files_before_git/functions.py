import os
import joblib
import numpy as np
import glob
import torch.nn as nn
from openpyxl.descriptors import NoneSet
from scipy import signal
from scipy.io import wavfile
from scipy.signal import decimate, stft
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pesq import pesq
from pystoi import stoi
import CFG
from custom_losses import find_best_permutation_supervised, SupervisedLoss
import torchiva
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as t_audio_SI_SDR
from torchmetrics.functional.audio import signal_distortion_ratio as t_audio_SDR
from utils import throwlow, findextremeSPA, FeatureExtr, MaskErr, smoother
from beamforming import beamformer, beamformer_nonoise, bss_eval_sources, MVDR_over_speakers
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance_matrix
from tqdm import trange
import random
import torch
import itertools
from itertools import permutations
import soundfile as sf
from pyroomacoustics.bss.ilrma import ilrma
random.seed(CFG.seed0)
np.random.seed(CFG.seed0)

def initialize_arrays():
    MD = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    FA = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    MD_pe = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    FA_pe = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SDR = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SIR = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SDR_pe = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SIR_pe = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SDRi = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SIRi = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SDRiva = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SIRiva = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SDRp = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SIRp = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SDRp_pe = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SIRp_pe = np.zeros((CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    SNRin = np.zeros((CFG.Iter, len(CFG.revs)))
    den = np.zeros((5, CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    sumin = np.zeros((5, CFG.Iter, len(CFG.SNRs), len(CFG.revs)))
    spk = np.empty((CFG.Iter, CFG.Q), dtype=int)
    chr = np.empty((CFG.Iter, CFG.Q), dtype=int)
    return MD, FA, MD_pe, FA_pe, SDR, SIR, SDR_pe, SIR_pe, SDRi, SIRi, SDRiva, SIRiva, SDRp, SIRp, SDR_pe, SIR_pe, SNRin, den, sumin, spk, chr


def generate_test_positions(spk, chr):
    for ii in range(CFG.Iter):
        rand_spk = np.random.permutation(len(CFG.speakers))[:CFG.Q]
        spk[ii, :] = rand_spk
        rand_chr = np.random.permutation(len(CFG.chairs))[:CFG.Q]
        chr[ii, :] = rand_chr


def process_signals(rr, rev, spk, chr, ss, SNR):
    db_base_path = os.path.join(CFG.db_base_path0, rev)

    for ii in range(CFG.Iter):
        xq = np.zeros((CFG.lens, CFG.M, CFG.Q))
        for q in range(CFG.Q):
            db_speaker_path = os.path.join(db_base_path, CFG.speakers[spk[ii, q]])
            dp_speaker_chair = os.path.join(db_speaker_path, f'*Chair_{CFG.chairs[chr[ii, q]]}*.wav')
            spkdir = sorted(glob.glob(dp_speaker_chair))[0]
            fs_sk, sk = wavfile.read(spkdir)
            info = {'name': CFG.speakers[spk[ii, q]], 'chair': str(CFG.chairs[chr[ii, q]]),
                    'first_sentence_number': spkdir.split('S_')[1].split('_')[0]}
            randint = np.random.randint(0, 3 * CFG.fs_h)
            deci = decimate(sk[randint:(randint + (CFG.lens_h - CFG.sile_h)), CFG.mics], 3, axis=0)
            xq[CFG.sile:CFG.sile + len(deci), :, q] = decimate(
                sk[randint:(randint + (CFG.lens_h - CFG.sile_h)), CFG.mics], 3, axis=0)

        xqf = np.zeros((CFG.lens, CFG.M, CFG.Q))
        for q in range(CFG.Q):
            for m in range(CFG.M):
                xqf[:, m, q] = signal.filtfilt(CFG.highpass, 1, xq[:, m, q])
        x = np.sum(xqf, axis=2)

        db_noise_path = os.path.join(CFG.db_base_path0, rev, f'Noises/In/{CFG.Noise_type}_in_{rev}.wav')
        fs_nk, nk = wavfile.read(db_noise_path)
        noise = decimate(nk[:CFG.lens_h, CFG.mics], 3, axis=0)

        Xq = np.empty((CFG.NFFT // 2 + 1, CFG.N_frames, CFG.M, CFG.Q), dtype=complex)
        N = np.empty((CFG.NFFT // 2 + 1, CFG.N_frames, CFG.M), dtype=complex)
        for m in range(CFG.M):
            N[:, :, m] = stft(noise[:, m], nperseg=CFG.NFFT, noverlap=CFG.olap * CFG.NFFT, fs=CFG.fs)[2]
            for q in range(CFG.Q):
                Xq[:, :, m, q] = stft(xqf[:, m, q], nperseg=CFG.NFFT, noverlap=CFG.olap * CFG.NFFT, fs=CFG.fs)[2]
        absql = np.abs(Xq[CFG.F0, :, 0, :])
        absNl = np.abs(N[CFG.F0, :, 0])

        # Add Noise
        Gn = np.sqrt(np.mean(np.var(xqf[:, 0, :], axis=0)) / np.var(noise[:, 0]) * 10 ** (-SNR / 10))
        xt = x + Gn * noise


        Xt = np.empty((CFG.NFFT // 2 + 1, CFG.N_frames, CFG.M), dtype=complex)
        for m in range(CFG.M):
            f, t, Xt[:, :, m] = stft(xt[:, m], nperseg=CFG.NFFT, noverlap=CFG.olap * CFG.NFFT, fs=CFG.fs)

        Tmask = np.argmax(np.concatenate((absql, Gn * absNl[:, :, None]), axis=2), axis=-1)
        Tmask[20 * np.log10(np.abs(Xt[:, :, 0])) <= -10] = CFG.Q
        Tmask[:, np.sum(Tmask == CFG.Q, axis=0) / (CFG.NFFT // 2 + 1) > 0.85] = CFG.Q

        return Xt, Tmask, f, t, xqf


def feature_extraction(Xt):
    d = 2
    Hl, Hlm = FeatureExtr(Xt, CFG.F0, d)
    F = np.arange(int(np.ceil(CFG.f1 * CFG.NFFT / CFG.fs)), int(np.floor(CFG.f2 * CFG.NFFT / CFG.fs)) + 1)
    lenF = len(F)
    Fall = (np.tile(np.arange(0, CFG.lenF0 * (CFG.M - 2) + 1, CFG.lenF0), (lenF, 1)) + np.tile(F.reshape(-1, 1), (
    1, CFG.M - 1))).flatten()

    Hlf = np.zeros((CFG.lenF0, CFG.N_frames, (CFG.M - 1) * 2), dtype=complex)
    for kk in range(CFG.NFFT // 2 + 1):
        Lb = 1
        Fk = np.array([kk])
        Fall_f = np.tile(np.arange(0, CFG.lenF0 * (CFG.M - 1), CFG.lenF0)[:, np.newaxis], (1, Lb)) + np.tile(
            Fk[np.newaxis, :],(CFG.M - 1, 1))
        Fall_f = Fall_f.flatten()

        hlf = np.concatenate((np.real(Hl[Fall_f, :]), np.imag(Hl[Fall_f, :])), axis=0)
        mnorm = np.sqrt(np.sum(hlf ** 2, axis=0))
        Hlf[kk, :, :] = (hlf / np.tile((mnorm + 1e-12), ((CFG.M - 1) * Lb * 2, 1))).T

    return Hl, Hlm, Hlf, Fall, lenF, F


def calculate_W_U_realSimplex(Hl, Fall, Tmask, lenF, F, J=CFG.Q, file=None, add_noise=CFG.add_noise):
    Hl = np.nan_to_num(Hl, nan=0)
    Hlf = np.concatenate((np.real(Hl[Fall, :]), np.imag(Hl[Fall, :])), axis=0)
    mnorm = np.sqrt(np.sum(Hlf ** 2, axis=0))
    Hln = Hlf / (np.tile(mnorm + 1e-12, ((CFG.M - 1) * lenF * 2, 1)))
    Hln[:, 0] = np.mean(Hln[:, 0:], axis=1)
    KK = np.dot(Hln.T, Hln)

    diag = KK[range(CFG.N_frames), range(CFG.N_frames)]
    first_nonzero_l = np.where(diag > 0.9)[0][0]


    pr2 = np.zeros((CFG.N_frames, J + CFG.add_noise))
    for q in range(J + add_noise):
        pr2[:, q] = np.sum(Tmask[F, :] == q, axis=0) / lenF

    # if CFG.pad_flag:
    #     de, E0 = np.linalg.eig(KK[CFG.pad_tfs:-CFG.pad_tfs, CFG.pad_tfs:-CFG.pad_tfs])
    #     pr2[:CFG.pad_tfs, :] = 0
    #     pr2[-CFG.pad_tfs:, :] = 0
    # else:
    de, E0 = np.linalg.eig(KK)
    d_sorted = np.sort(de)[::-1]
    E0 = E0[:, np.argsort(de)[::-1]]
    return Hln, KK, E0, pr2, first_nonzero_l


def calculate_SPA_simplex(E0, pr2, J, add_noise=CFG.add_noise):
    ext0, _ = findextremeSPA(E0[:, :J], J)

    id0 = np.argmax(pr2[ext0, :J], axis=0)
    # id0 = ensure_permutation(id0, J)

    pe = np.dot(E0[:, :J], np.linalg.inv(E0[ext0, :J]))
    pe = pe * (pe > 0)
    pe[pe.sum(1) > 1, :] = pe[pe.sum(1) > 1, :] / pe[pe.sum(1) > 1, :].sum(1, keepdims=True)
    pe = throwlow(pe)
    if add_noise:
        pe = np.hstack((pe, 1 - pe.sum(1, keepdims=True)))


    loss_function = SupervisedLoss(L2_factor=CFG.L2_factor, SAD_factor=CFG.SAD_factor, J=J, add_noise=add_noise)
    _, pe, best_permutation, _, _ = find_best_permutation_supervised(loss_function, torch.from_numpy(pe).unsqueeze(0).float().to(
                                                                        CFG.device),
                                                                    torch.from_numpy(pr2).unsqueeze(0).float().to(
                                                                        CFG.device))
    pe = pe.cpu().numpy().squeeze(0)
    best_permutation = best_permutation[:J]
    ext0 = ext0[[best_permutation]]
    # pe[:, :J] = pe[:, id0]
    return pe, id0, ext0



def ensure_permutation(id0, J=CFG.Q):
    unique_values, counts = np.unique(id0, return_counts=True)
    duplicates = unique_values[counts > 1]  # Find duplicate elements

    if len(duplicates) > 0:
        all_possible_values = set(range(J))  # Set of all possible values
        chosen_values = set(id0)  # Set of already chosen values
        available_values = list(all_possible_values - chosen_values)  # Find available values

        if available_values:  # Ensure we have available values to replace duplicates
            for duplicate in duplicates:
                indices = np.where(id0 == duplicate)[0]
                for i in range(1, len(indices)):  # Start from the second occurrence
                    if available_values:
                        id0[indices[i]] = available_values.pop(0)  # Replace with available value

    return id0

def save_plots(t, f, Xt, Tmask):
    import matplotlib.pyplot as plt

    plt.figure()
    RdB = 60
    Sig = 20 * np.log10(np.abs(Xt[:, :, 0]))
    maxval = Sig.max()
    Sig[Sig < maxval - RdB] = maxval - RdB
    plt.pcolormesh(t, f, Sig)
    plt.savefig(os.path.join(CFG.figs_dir, 'mix_signal.png'))

    plt.figure()
    plt.pcolormesh(t, f, np.abs(Tmask))
    plt.savefig(os.path.join(CFG.figs_dir, 'true_mask.png'))

def plot_heat_mat(mat, plot_name='W', figs_directory='figures', title=None, save_flag=False, show_flag=True, d=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, cmap='hot', interpolation='nearest')
    plt.colorbar()
    if title:
        plt.title(title)
    if d:
        params_text = f'Model: {d["model_name"]}\n loss: {d["loss_name"]}\n value: {d["loss"]}\n lr: {d["lr"]}\n epochs: {d["epochs"]}'
        bbox_props = dict(boxstyle='round, pad=0.3', edgecolor='black', facecolor='white', alpha=0.8)
        plt.text(1, 1.05, params_text, transform=plt.gca().transAxes, fontsize=8, va='center', ha='left', bbox=bbox_props)
    if save_flag:
        plt.savefig(os.path.join(figs_directory, plot_name))
    if show_flag:
        plt.show()
    return mat

def plot_P_speakers(speakers, plot_name, figs_directory, noise=None, title=None, save_flag=True, show_flag=True, need_fig=True, t=None):
    colors = ['blue', 'red', 'green']
    if t is None:
        t = np.arange(0, len(speakers))
        xlabel = 'Timeframes'
    else:
        xlabel = 'Time [s]'
    if need_fig:
        plt.figure()
        plt.figure(figsize=(8, 6))

    for i in range(speakers.shape[1]):
        plt.plot(t,speakers[:,i], label=f'Speaker {i+1}', c=colors[i])
    if noise is not None and np.any(noise):
        plt.plot(t, noise,'gray', label='Noise', linewidth=0.5)
    plt.legend(loc='upper right')
    if title:
        plt.title(title)
    if save_flag:
        plt.savefig(os.path.join(figs_directory, plot_name))
    plt.xlabel(xlabel)
    plt.ylabel('Speakers probabilities')
    if show_flag:
        plt.show()

def plot_results(P, pr2, pe, id0=None, J=CFG.Q, no_title=False, plot_flag=CFG.plot_flag, t=None, SISDRs=None, noise_P=None):
    if not plot_flag:
        return
    # if id0 is not None and CFG.noise==1:
    #     pe[:, :J] = pe[:, id0]
    #
    #     # P[:, :J] = P[:, id0]

    save_model_plots = False
    if no_title:
        title_ideal = ''
        title_deep = ''
        title_simplex = ''

    else:
        if SISDRs is not None:
            title_ideal = f'Ideal global speaker probabilities, SI-SDR = {SISDRs[0]:.3f}'
            title_deep = f'Deep Simplex global speaker probabilities, SI-SDR = {SISDRs[1]:.3f}'
            title_simplex = f'Standard Simplex global speaker probabilities, SI-SDR = {SISDRs[2]:.3f}'
        else:
            title_ideal = f'Ideal global speaker probabilities'
            title_deep = f'Deep Simplex global speaker probabilities'
            title_simplex = f'Standard Simplex global speaker probabilities'

    noise_pr2 = None
    noise_pe = None

    if CFG.add_noise:
        noise_pr2 = pr2[:, -1]
        noise_pe = pe[:, -1]


    speaker_data = [
        (pr2[:, :J], 'real_P_speakers', title_ideal, noise_pr2),
        (P[:, :J], 'P_speakers', title_deep, noise_P),
        (pe[:, :J], 'SPA_model_P_speakers', title_simplex, noise_pe)
    ]

    # Create a new figure for subplots
    plt.figure(figsize=(18, 6))

    # Loop through each speaker data and plot as subplot
    for i, (speakers, plot_name, title, noise) in enumerate(speaker_data, 1):
        plt.subplot(1, 3, i)
        plot_P_speakers(speakers, plot_name, CFG.figs_dir, title=title, noise=noise, save_flag=False,
                        show_flag=False, need_fig=False, t=t)

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()


def plot3d_simplex(P, vertices_indices, vertices_weights=None, title='P Simplex', azim=30, elev=30, vector_type='v',
                   plot_name=None):
    P = np.real(P)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='orange', s=10, alpha=0.5, label='Data points', zorder=1)
    colors = ['blue', 'red', 'green']
    s = 200
    if len(vertices_indices.shape) > 1:
        for speaker_idx, color in enumerate(colors):
            speaker_indices = vertices_indices[:, speaker_idx]
            size = s
            if vertices_weights is None:
                for rank, vertex_idx in enumerate(speaker_indices):
                    ax.scatter(P[vertex_idx, 0], P[vertex_idx, 1], P[vertex_idx, 2], c=color,
                               s=size, label=f'Speaker {speaker_idx + 1}, Rank {vertex_idx + 1}', lw=0)
                    size /= 2
            else:
                for rank, vertex_idx in enumerate(speaker_indices):
                    size = s * vertices_weights[rank, speaker_idx]
                    ax.scatter(P[vertex_idx, 0], P[vertex_idx, 1], P[vertex_idx, 2], c=color,
                               s=size, label=f'Speaker {speaker_idx + 1}, Rank {vertex_idx + 1}',
                               zorder=4)

    else:
        ax.scatter(P[vertices_indices[0], 0], P[vertices_indices[0], 1], P[vertices_indices[0], 2], c=colors[1], s=s,
                   label='Speaker 1', zorder=4, edgecolors='black', depthshade=False)
        ax.scatter(P[vertices_indices[1], 0], P[vertices_indices[1], 1], P[vertices_indices[1], 2], c=colors[1], s=s,
                   label='Speaker 2', zorder=4, edgecolors='black')
        ax.scatter(P[vertices_indices[2], 0], P[vertices_indices[2], 1], P[vertices_indices[2], 2], c=colors[1], s=s,
                   label='Speaker 3', zorder=4, edgecolors='black')

    if len(P.shape) == 4:
        ax.scatter(P[vertices_indices[3], 0], P[vertices_indices[3], 1], P[vertices_indices[3], 2], s=100,
                   label='Speaker 4')
    elif len(P.shape) == 5:
        ax.scatter(P[vertices_indices[4], 0], P[vertices_indices[4], 1], P[vertices_indices[4], 2], s=100,
                   label='Speaker 5')
    ax.view_init(azim=azim, elev=elev)
    ax.set_xlabel(r"$\mathbf{" + vector_type + r"}_1$", fontsize=14)
    ax.set_ylabel(r"$\mathbf{" + vector_type + r"}_2$", fontsize=14)
    ax.set_zlabel(r"$\mathbf{" + vector_type + r"}_3$", fontsize=14)
    if vector_type == 'u':
        ax.set_xticks([-0.05, 0, 0.05])
        ax.set_yticks([-0.05, 0, 0.05])
        ax.set_zticks([-0.05, 0, 0.05])
    else:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_zticks([0, 0.5, 1])
    ax.set_title(title)
    # ax.legend(fontsize='8', loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    if plot_name is not None:
        plt.savefig(f'experiments_results/{plot_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.3)

    plt.show()
def plot_simplex(vectors, vertices_indexes, Q, seconds=20, SNR=20, figs_dir='figures'):
    fig = plt.figure(figsize=(8, 8))
    sc = plt.scatter(vectors[:, 0], vectors[:, 1], color='orange')  # Scatter plot the points
    colors = ['blue', 'red', 'green']
    for idx, ext in enumerate(vertices_indexes[0]):
        plt.scatter(vectors[ext, 0], vectors[ext, 1], color=colors[idx], marker='o', s=100)

    plt.xlabel('U0')
    plt.ylabel('U1')
    plt.title(f'')
    plt.grid(True)
    cbar = plt.colorbar(sc)
    plt.show()

def find_top_active_indices(pe, P, P2, pr2, Ne=10, P_method=CFG.P_method, add_noise=CFG.add_noise):
    fh = []
    fh2 = []
    fh22 = []
    fh2_pe = []
    J = P.shape[1]
    for q in range(J + add_noise):
        srow = np.argsort(pr2[:, q])[::-1]
        fh.append(srow[:Ne])

        srow = np.argsort(P[:, q])[::-1]
        fh2.append(srow[:Ne])

        srow_pe = np.argsort(pe[:, q])[::-1]
        fh2_pe.append(srow_pe[:Ne])

        if P_method=='both':
            srow2 = np.argsort(P2[:, q])[::-1]
            fh22.append(srow2[:Ne])
    return fh2, fh22, fh2_pe, fh

def local_mapping(pe, P, P2, pr2, Hlf, Xt, low_energy_mask, J, f, t, P_method, model_tested,
                  Tmask, add_noise=CFG.add_noise, plot_Emask=False):

    print('Running Nearest Neighbour Local Mapping...')


    # Local Mapping
    Emask = np.zeros((CFG.NFFT // 2 + 1, CFG.N_frames))
    Emask2 = np.zeros((CFG.NFFT // 2 + 1, CFG.N_frames))
    Emask_pe = np.zeros((CFG.NFFT // 2 + 1, CFG.N_frames))
    lenF0 = len(CFG.F0)

    for kk in range(CFG.NFFT // 2 + 1):

        pdistEE = distance_matrix(Hlf[kk,:,:], Hlf[kk,:,:])
        p_dist_local = np.exp(-pdistEE)

        decide = np.dot(p_dist_local, P) / (np.tile(P.sum(axis=0) + 1e-12, (CFG.N_frames, 1)))
        idk = np.argmax(decide, axis=1)
        Emask[kk, :] = idk

        if P_method=='both':
            decide2 = np.dot(p_dist_local, P2) / (np.tile(P2.sum(axis=0)+ 1e-12, (CFG.N_frames, 1)))
            idk2 = np.argmax(decide2, axis=1)
            Emask2[kk, :] = idk2

        decide_pe = np.dot(p_dist_local, pe) / (np.tile(pe.sum(axis=0) + 1e-12, (CFG.N_frames, 1)))
        idk_pe = np.argmax(decide_pe, axis=1)
        Emask_pe[kk, :] = idk_pe

    if CFG.add_noise == 1:
        Emask[low_energy_mask] = J
        Emask[:, P[:, J] > 0.85] = J

        if P_method == 'both':
            Emask2[low_energy_mask] = J
            Emask2[:, P[:, J] > 0.85] = J

        Emask_pe[low_energy_mask] = J
        Emask_pe[:, pe[:, J] > 0.85] = J

    else:
        Emask[low_energy_mask] = J


        if P_method == 'both':
            Emask2[low_energy_mask] = J
        Emask_pe[low_energy_mask] = J

    if plot_Emask:
        plot_masks(Tmask, Emask_pe, Emask, P_method='vertices')

        plot_masks(Tmask, Emask_pe, Emask2, P_method='prob')





    return Emask, Emask2, Emask_pe


def dist_scores(P, pr2, J, P_method):

    # L2 = np.sum((pr2[:, :J] - P[:, :J]) ** 2)
    mse = mean_squared_error(pr2[:, :J], P[:, :J])


    return mse


def si_sdr(s, s_hat):
    alpha = np.sum(s_hat * s, axis=0) / np.linalg.norm(s, axis=0)**2
    sdr = 10*np.log10(np.linalg.norm(alpha*s, axis=0) **2
                      / np.linalg.norm(alpha*s - s_hat, axis=0) **2)
    return sdr.mean()

def compute_true_RTFs_from_Xq(Xq):
    """
    Xq: F × T × M × J (complex STFT)
    Returns: H_gt of shape F × M × J
    """
    F, T, M, J = Xq.shape
    H_gt = np.zeros((F, M, J), dtype=np.complex128)
    for j in range(J):
        for m in range(M):
            H_est = Xq[:, :, m, j] / (Xq[:, :, 0, j] + 1e-8)  # Mic m / Mic 0
            H_gt[:, m, j] = np.mean(H_est, axis=1)  # average over time
    return H_gt

def audio_scores(pr2, P, Tmask, Emask,
                 Hl, Xt, Hq, Xq, xqf, fh2, fh, J=CFG.Q, compute_ideal=False,
                 P_method=CFG.P_method, local_method='NN', print_scores=True):
    key_suffix = P_method + '_' + local_method
    olap, lens, att, fs = CFG.olap, CFG.lens, CFG.att, CFG.old_fs
    SDRi, sisdri, yi = None, None, None
    SDRii, sisdrii, yii = None, None, None
    u_GT = np.nan_to_num(np.asarray(xqf[:, 0, :].real, dtype=np.float32))

    L2 = dist_scores(P, pr2, J, P_method)
    print(f'L2({P_method}, real): {L2:.4f}')
    # Compute SDRii, sisdrii, yii, SDRi, sisdri, yi if compute_ideal is True
    if compute_ideal:
        # H_gt = compute_true_RTFs_from_Xq(Xq)
        # Compute ideal
        SDRi, sisdri, yi = beamformer_nonoise(Xt, Tmask, xqf, J, fh, olap, lens, 0.0001, 1, CFG.fs, CFG.att,
                                              Hq = None)
        # SDRii, sisdrii, yii = beamformer_nonoise(Xt, None, xqf, J, fh2, olap, lens, 0.0001, 0, CFG.fs, CFG.att, Xq, Hq, Hl)

        # try:
        #     SDR_aux, sisdr_aux, stoi_aux, pesq_aux = calc_auxip(Xt, xqf[:, 0, :], J=J)
        # except Exception as e:
        #     SDR_aux, sisdr_aux, stoi_aux, pesq_aux = 4, 3, 0.6, 2
        #     print(f"Error in AuxIVA-IP processing: {e}")



        ui = np.nan_to_num(np.asarray(yi.real, dtype=np.float32))
        stoi_ideal = np.mean([stoi(u_GT[:, j], ui[:, j], fs) for j in range(J)])
        pesq_ideal = calc_psq(u_GT, ui)

        if print_scores:
            print(f'Ideal mask SDRi and SI-SDRi: {SDRi:.2f} {sisdri:.2f}')
            # print(f'Ideal RTF estimated SDRii and SI-SDRii: {SDRii:.2f} {sisdrii:.2f}')
            # print(f'AuxIVA-IP SI-SDR: {sisdr_aux}')


    # Select C_P based on beamforming type
    C_P = None
    if CFG.beamformer_type[-1] == 'C':
        calib = np.load('calib.npz')
        if P_method == 'vertices':
            C_suffix = ''
        else:
            C_suffix = key_suffix
        lhat_P = calib[f'lhats_{C_suffix}'][1]
        C_P = P > lhat_P

    # Compute MD and FA
    MD, FA, Err = MaskErr(Tmask, Emask, J)

    # Compute SDR, sisdr, ym
    SDR, sisdr, ym = beamformer_nonoise(Xt, Emask, xqf, J, fh2, olap, lens, 0.01, 1, CFG.fs, CFG.att, C=C_P)

    # Compute STOI and PESQ
    um = np.nan_to_num(np.asarray(ym.real, dtype=np.float32))

    stoi_score = np.mean([stoi(u_GT[:, j], um[:, j], fs) for j in range(J)])
    pesq_score = calc_psq(u_GT, um)

    if print_scores:
        # print(f'{global_name} {local_name} MD and FA: {MD:.2f} {FA:.2f}')
        print(f'{P_method} {local_method} SDR and SI-SDR: {SDR:.2f} {sisdr:.2f}')
        # print(f'{global_name} {local_name} STOI and PESQ: {stoi_score:.3f}, {pesq_score:.3f}')

    # Return computed values in dictionary
    if compute_ideal:
        scores = {
            f'L2_P_ideal': 0, f'Err_ideal': 0,
            f"MD_ideal": 0, f"FA_ideal": 0,
            f"stoi_ideal": stoi_ideal, f"pesq_ideal": pesq_ideal,
            "SDR_ideal": SDRi, "si-sdr_ideal": sisdri,
            "SDRii": 0, "si-sdrii": 0,
            f'L2_P_Aux': 0, f'Err_Aux': 0,
            f"MD_Aux": 0, f"FA_Aux": 0,
            # f"SDR_Aux": SDR_aux, f"si-sdr_Aux": sisdr_aux,
            # f"stoi_Aux": stoi_aux, f"pesq_Aux": pesq_aux,

            f'L2_P_{key_suffix}': L2, f'Err_{key_suffix}': Err,
            f"MD_{key_suffix}": MD, f"FA_{key_suffix}": FA, f"SDR_{key_suffix}": SDR, f"si-sdr_{key_suffix}": sisdr,
            f"stoi_{key_suffix}": stoi_score, f"pesq_{key_suffix}": pesq_score
        }
    else:
        scores = {f'L2_P_{key_suffix}': L2, f'Err_{key_suffix}': Err,
            f"MD_{key_suffix}": MD, f"FA_{key_suffix}": FA, f"SDR_{key_suffix}": SDR, f"si-sdr_{key_suffix}": sisdr,
            f"stoi_{key_suffix}": stoi_score, f"pesq_{key_suffix}": pesq_score
        }
    return scores


def calc_needed_audio_scores(dict_list, pr2, fh, Tmask, Hl, Xt, Hq, Xq, xqf, J=CFG.Q, show_best_local=True, show_best_global=True, print_scores=True):
    results = {}
    first_run = True
    metrics = ["L2_P", "Err", "MD", "FA", "SDR", "si-sdr", "stoi", "pesq"]
    for d in dict_list:
        P, local_mask, fh2 = d['P'], d['local_mask'], d['fh2']
        P_method, local_method = d['P_method'], d['local_method']

        scores = audio_scores(
            pr2=pr2,  # Assuming pr2 is not needed for this function
            P=P, J=J,
            Tmask=Tmask,  # Assuming Tmask is handled inside audio_scores
            Emask=local_mask,
            Hl=Hl, Xt=Xt, Hq=Hq, Xq=Xq, xqf=xqf, fh2=fh2, fh=fh,
            compute_ideal=first_run,  # Compute ideal only for the first run
            P_method=P_method,
            local_method=local_method,
            print_scores=print_scores
        )

        first_run = False  # Ensure compute_ideal is only True for the first call
        results.update(scores)

    if show_best_local:
        best_local_key = "best_global_deep"
        for metric in metrics:
            key_vertices = f"{metric}_vertices_deep"
            key_prob = f"{metric}_prob_deep"
            if metric in ["L2_P", "MD", "FA", "Err"]:
                results[f"{metric}_{best_local_key}"] = min(results.get(key_vertices, float('inf')),
                                                            results.get(key_prob, float('inf')))
            else:
                results[f"{metric}_{best_local_key}"] = max(results.get(key_vertices, float('-inf')),
                                                            results.get(key_prob, float('-inf')))

    if show_best_global:
        best_global_key = "best_global_NN"
        for metric in metrics:
            key_vertices = f"{metric}_vertices_NN"
            key_prob = f"{metric}_prob_NN"
            if metric in ["L2_P", "MD", "FA", "Err"]:
                results[f"{metric}_{best_global_key}"] = min(results.get(key_vertices, float('inf')),
                                                             results.get(key_prob, float('inf')))
            else:
                results[f"{metric}_{best_global_key}"] = max(results.get(key_vertices, float('-inf')),
                                                             results.get(key_prob, float('-inf')))

    return results



def calc_psq(ui, um, default_val=1.5, fs=CFG.old_fs):
    psq_vals = []
    for i in range(ui.shape[1]):
        try:
            psq = pesq(fs, ui[:, i], um[:, i], 'nb')  # Try computing PESQ
            psq_vals.append(psq)
        except Exception as e:
            print(f"Skipping channel {i} due to PESQ error: {e}")
            psq_vals.append(default_val)  # Assign default value instead of crashing
    return np.mean(psq_vals)


def plot_metrics(train_losses, train_rmses, train_sads, val_losses, val_rmses, val_sads):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 5))

    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # Plot RMSE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_rmses, label='Train RMSE')
    plt.plot(epochs, val_rmses, label='Val RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('RMSE')
    plt.legend()

    # Plot SAD
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_sads, label='Train SAD')
    plt.plot(epochs, val_sads, label='Val SAD')
    plt.xlabel('Epochs')
    plt.ylabel('SAD')
    plt.title('SAD')
    plt.legend()

    plt.tight_layout()
    plt.show()

def cosine_sim1(P):
    b, L, cols = P.size()

    cos_sims = []
    col_indices = list(range(cols))
    permutations = list(itertools.permutations(col_indices))

    for perm in permutations:
        P_permuted = P[:, :, perm]

        P_permuted_reshaped = P_permuted.view(b * cols, L, 1)

        P_reshaped = P_permuted.view(b * cols, L, 1)
        P_norm = torch.sqrt(torch.bmm(P_reshaped.transpose(1, 2), P_reshaped).view(b, cols, 1))

        permuted_norm  = torch.sqrt(torch.bmm(P_permuted_reshaped.transpose(1, 2), P_permuted_reshaped).view(b, cols, 1))

        summation = torch.bmm(P_reshaped.transpose(1, 2), P_permuted_reshaped).view(b, cols, 1)
        cos_sim = summation / (P_norm * permuted_norm)
        cos_sims.append(cos_sim.mean())

    return sum(cos_sims)


def cosine_sim(P):
    b, L, cols = P.size()


    P_norm = torch.norm(P, dim=1, keepdim=True)
    P_normalized = P / (P_norm + 1e-8)  # Avoid division by zero

    # Initialize tensor to accumulate cosine similarities
    cos_sim_sum = torch.zeros(b, device=P.device)

    # Compute pairwise cosine similarities for each batch
    for i in range(cols):
        for j in range(i + 1, cols):
            cos_sim = torch.sum(P_normalized[:, :, i] * P_normalized[:, :, j], dim=1)
            cos_sim_sum += cos_sim
    num_pairs = cols * (cols - 1) / 2
    cos_sim_avg = cos_sim_sum / num_pairs

    return cos_sim_avg.mean()




def compute_rmse(x_true, x_pre):
    squared_diff = (x_true - x_pre) ** 2

    mean_mse = torch.mean(squared_diff, dim=(1,2))
    mean_rmse = torch.sqrt(mean_mse)

    return torch.mean(mean_rmse) ## Average over batch

def calc_calibration_loss(P, pr2, lambdas=np.arange(0, 0.2, 0.01).tolist()):
    loss = np.zeros((CFG.N_frames, len(lambdas)))
    for i, lam in enumerate(lambdas):
        C = P > lam
        C_true = pr2 > 0

        FN = np.mean(1 - np.sum(C * C_true, axis=1) / (C_true.sum(axis=1) + 1e-8))

        FP = np.mean(np.sum(C * (~C_true), axis=1) / np.maximum(1, (~C_true).sum(axis=1)))

        loss[:, i] = FN

    return loss

def get_lhat(calib_loss_table, lambdas, alpha, B=1):
    n = calib_loss_table.shape[0]
    rhat = calib_loss_table.mean(axis=0)
    lhat_idx = max(np.argmax(((n/(n+1)) * rhat + B/(n+1) ) >= alpha) - 1, 0) # Can't be -1.
    return lambdas[lhat_idx], lhat_idx

def calibration(loss_list, alpha_list=[0.05, 0.1, 0.15], lambdas=np.arange(0, 0.2, 0.01).tolist()):
    # Extract individual loss lists
    loss_dict_keys = loss_list[0].keys()  # Get keys like 'C_P_loss', 'C_P2_loss', 'C_pe_loss'
    calib_results = {}

    lhats_dict = {}  # Dictionary to store lhats for each loss type

    for key in loss_dict_keys:  # Loop through 'C_P_loss', 'C_P2_loss', 'C_pe_loss'
        # Collect the corresponding losses for this key
        loss_matrix = [loss_dict[key] for loss_dict in loss_list]  # Extract loss type
        calib_loss_table = np.concatenate(loss_matrix, axis=0)  # Combine over runs

        # Compute mean loss per lambda
        # for i, lam in enumerate(lambdas):
        #     print(f"{key} - Lambda {lam}: {calib_loss_table[:, i].mean()}")

        # Compute lhat values for different alpha levels
        lhats = np.zeros(len(alpha_list))
        for j, alpha in enumerate(alpha_list):
            lhats[j], lhat_idx = get_lhat(calib_loss_table, np.array(lambdas), alpha, B=1)
            print(f"{key} - Alpha {alpha}: {lhats[j]}")

        # Store results in a dictionary
        lhats_dict[key] = lhats  # Save lhats for this loss type
        calib_results[key] = {'lhats': lhats, 'alpha_list': alpha_list}

    # Save all lhats in a single file
    np.savez('calib.npz', lhats_P=lhats_dict['C_P_loss'], lhats_P2=lhats_dict['C_P2_loss'], lhats_pe=lhats_dict['C_pe_loss'], alpha_list=np.array(alpha_list))

    return calib_results

def save_np_arrays(array_list, names_list):
    for filename, arr in zip(names_list, array_list):
        np.save(f'example_data/{filename}.npy', arr)
def load_np_arrays(filenames):
    arrays_dict = {filename: np.load(f'example_data/{filename}.npy')for filename in filenames}
    return arrays_dict

def load_all_dicts(folder_path="array_data"):

    dict_list = []

    # List all .pkl files in the folder
    pkl_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pkl")])

    # Load each .pkl file into a dictionary and add it to the list
    for pkl_file in pkl_files:
        file_path = os.path.join(folder_path, pkl_file)
        try:
            loaded_dict = joblib.load(file_path)
            dict_list.append(loaded_dict)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")

    return dict_list


def plot_masks(Tmask, deep_mask, Emask=None, P_method=CFG.P_method):


    vmin, vmax = 0, CFG.Q  # Set limits for the colormap
    cmap = plt.get_cmap("viridis", CFG.Q + 1)  # Discrete colormap
    norm = mcolors.BoundaryNorm(boundaries=np.arange(vmin, vmax + 2) - 0.5, ncolors=CFG.Q + 1)
    if Emask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 20))  # Create three subplots for masks
    else:
        fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    # Plot Tmask
    im1 = axes[0].imshow(Tmask, aspect='auto', cmap=cmap, norm=norm)
    axes[0].set_title("True Mask", fontsize=50)
    axes[0].tick_params(axis='both', which='major', labelsize=40)


    # Plot deep_mask with P_method
    im2 = axes[1].imshow(deep_mask, aspect='auto', cmap=cmap, norm=norm)
    axes[1].set_title(f"Deep Mask ({P_method})", fontsize=50)
    axes[1].tick_params(axis='both', which='major', labelsize=40)

    if Emask is not None:
        # Plot Emask
        im3 = axes[2].imshow(Emask, aspect='auto', cmap=cmap, norm=norm)
        axes[2].set_title("Estimated Mask", fontsize=50)
        axes[2].tick_params(axis='both', which='major', labelsize=40)

    # Add colorbar separately
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Manually set colorbar position
    cbar = fig.colorbar(im2, cax=cbar_ax, ticks=np.arange(vmin, vmax + 1))
    cbar.set_label("Mask Value", fontsize=40)
    cbar.ax.tick_params(labelsize=40)

    plt.subplots_adjust(right=0.85)  # Adjust space to fit the colorbar
    plt.show()
#
def calc_ilrma(Xt, y, J=CFG.Q, NFFT=CFG.NFFT, olap=CFG.olap, fs=CFG.fs):
    Y_stft = ilrma(Xt[:,:,:J], n_iter=100, n_src=y.shape[-1])
    ym = np.zeros_like(y)
    for q in range(J):

        ym[:, q] = signal.istft(Y_stft[:, :, q], nperseg=CFG.NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1][:ym[:, q].shape[0]]


    # Convert to Torch tensors
    y_torch = torch.from_numpy(np.real(y)).T  # (J, time)
    ym_torch = torch.from_numpy(np.real(ym)).T  # (J, time)

    permutations = list(itertools.permutations(range(J)))

    max_sisdr = float('-inf')
    for perm in permutations:
        perm_speakers_ym = ym_torch[perm, :]
        sisdr = t_audio_SI_SDR(perm_speakers_ym, y_torch).mean().item()

        if sisdr > max_sisdr:
            max_sisdr = sisdr
            best_y = perm_speakers_ym

    SDR = t_audio_SDR(best_y, y_torch).mean().item()
    ui = np.nan_to_num(best_y.numpy().T)
    um = np.nan_to_num(y_torch.numpy().T)
    st = np.mean([stoi(ui[:, j], um[:, j], fs) for j in range(J)])
    pesq = calc_psq(ui, um)
    return SDR, max_sisdr, st, pesq

def calc_auxip(Xt, y, J=CFG.Q, NFFT=CFG.NFFT, olap=CFG.olap, fs=CFG.fs):
    aux = torchiva.AuxIVA_IP(n_iter=30, n_src=y.shape[-1])
    Xt = Xt.transpose(2, 0, 1)
    Y_stft = aux(torch.from_numpy(Xt))
    Y_stft = Y_stft.permute(1, 2, 0)
    ym = np.zeros_like(y)
    for q in range(J):
        Y_stft[:, :, q]
        ym[:, q] = signal.istft(Y_stft[:, :, q], nperseg=CFG.NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1][:ym[:, q].shape[0]]


    # Convert to Torch tensors
    y_torch = torch.from_numpy(np.real(y)).T  # (J, time)
    ym_torch = torch.from_numpy(np.real(ym)).T  # (J, time)

    permutations = list(itertools.permutations(range(J)))

    max_sisdr = float('-inf')
    for perm in permutations:
        perm_speakers_ym = ym_torch[perm, :]
        sisdr = t_audio_SI_SDR(perm_speakers_ym, y_torch).mean().item()

        if sisdr > max_sisdr:
            max_sisdr = sisdr
            best_y = perm_speakers_ym

    SDR = t_audio_SDR(best_y, y_torch).mean().item()
    ui = np.nan_to_num(best_y.numpy().T)
    um = np.nan_to_num(y_torch.numpy().T)
    st = np.mean([stoi(ui[:, j], um[:, j], fs) for j in range(J)])
    pesq = calc_psq(ui, um)
    return SDR, max_sisdr, st, pesq

def calc_auxip2(Xt, y, J=CFG.Q, NFFT=CFG.NFFT, olap=CFG.olap, fs=CFG.fs):
    aux = torchiva.AuxIVA_IP2(n_iter=30)
    Xt = Xt.transpose(2, 0, 1)

    Y_stft = aux(torch.from_numpy(Xt[:J,:,:]))
    Y_stft = Y_stft.permute(1, 2, 0)
    ym = np.zeros_like(y)
    for q in range(J):
        Y_stft[:, :, q]
        ym[:, q] = signal.istft(Y_stft[:, :, q], nperseg=CFG.NFFT, noverlap=olap * NFFT, nfft=NFFT, fs=fs)[1][:ym[:, q].shape[0]]


    # Convert to Torch tensors
    y_torch = torch.from_numpy(np.real(y)).T  # (J, time)
    ym_torch = torch.from_numpy(np.real(ym)).T  # (J, time)

    permutations = list(itertools.permutations(range(J)))

    max_sisdr = float('-inf')
    for perm in permutations:
        perm_speakers_ym = ym_torch[perm, :]
        sisdr = t_audio_SI_SDR(perm_speakers_ym, y_torch).mean().item()

        if sisdr > max_sisdr:
            max_sisdr = sisdr
            best_y = perm_speakers_ym

    SDR = t_audio_SDR(best_y, y_torch).mean().item()
    ui = np.nan_to_num(best_y.numpy().T)
    um = np.nan_to_num(y_torch.numpy().T)
    st = np.mean([stoi(ui[:, j], um[:, j], fs) for j in range(J)])
    pesq = calc_psq(ui, um)
    return SDR, max_sisdr, st, pesq

if __name__ == '__main__':
    pe_complete = np.load('experiments_results/pe_complete.npy')
    pr2_complete = np.load('experiments_results/pr2_complete.npy')
    P2_complete = np.load('experiments_results/P2_complete.npy')
    E0_complete = np.load('experiments_results/E0_complete.npy')
    ext0_complete = np.load('experiments_results/ext0_complete.npy')
    t = np.load('experiments_results/t.npy')
    pe_incomplete = np.load('experiments_results/pe_incomplete.npy')
    pr2_incomplete = np.load('experiments_results/pr2_incomplete.npy')
    P2_incomplete = np.load('experiments_results/P2_incomplete.npy')
    E0_incomplete = np.load('experiments_results/E0_incomplete.npy')
    ext0_incomplete = np.load('experiments_results/ext0_incomplete.npy')
    J=3
    plot3d_simplex(E0_incomplete[:, :J], ext0_incomplete[0], title='', vector_type='u',
                   elev=30, azim=140)
    j = 0
    plot_name = f'speaker{j+1}_methods_comparison'
    plt.plot(t[0:62], P2_incomplete[200:262, j], color='blue', label='Deep Method')
    plt.plot(t[0:62], pr2_incomplete[200:262, j], color='red', label='Ideal Method')
    plt.plot(t[0:62], pe_incomplete[200:262, j], color='green', label='Standard Method')
    plt.xlabel('Time [s]', fontsize=15)
    plt.ylabel('Global Probabilities', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.savefig(f'experiments_results/{plot_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.show()
