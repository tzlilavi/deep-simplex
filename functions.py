import itertools
import joblib
import os
import random

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from pesq import pesq
from pystoi import stoi
from scipy import signal
from scipy.spatial import distance_matrix
from sklearn.metrics import mean_squared_error

import CFG
import torchiva
from beamforming import beamformer, bss_eval_sources, MVDR_over_speakers
from custom_losses import find_best_permutation_supervised, SupervisedLoss
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as t_audio_SI_SDR
from torchmetrics.functional.audio import signal_distortion_ratio as t_audio_SDR
from utils import throwlow, findextremeSPA, FeatureExtr, smoother

random.seed(CFG.seed0)
np.random.seed(CFG.seed0)


def feature_extraction(Xt):
    """
    Extracts features from the multichannel STFT signal.

    Args:
        Xt (np.ndarray): STFT matrix of shape [freq_bins, time_frames, num_mics]

    Returns:
        Hl (np.ndarray): Full set of features.
        Hlm (np.ndarray): Mean-normalized features or similar (from FeatureExtr).
        Hlf (np.ndarray): Normalized feature tensor used in model input [freq x time x features].
        Fall (np.ndarray): Indices of selected frequency bins across microphones.
        lenF (int): Number of selected frequencies in band [f1, f2].
        F (np.ndarray): Array of frequency bin indices in the selected band.
    """

    d = 2 # Window length for local averaging used in FeatureExtr

    # Extract raw coherence features between mic 0 and others
    Hl, Hlm = FeatureExtr(Xt, CFG.F0, d)

    # Define frequency band of interest based on f1 and f2
    F = np.arange(
        int(np.ceil(CFG.f1 * CFG.NFFT / CFG.fs)),
        int(np.floor(CFG.f2 * CFG.NFFT / CFG.fs)) + 1
    )
    lenF = len(F)

    # Compute the flattened frequency indices to extract from Hl
    Fall = (
            np.tile(np.arange(0, CFG.lenF0 * (CFG.M - 2) + 1, CFG.lenF0), (lenF, 1))
            + np.tile(F.reshape(-1, 1), (1, CFG.M - 1))
    ).flatten()

    # Prepare normalized features across all frequencies
    Hlf = np.zeros((CFG.lenF0, CFG.N_frames, (CFG.M - 1) * 2), dtype=complex)

    for kk in range(CFG.NFFT // 2 + 1):
        Lb = 1
        Fk = np.array([kk])
        # Determine frequency indices for this bin
        Fall_f = (
                np.tile(np.arange(0, CFG.lenF0 * (CFG.M - 1), CFG.lenF0)[:, np.newaxis], (1, Lb))
                + np.tile(Fk[np.newaxis, :], (CFG.M - 1, 1))
        ).flatten()

        # Stack real and imaginary parts
        hlf = np.concatenate((np.real(Hl[Fall_f, :]), np.imag(Hl[Fall_f, :])), axis=0)

        # Normalize per time frame
        mnorm = np.sqrt(np.sum(hlf ** 2, axis=0))
        Hlf[kk, :, :] = (hlf / np.tile((mnorm + 1e-12), ((CFG.M - 1) * Lb * 2, 1))).T

    return Hl, Hlm, Hlf, Fall, lenF, F


def calculate_W_U_realSimplex(Hl, Fall, Tmask, lenF, F, J=CFG.Q, file=None, add_noise=CFG.add_noise):
    """
        Computes the normalized input matrix Hln, its outer product W (KK), its eigen-decomposition,
        and the initial probability estimates pr2.

        Args:
            Hl (np.ndarray): Coherence features matrix, shape [(M-1)*lenF, N_frames]
            Fall (np.ndarray): Frequency indices used to select features
            Tmask (np.ndarray): Ground truth time-frequency mask of shape [freq_bins, time_frames]
            lenF (int): Number of frequency bins in the selected range
            F (np.ndarray): Frequency bin indices used for Tmask
            J (int): Number of sources (default: CFG.Q)
            file: (Unused, legacy)
            add_noise (bool): Whether to add a noise column to pr2

        Returns:
            Hln (np.ndarray): Normalized real feature matrix, shape [features, time_frames]
            KK (np.ndarray): Input outer product matrix W = Hln.T @ Hln, shape [L, L]
            E0 (np.ndarray): Eigenvectors of KK
            pr2 (np.ndarray): Estimated global probabilities per frame (L x (J + noise))
            first_nonzero_l (int): First index where diagonal of KK > 0.9
    """

    # Replace NaNs with zeros
    Hl = np.nan_to_num(Hl, nan=0)
    # Build real feature matrix by concatenating real and imaginary parts of selected frequencies
    Hlf = np.concatenate((np.real(Hl[Fall, :]), np.imag(Hl[Fall, :])), axis=0)

    # Build W matrix (KK)
    mnorm = np.sqrt(np.sum(Hlf ** 2, axis=0))
    Hln = Hlf / (np.tile(mnorm + 1e-12, ((CFG.M - 1) * lenF * 2, 1)))
    Hln[:, 0] = np.mean(Hln[:, 0:], axis=1)
    KK = np.dot(Hln.T, Hln)

    diag = KK[range(CFG.N_frames), range(CFG.N_frames)]
    first_nonzero_l = np.where(diag > 0.9)[0][0]

    # Estimate probability matrix from Tmask (ground truth frame-level label proportions)
    pr2 = np.zeros((CFG.N_frames, J + CFG.add_noise))
    for q in range(J + add_noise):
        pr2[:, q] = np.sum(Tmask[F, :] == q, axis=0) / lenF

    # Eigen-decomposition of W to get E0 (U)
    de, E0 = np.linalg.eig(KK)
    d_sorted = np.sort(de)[::-1]
    E0 = E0[:, np.argsort(de)[::-1]]
    return Hln, KK, E0, pr2, first_nonzero_l


def calculate_SPA_simplex(E0, pr2, J, add_noise=CFG.add_noise):
    """
        Estimates the speaker probability matrix (pe) using the SPA (Successive Projection Algorithm)
        on the eigenvector matrix E0 and aligns it to the ground truth probability matrix pr2.

        Args:
            E0 (np.ndarray): Eigenvector matrix (NxJ).
            pr2 (np.ndarray): Ground truth probability matrix (Nx(J+1)).
            J (int): Number of sources (speakers).
            add_noise (bool): Whether to append a noise column (1 - row sum) to pe.

        Returns:
            pe (np.ndarray): Estimated speaker probability matrix after SPA and alignment.
            id0 (np.ndarray): Indices of the speakers most associated with each extracted vertex.
            ext0 (np.ndarray): Indices of the extracted extreme points in E0.
    """
    # Extract J extreme indices using SPA
    ext0, _ = findextremeSPA(E0[:, :J], J)

    # For each extreme point, find the index of the most dominant speaker (argmax over pr2)
    id0 = np.argmax(pr2[ext0, :J], axis=0)

    # Compute projection of E0 onto the subspace defined by the extracted extremes
    pe = np.dot(E0[:, :J], np.linalg.inv(E0[ext0, :J]))
    # Remove negative values and normalize rows that exceed simplex constraints
    pe = pe * (pe > 0)
    pe[pe.sum(1) > 1, :] = pe[pe.sum(1) > 1, :] / pe[pe.sum(1) > 1, :].sum(1, keepdims=True)
    pe = throwlow(pe)
    if add_noise:
        pe = np.hstack((pe, 1 - pe.sum(1, keepdims=True)))

    # Align estimated pe with ground truth pr2 using supervised loss and permutation matching
    loss_function = SupervisedLoss(L2_factor=CFG.L2_factor, SAD_factor=CFG.SAD_factor, J=J, add_noise=add_noise)
    _, pe, best_permutation, _, _ = find_best_permutation_supervised(loss_function, torch.from_numpy(pe).unsqueeze(0).float().to(
                                                                        CFG.device),
                                                                    torch.from_numpy(pr2).unsqueeze(0).float().to(
                                                                        CFG.device))
    pe = pe.cpu().numpy().squeeze(0)
    best_permutation = best_permutation[:J]
    ext0 = ext0[[best_permutation]]
    return pe, id0, ext0



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
    """
        Identifies the top Ne most active timeframes for each speaker (and optional noise)
        based on different estimated probability matrices.

        Args:
            pe (np.ndarray): Probability matrix from SPA (L x (J + noise)).
            P (np.ndarray): Probability matrix from the deep model (L x (J + noise)).
            P2 (np.ndarray): Secondary probability matrix (used when P_method == 'both').
            pr2 (np.ndarray): Ground truth probability matrix (L x (J + noise)).
            Ne (int): Number of top timeframes to return per speaker.
            P_method (str): Determines whether to also compute for P2.
            add_noise (bool): Whether the matrices include a noise column.

        Returns:
            fh2 (list of np.ndarray): Top Ne indices per speaker from P.
            fh22 (list of np.ndarray): Top Ne indices per speaker from P2 (only if P_method == 'both').
            fh2_pe (list of np.ndarray): Top Ne indices per speaker from pe (SPA).
            fh (list of np.ndarray): Top Ne indices per speaker from pr2 (ground truth).
        """
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
    """
        Computes local masks for each time-frequency bin using a nearest-neighbor approach
        based on cosine similarity in the feature space (Hlf). Applies to multiple methods.

        Args:
            pe (np.ndarray): SPA-estimated global probabilities (L x (J + noise)).
            P (np.ndarray): Deep model estimated global probabilities (L x (J + noise)).
            P2 (np.ndarray): Second deep model (used when P_method == 'both').
            pr2 (np.ndarray): Ground truth probabilities (unused here).
            Hlf (np.ndarray): Local features per frequency (F x L x D).
            Xt (np.ndarray): Input STFT matrix (not used here).
            low_energy_mask (np.ndarray): Boolean mask of low-energy timeframes.
            J (int): Number of speakers.
            f (np.ndarray): Frequency axis.
            t (np.ndarray): Time axis.
            P_method (str): One of 'vertices', 'prob', or 'both'.
            model_tested (str): Name of the global model (for plotting).
            Tmask (np.ndarray): Ground truth speaker mask.
            add_noise (bool): Whether there's an added noise speaker.
            plot_Emask (bool): Whether to plot the estimated masks.

        Returns:
            Emask (np.ndarray): Local mask (freq x time) for model P.
            Emask2 (np.ndarray): Local mask for model P2 (if P_method == 'both').
            Emask_pe (np.ndarray): Local mask for SPA model.
        """
    print('Running Nearest Neighbour Local Mapping...')


    # Local Mapping
    Emask = np.zeros((CFG.NFFT // 2 + 1, CFG.N_frames))
    Emask2 = np.zeros((CFG.NFFT // 2 + 1, CFG.N_frames))
    Emask_pe = np.zeros((CFG.NFFT // 2 + 1, CFG.N_frames))
    lenF0 = len(CFG.F0)

    for kk in range(CFG.NFFT // 2 + 1):
        # Compute similarity matrix using gaussian kernel based on Euclidean distances with a  in feature space
        pdistEE = distance_matrix(Hlf[kk,:,:], Hlf[kk,:,:])
        p_dist_local = np.exp(-pdistEE)

        # NN masks from the different models

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
    # Post-process to mask noise regions
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

    mse = mean_squared_error(pr2[:, :J], P[:, :J])

    return mse

def MaskErr(Tmask, Emask, Q):
    """
        Compute Miss Detection (MD), False Alarm (FA), and Error rate of estimated masks.

        Args:
            Tmask (np.ndarray): Ground truth [F, T] mask.
            Emask (np.ndarray): Estimated [F, T] mask.
            Q (int): Number of speakers.

        Returns:
            Tuple: MD (float), FA (float), Err (float)
    """
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
    """
        Computes separation quality metrics (MSE (L2), SDR, SI-SDR, PESQ, STOI, MaskErr, MD, FA)
        for a given global estimate P and corresponding local mask Emask.

        Args:
            pr2 (np.ndarray): Ground truth global probability matrix.
            P (np.ndarray): Estimated global probability matrix.
            Tmask (np.ndarray): Ground truth local time-frequency mask.
            Emask (np.ndarray): Estimated local mask.
            Hl, Xt, Hq, Xq, xqf: Input features, STFTs, and signals.
            fh2 (list): Indices of top frames by P.
            fh (list): Indices of top frames by ground truth.
            J (int): Number of speakers.
            compute_ideal (bool): Whether to compute ideal performance for reference.
            P_method (str): Global method used ('vertices' / 'prob').
            local_method (str): Local method used ('NN' / 'SpatialNet').
            print_scores (bool): Whether to print metrics.

        Returns:
            dict: Metrics computed for the method.
        """

    key_suffix = P_method + '_' + local_method
    olap, lens, att, fs = CFG.olap, CFG.lens, CFG.att, CFG.old_fs

    u_GT = np.nan_to_num(np.asarray(xqf[:, 0, :].real, dtype=np.float32))
    L2 = dist_scores(P, pr2, J, P_method)
    print(f'L2({P_method}, real): {L2:.4f}')

    # Compute ideal mask-based beamforming separation if requested
    SDRi, sisdri, yi = None, None, None
    if compute_ideal:
        # H_gt = compute_true_RTFs_from_Xq(Xq)
        # Compute ideal
        SDRi, sisdri, yi = beamformer(Xt, Tmask, xqf, J, fh, olap, lens, 0.0001, 1, CFG.fs, CFG.att,
                                              Hq = None)
        # === Optional AuxIVA baseline ===
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



    # Compute MD and FA
    MD, FA, Err = MaskErr(Tmask, Emask, J)

    # Compute SDR, sisdr, ym
    SDR, sisdr, ym = beamformer(Xt, Emask, xqf, J, fh2, olap, lens, 0.01, 1, CFG.fs, CFG.att)

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
    """
        Runs audio evaluation (SDR, PESQ, STOI, etc.) for a list of (P, mask) configurations.

        Args:
            dict_list (list): Each item includes 'P', 'local_mask', 'P_method', 'local_method', 'fh2'.
            pr2 (np.ndarray): Ground-truth global probability matrix.
            fh (list): Top-frame indices by ground truth (for beamforming).
            Tmask (np.ndarray): Ground-truth TF mask.
            Hl, Xt, Hq, Xq, xqf: Input features and signals.
            J (int): Number of sources.
            show_best_local (bool): Also compute best result between deep local methods.
            show_best_global (bool): Also compute best result between global NN methods.
            print_scores (bool): Print metric values.

        Returns:
            dict: All computed metric values, including best-of summaries.
    """
    results = {}
    first_run = True
    metrics = ["L2_P", "Err", "MD", "FA", "SDR", "si-sdr", "stoi", "pesq"]
    for d in dict_list:
        P, local_mask, fh2 = d['P'], d['local_mask'], d['fh2']
        P_method, local_method = d['P_method'], d['local_method']

        scores = audio_scores(
            pr2=pr2,
            P=P, J=J,
            Tmask=Tmask,
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

#
# if __name__ == '__main__':
#     pe_complete = np.load('experiments_results/pe_complete.npy')
#     pr2_complete = np.load('experiments_results/pr2_complete.npy')
#     P2_complete = np.load('experiments_results/P2_complete.npy')
#     E0_complete = np.load('experiments_results/E0_complete.npy')
#     ext0_complete = np.load('experiments_results/ext0_complete.npy')
#     t = np.load('experiments_results/t.npy')
#     pe_incomplete = np.load('experiments_results/pe_incomplete.npy')
#     pr2_incomplete = np.load('experiments_results/pr2_incomplete.npy')
#     P2_incomplete = np.load('experiments_results/P2_incomplete.npy')
#     E0_incomplete = np.load('experiments_results/E0_incomplete.npy')
#     ext0_incomplete = np.load('experiments_results/ext0_incomplete.npy')
#     J=3
#     plot3d_simplex(E0_incomplete[:, :J], ext0_incomplete[0], title='', vector_type='u',
#                    elev=30, azim=140)
#     j = 0
#     plot_name = f'speaker{j+1}_methods_comparison'
#     plt.plot(t[0:62], P2_incomplete[200:262, j], color='blue', label='Deep Method')
#     plt.plot(t[0:62], pr2_incomplete[200:262, j], color='red', label='Ideal Method')
#     plt.plot(t[0:62], pe_incomplete[200:262, j], color='green', label='Standard Method')
#     plt.xlabel('Time [s]', fontsize=15)
#     plt.ylabel('Global Probabilities', fontsize=15)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.legend()
#     plt.savefig(f'experiments_results/{plot_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
#     plt.show()
