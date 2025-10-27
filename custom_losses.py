import CFG
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from beamforming import beamformer_torch

from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore as DNSMOS
from scipy.spatial import distance_matrix
from haaqi_net.src.HAAQI_Net import HAAQI_Net_setup
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


class SAD(nn.Module):
    """
        Sum of Angular Distances (SAD) loss between columns of PP^T and W.
    """
    def __init__(self, Jplus1=CFG.Q+CFG.add_noise, epsilon=1e-10):
        super(SAD, self).__init__()
        self.epsilon = epsilon
        self.Jplus1 = Jplus1

    def forward(self, out, target):
        try:
            b, L, cols = out.size()
            out_reshaped = out.reshape(b * cols, L, 1)
            target_reshaped = target.reshape(b * cols, L, 1)

            an = torch.norm(out_reshaped, dim=1)
            bn = torch.norm(target_reshaped, dim=1)
            ab = torch.sum(out_reshaped * target_reshaped, dim=1)
            normalized_mult = ab / (an * bn + 1e-12)
            ang = torch.acos(normalized_mult + 1e-12)
            w_ang = (ang * bn).mean()
            m_w = torch.pi * bn.max()
            normalized_angle = (w_ang - 0) / (m_w - 0)
        except ValueError:
            return 0.0
        return normalized_angle

class NonZeroClipper(object):
    """Clamp all weights in a module to [1e-6, 1] range."""
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)
def log_cosh_loss(y_true, y_pred):
    """Log-cosh loss function (smooth L1)."""
    diff = y_pred - y_true
    return torch.mean(torch.log(torch.cosh(diff + 1e-12)))

class Unsupervised_Loss(nn.Module):
    """
        Unsupervised loss for global model combining SAD and L2 between PP^T and W.
    """
    def __init__(self, first_non0=0, SAD_factor = CFG.SAD_factor, L2_factor = CFG.L2_factor, P_method=CFG.P_method,
                 input_mask=None, noise_col=CFG.noise_col, noise_col_weight=CFG.noise_col_weight,):
        super(Unsupervised_Loss, self).__init__()
        self.name = 'SAD + mse'
        self.P_method = P_method
        self.first_non0 = first_non0
        self.noise_col = noise_col
        self.noise_col_weight = noise_col_weight
        self.input_mask = input_mask
        self.L2_loss = nn.MSELoss(reduction='mean')
        self.SAD_loss = SAD()
        self.L2_factor = L2_factor
        self.SAD_factor = SAD_factor
        self.SAD_losses = []
        self.L2_losses = []
        self.losses = []

    def forward(self, P_output, W_target, input_mask=None, W_output=None, E_output=None, epoch=None):
        """
                        Compute combined global loss.

                        Args:
                            P_output (torch.Tensor): [B, L, J] estimated global speaker probabilities.
                            W_target (torch.Tensor): [B, L, L] RTF correlation matrix.

                        Returns:
                            Tuple[loss, SAD_loss, L2_loss, PPT_output]
        """
        L = CFG.N_frames  # L
        J = P_output.size(2)
        P_scaled = P_output.clone()

        if self.noise_col:
            P_scaled[:, :, -1] *= self.noise_col_weight

        PPt_output = torch.bmm(P_scaled, P_scaled.transpose(1, 2))
        ## Enforcing P diagonal elements 1 to match W
        PPt_output[:, range(self.first_non0, CFG.N_frames), range(self.first_non0, CFG.N_frames)] = 1
        # PPt_output = PPt_output + torch.diag(torch.sum(((1 - P_scaled) * P_scaled), dim=-1).squeeze(0)).unsqueeze(0)


        if self.input_mask is not None:
            PPt_output = PPt_output * self.input_mask
        if input_mask is not None:
            PPt_output = PPt_output * input_mask


        loss1 = self.SAD_loss(PPt_output[:, self.first_non0:CFG.N_frames, self.first_non0:CFG.N_frames],
                                 W_target[:, self.first_non0:CFG.N_frames, self.first_non0:CFG.N_frames])


        loss2 = self.L2(PPt_output[:, self.first_non0:CFG.N_frames, self.first_non0:CFG.N_frames],
                      W_target[:, self.first_non0:CFG.N_frames, self.first_non0:CFG.N_frames])


        loss = self.SAD_factor * loss1 + self.L2_factor * loss2


        self.losses.append(loss.item())
        self.SAD_losses.append(loss1.item())
        self.L2_losses.append(loss2.item())

        return loss, loss1, loss2, PPt_output
    def L2(self,PPt_output, W):
        L = W.shape[2]
        return torch.sum((PPt_output - W)**2) / (L*L)

    def plot_loss(self):
        """Plot the progression of total loss, SAD loss, and RE loss."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.losses, label="Total Loss", linestyle="-")
        plt.plot(np.array(self.SAD_losses) * self.SAD_factor, label=f"SAD Loss * {self.SAD}", linestyle="--")
        plt.plot(np.array(self.L2_losses) * self.L2_factor, label=f"L2 Loss * {self.L2_factor}", linestyle=":")

        plt.xlabel("Iterations")
        plt.ylabel("Loss Value")
        plt.title("Loss Progression")
        plt.legend()
        plt.grid(True)
        plt.show()


class SupervisedLoss(nn.Module):
    """
        Supervised loss combining SAD and L2 between estimated P and target P. Used for aligning ground truth with estimation for calculating scores.
    """
    def __init__(self, J=CFG.Q, L2_factor=CFG.L2_factor, SAD_factor = CFG.SAD_factor, add_noise=CFG.add_noise):
        super(SupervisedLoss, self).__init__()
        self.name = 'SAD + L2'
        self.J = J
        self.L2_loss = nn.MSELoss(reduction='mean')
        self.SAD_loss = SAD()
        self.L2_factor = L2_factor
        self.SAD_factor = SAD_factor
        self.SAD_losses = []
        self.L2_losses = []
        self.losses = []
        self.add_noise = add_noise
    def forward(self, output, target):
        L = CFG.N_frames  # L


        loss1 = self.SAD_loss(output, target)

        loss2 = self.L2_loss(output, target)


        loss = self.SAD_factor * loss1 + self.L2_factor * loss2
        self.losses.append(loss.item())
        self.SAD_losses.append(loss1.item())
        self.L2_losses.append(loss2.item())


        return loss, loss1, loss2
    def plot_loss(self):
        """Plot the progression of total loss, SAD loss, and RE loss."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.losses, label="Total Loss", linestyle="-")
        plt.plot(np.array(self.SAD_losses) * self.SAD_factor, label=f"SAD Loss * {self.SAD}", linestyle="--")
        plt.plot(np.array(self.L2_losses) * self.L2_factor, label=f"L2 Loss * {self.L2_factor}", linestyle=":")

        plt.xlabel("Iterations")
        plt.ylabel("Loss Value")
        plt.title("Loss Progression")
        plt.legend()
        plt.grid(True)
        plt.show()


def find_best_permutation_supervised(loss_func, P_output, P_target):
    """
        Find the speaker permutation of P_output that best aligns with P_target.

        Args:
            loss_func (nn.Module): A supervised loss function.
            P_output (torch.Tensor): [B, L, J] predicted speaker probabilities.
            P_target (torch.Tensor): [B, L, J] ground truth speaker probabilities.

        Returns:
            Tuple: best loss, aligned output, best permutation, SAD loss, L2 loss
    """
    batch_size, L, J = P_output.size()
    min_loss = float('inf')
    min_SAD_loss = float('inf')
    min_RE_loss = float('inf')
    best_permutation = None

    # Generate all permutations of the column indices
    permutations = list(itertools.permutations(range(J)))

    for perm in permutations:
        # Permute the columns of P_output
        permuted_output = P_output[:, :, perm]

        # Compute the loss
        loss, SAD_loss, loss_RE = loss_func(permuted_output, P_target)

        # Check if this permutation yields a lower loss
        if loss < min_loss:
            min_loss = loss
            best_permutation = perm
            min_SAD_loss = SAD_loss
            min_RE_loss = loss_RE


    # Apply the best permutation
    best_P_output = P_output[:, :, best_permutation]

    return min_loss, best_P_output, best_permutation, min_SAD_loss, min_RE_loss

def find_best_permutation_supervised_local(mask_output, mask_target):
    batch_size, F, T, J = mask_output.size()
    min_loss = float('inf')
    best_permutation = None
    permutations = list(itertools.permutations(range(J)))

    for perm in permutations:
        # Safely permute speaker dimension
        permuted_output = mask_output[..., list(perm)]

        # Compute loss
        loss = Functional.mse_loss(permuted_output, mask_target)

        if loss < min_loss:
            min_loss = loss
            best_permutation = perm
    best_M_output = mask_output[:, :, :, best_permutation]
    return min_loss, best_M_output, best_permutation

def neg_si_sdr(preds, target):
    batch_size = target.shape[0]
    si_sdr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_sdr_val.view(batch_size, -1), dim=1)

def custom_CE(pred, target):
    epsilon = 1e-10  # To prevent log(0)
    target_sig = torch.sigmoid(target)
    pred_sig = torch.sigmoid(pred)
    bce_loss = - (target_sig * torch.log(pred_sig + epsilon) + (1 - target_sig) * torch.log(1 - pred_sig + epsilon))
    return torch.mean(bce_loss)

def center_reg(E, W):
    b = E.shape[0]
    m = torch.mean(E, dim=1, keepdim=True)
    # ones = torch.ones(E.shape[0], 1, E.shape[2]).to(CFG.device)
    # m_1 = torch.bmm(m, ones)
    m_1 = m.repeat(1, 626, 1)
    return Functional.mse_loss(E, m_1, reduction='sum')

def TV_reg(E):
    J = E.shape[2]
    ones_J = torch.ones(J,1).to(CFG.device)

    I_j = torch.eye(J).to(CFG.device)

    mean_mat = ones_J @ ones_J.T / J

    TV_diff = I_j - mean_mat
    loss_reg = torch.norm(E @ TV_diff, p='fro')

    return loss_reg


class LocalLoss(nn.Module):
    """
        Local loss for training SpatialNet: combines RTF covariance loss and global CE loss.
    """
    def __init__(self, F=CFG.lenF0, T=CFG.N_frames, C=(CFG.M-1)*2, J=CFG.Q, RTF_factor=CFG.RTF_factor, Emask=None,
                 soft_Emask=None, P=None, Xt=None, loss_names=['global', 'globalAVG', 'RTF', 'NN'], low_energy_mask=None,
                 global_factor=CFG.global_factor, globalAVG_factor=CFG.globalAVG_factor, weight_decay=1e-8,
                 epochs=CFG.epochs_local):

            super(LocalLoss, self).__init__()
            # self.RTF_factor = RTF_factor
            # self.global_factor = global_factor
            # self.globalAVG_factor = globalAVG_factor
            # self.weight_decay = weight_decay
            self.Emask = Emask
            self.soft_Emask = soft_Emask
            self.losses = []
            self.RTF_losses = []
            self.global_losses = []
            self.F, self.T, self.C, self.J = F, T, C, J
            self.NFFT = CFG.NFFT
            self.olap = CFG.olap
            self.epochs = epochs
            self.Xt = Xt
            self.P_original = P
            self.P_mask, self.fh_p = self.find_Pmask(low_energy_mask)
            self.y_P, self.Y_P = beamformer_torch(Xt.squeeze(), self.P_mask, self.fh_p, calc_istft=True, apply_mask=False)
            self.low_energy_mask = low_energy_mask
            # self.haaqinet = HAAQI_Net_setup(device=CFG.device)
            # self.hl = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0.]], device=CFG.device)
            # self.X_Encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
            #                                                 savedir="pretrained_models/spkrec-xvect-voxceleb", run_opts={'device':CFG.device})
            self.loss_names = loss_names
            self.name = 'loss' + ''.join(f'_{name}' for name in loss_names)

    def forward(self, mask_output, R, P, expanded_P=None, soft_Emask=None, epoch=None):
        losses = {}
        if 'global' in self.loss_names:
            losses['global'] = self.global_loss(mask_output, expanded_P)
        if 'globalAVG' in self.loss_names:
            losses['globalAVG'] = self.globalAVG_loss(mask_output, P)
        if 'RTF' in self.loss_names:
            losses['RTF'] = self.RTF_loss(mask_output, R)
        if 'NN' in self.loss_names:
            losses['NN'] = self.NN_loss(mask_output, self.Emask[:, :, :, :-1])
        if 'BF' in self.loss_names:
            losses['BF'] = self.BF_loss(self.Xt.squeeze(), mask_output.squeeze(), self.Y_P, self.fh_p, self.P_mask, low_energy_mask=self.low_energy_mask)

        if epoch == 0:
            self.factors = {}
            for name, val in losses.items():
                self.factors[name] = 10.0 / val.item()

        loss = sum(self.factors[name] * val for name, val in losses.items())

        if not CFG.param_search_flag and epoch is not None and epoch % 10 == 0:
            msg = f"Epoch {epoch + 1}/{self.epochs}, Total Loss: {loss.item():.4f}"
            for name, val in losses.items():
                msg += f", {name}: {val.item():.4f}"
            print(msg)
        return loss


    def RTF_loss(self, mask_output, R):

        mask_clone = mask_output.clone()
        if CFG.local_noise_col:
            mask_clone[:, :, :, -1] = 0

        mask_cov = torch.einsum('bflj,bfji->bfli', mask_clone, mask_clone.transpose(2, 3))
        R_cov = torch.einsum('bflh,bfhi->bfli', R, R.transpose(2, 3))

        # Enforce diag(M Mᵀ) = 1
        diag_idx = torch.arange(mask_cov.size(-1), device=mask_cov.device)
        mask_cov[:, :, diag_idx, diag_idx] = 1.0

        # diff_s = mask_cov.mean(dim=(0,1)) - R_cov.mean(dim=(0,1))
        # loss = torch.linalg.norm(diff_s, ord='fro')/ (self.L * self.L)

        diff_s = mask_cov - R_cov
        L2_loss = torch.linalg.norm(diff_s, ord='fro', dim=(2, 3)).sum()


        return L2_loss

    def global_loss(self, mask_output, P):

        # loss = -torch.einsum('bjt,bftj->bfj', P.transpose(1, 2), torch.log(mask_output + 1e-10)).sum()
        loss = Functional.kl_div(mask_output.log(), P, reduction="batchmean").squeeze().sum(-1)
        # diff = mask_output - P[:,None,:,:]
        # loss = torch.linalg.norm(diff) / (self.F)

        return loss

    def globalAVG_loss(self, mask_output, P):
        # loss = -(P * torch.log(mask_output.mean(dim=1) + 1e-10)).sum(dim=(1, 2))
        loss = Functional.kl_div(mask_output.mean(dim=1).log(), P, reduction="batchmean").squeeze().sum(-1)
        return loss

    def NN_loss(self, mask_output, mask_input):

        # diff_s = mask_output - mask_input
        L2_loss = Functional.mse_loss(mask_output, mask_input)
        # ce = -mask_input.clamp(min=1e-8, max=1.0) * torch.log(mask_output.clamp(min=1e-8, max=1.0))

        return L2_loss

    def find_Pmask(self, low_energy_mask=None, Np=10):

        Pmask = (self.J) * torch.ones((self.F , self.T), device=CFG.device)
        fq = []
        for j in range(self.J):
            _, top_indices = torch.topk(self.P_original[:, j], k=Np)
            fq.append(top_indices.detach().cpu())
            Pmask[:, fq[j]] = j
        Pmask[low_energy_mask, :] = 0
        return Pmask, fq

    def BF_loss(self, Xt, mask_output, Y_P, f_p, P_mask, low_energy_mask=None):
        M_mask = mask_output.argmax(-1)
        M_mask[low_energy_mask,:] = CFG.Q
        ym, Y_M = beamformer_torch(Xt, M_mask, f_p, calc_istft=True)

        # loss = Functional.mse_loss(Y_M.real, Y_P.real) + Functional.mse_loss(Y_M.imag, Y_P.imag)
        loss = -si_sdr(ym, self.y_P).mean()
        return loss

    def HAAQI_Net_loss(self, stft, mask_output, epoch=None):
        enhanced_stft = stft[0, :, :, :, None] * mask_output[0, :, :, None, :]
        n_fft = self.NFFT
        win_length = self.NFFT
        olap = self.olap
        hop_length = int(n_fft - olap * win_length)
        window = torch.hann_window(win_length, device=stft.device)

        enhanced_waveform = torch.istft(input=enhanced_stft[:, :, 0, :].permute(2, 0, 1), n_fft=n_fft,
                                        hop_length=hop_length,
                                        win_length=win_length, window=window, center=True, return_complex=False, ).float().T
        haaqinet_sum = 0
        target = torch.ones(1, device=CFG.device) * 0.7

        for audio in enhanced_waveform.T:
            haaqinet_sum += self.haaqinet(audio.unsqueeze(0), self.hl)[1][0]

        noisy_haqinet_sum = 0

        if epoch==0 or epoch==150:
            noisy_enhanced_waveform = add_noise(enhanced_waveform, snr_db=5)
            for audio in noisy_enhanced_waveform.T:
                noisy_haqinet_sum += self.haaqinet(audio.unsqueeze(0), self.hl)[1][0]

        loss = Functional.mse_loss(haaqinet_sum / self.J, target)

        return loss, haaqinet_sum / self.J, noisy_haqinet_sum / self.J

    def X_vectors_loss(self, stft, mask_output, J=CFG.Q, epoch=None):
        enhanced_stft = stft[0, :, :, :, None] * mask_output[0, :, :, None, :]
        n_fft = self.NFFT
        win_length = self.NFFT
        olap = self.olap
        hop_length = int(n_fft - olap * win_length)
        window = torch.hann_window(win_length, device=stft.device)

        enhanced_waveform = torch.istft(input=enhanced_stft[:, :, 0, :].permute(2, 0, 1), n_fft=n_fft,
                                        hop_length=hop_length,
                                        win_length=win_length, window=window, center=True, return_complex=False, ).float().T

        X_vectors = []
        for audio in enhanced_waveform.T:
            X_vectors.append(self.X_Encoder.encode_batch(audio).squeeze(0,1))
        X = torch.stack(X_vectors)

        pairs = list(itertools.combinations(range(J), 2))
        vec_i = []
        vec_j = []
        for i, j in pairs:
            vec_i.append(X[i])
            vec_j.append(X[j])
        vec_i = torch.stack(vec_i)
        vec_j = torch.stack(vec_j)

        # pairwise_L2 = ((vec_i - vec_j) ** 2)
        #
        # loss = torch.exp(-pairwise_L2).mean()
        margin = 0.2
        cos_sim = Functional.cosine_similarity(vec_i, vec_j, dim=1)  # ∈ [-1, 1]
        loss = torch.clamp(cos_sim - margin, min=0).mean()

        return loss

    def similarity_loss(self, mask_output, R, sigma = 1.0):
        B, F, T, C = R.shape
        R_norm = (R ** 2).sum(dim=-1, keepdims=True)
        R_dists = R_norm + R_norm.transpose(-2, -1) - 2 * torch.matmul(R, R.transpose(-2, -1))

        M_norm = (mask_output ** 2).sum(dim=-1, keepdims=True)
        M_dists = M_norm + M_norm.transpose(-2, -1) - 2 * torch.matmul(mask_output, mask_output.transpose(-2, -1))

        loss = Functional.mse_loss(torch.exp(-R_dists / (sigma ** 2 + 1e-10)), torch.exp(-M_dists / (sigma ** 2 + 1e-10)))
        return loss

    def smoothness_loss(self, mask_output):
        grad = mask_output[:, :, :, 1:] - mask_output[:, :, :, :-1]

        return torch.abs(grad).mean()

    def confidence_loss(self, mask_output):
        probs = mask_output / (mask_output.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy.mean()

    def plot_loss(self):
        """Plot the progression of total loss, RTF loss, and global loss."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.losses, label="Total Loss", linestyle="-")
        plt.plot(np.array(self.RTF_losses) * self.RTF_factor, label=f"RTF Loss * {self.RTF_factor}", linestyle="--")
        plt.plot(np.array(self.global_losses) * self.global_factor, label=f"Global Loss * {self.global_factor}", linestyle=":")

        plt.xlabel("Iterations")
        plt.ylabel("Loss Value")
        plt.title("Loss Progression")
        plt.legend()
        plt.grid(True)
        plt.show()





class CombinedLoss(nn.Module):
    def __init__(self, F=CFG.lenF0, T=CFG.N_frames, C=(CFG.M-1)*2, J=CFG.Q, RTF_factor=CFG.RTF_factor, Emask=None, soft_Emask=None,
                 global_factor=CFG.global_factor, globalAVG_factor=CFG.globalAVG_factor, weight_decay=1e-8, epochs=CFG.epochs_local,
                 first_non0=0, SAD_factor=CFG.SAD_factor, L2_factor=CFG.L2_factor, P_method=CFG.P_method, P_global_final=None,
                 input_mask=None, noise_col=CFG.noise_col, noise_col_weight=CFG.noise_col_weight, ):
            super(CombinedLoss, self).__init__()
            self.name = 'RTF_L2 + global_CE'
            # self.RTF_factor = RTF_factor
            # self.global_factor = global_factor
            # self.globalAVG_factor = globalAVG_factor
            # self.weight_decay = weight_decay
            self.SAD_factor = SAD_factor
            self.L2_factor = L2_factor
            self.L2_loss = nn.MSELoss(reduction='mean')
            self.SAD_loss = SAD()
            self.P_method = P_method
            self.first_non0 = first_non0
            self.Emask = Emask
            self.soft_Emask = soft_Emask
            self.losses = []
            self.RTF_losses = []
            self.global_losses = []
            self.F, self.T, self.C, self.J = F, T, C, J
            self.epochs = epochs
            self.P_global_final = P_global_final


    def forward(self, mask_output, R, W_target, P_output=None, epoch=None):

        if self.P_global_final:
            P_output = self.P_global_final

        PPt_output = torch.bmm(P_output, P_output.transpose(1, 2))
        PPt_output[:, range(self.first_non0, CFG.N_frames), range(self.first_non0, CFG.N_frames)] = 1

        loss_SAD = self.SAD_loss(PPt_output[:, self.first_non0:CFG.N_frames, self.first_non0:CFG.N_frames],
                              W_target[:, self.first_non0:CFG.N_frames, self.first_non0:CFG.N_frames])

        loss_L2 = self.L2_loss(PPt_output[:, self.first_non0:CFG.N_frames, self.first_non0:CFG.N_frames],
                        W_target[:, self.first_non0:CFG.N_frames, self.first_non0:CFG.N_frames])



        # loss1 = self.RTF_loss(mask_output, R)
        # loss2 = self.global_loss(mask_output, P_output)
        # loss3 = self.globalAVG_loss(mask_output, P_output)

        mask_input = self.Emask[:,:,:,:-1]
        mask_input, soft_mask_input = self.NN(R, P_output)
        loss4 = self.NN_loss(mask_output, mask_input)

        loss1 = torch.tensor(1.0, device=CFG.device)
        loss2 = torch.tensor(1.0, device=CFG.device)
        loss3 = torch.tensor(1.0, device=CFG.device)
        # loss4 = torch.tensor(1.0, device=CFG.device)


        if epoch == 0:
            self.L2_factor = 10 / loss_L2.item()
            self.SAD_factor = 10 / loss_SAD.item()

            self.RTF_factor = 10 / loss1.item()
            self.global_factor = 10 / loss2.item()
            self.globalAVG_factor = 10 / loss3.item()
            self.NN_factor = 10 / loss4.item()



        loss_global = self.SAD_factor * loss_SAD + self.L2_factor * loss_L2
        # loss_local = (self.RTF_factor * loss1 + self.global_factor * loss2 + self.globalAVG_factor * loss3 +
        #               self.NN_factor * loss4)
        loss_local = self.NN_factor * loss4

        loss = loss_local + loss_global


        self.losses.append(loss.item())


        if not CFG.param_search_flag:
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()},"
                      f" loss_global: {loss_global.item()},"
                      f" loss_local: {loss_local.item()},")

        return loss, loss_global, loss_local

    def RTF_loss(self, mask_output, R):

        mask_clone = mask_output.clone()
        if CFG.local_noise_col:
            mask_clone[:, :, :, -1] = 0

        mask_cov = torch.einsum('bflj,bfji->bfli', mask_clone, mask_clone.transpose(2, 3))
        R_cov = torch.einsum('bflh,bfhi->bfli', R, R.transpose(2, 3))

        # Enforce diag(M Mᵀ) = 1
        diag_idx = torch.arange(mask_cov.size(-1), device=mask_cov.device)
        mask_cov[:, :, diag_idx, diag_idx] = 1.0

        # diff_s = mask_cov.mean(dim=(0,1)) - R_cov.mean(dim=(0,1))
        # loss = torch.linalg.norm(diff_s, ord='fro')/ (self.L * self.L)

        diff_s = mask_cov - R_cov
        L2_loss = torch.linalg.norm(diff_s, ord='fro', dim=(2, 3)).sum()


        return L2_loss

    def global_loss(self, mask_output, P):

        loss = -torch.einsum('bjt,bftj->bfj', P.transpose(1, 2), torch.log(mask_output + 1e-10)).sum()
        # diff = mask_output - P[:,None,:,:]
        # loss = torch.linalg.norm(diff) / (self.F)

        return loss

    def globalAVG_loss(self, mask_output, P):
        loss = -(P * torch.log(mask_output.mean(dim=1) + 1e-10)).sum(dim=(1, 2))
        return loss

    def NN(self, R, P):
        P = P.squeeze(0)
        R = R.squeeze(0)
        dist = torch.cdist(R, R, p=2)
        affinity = torch.exp(-dist)
        speaker_weight = P.sum(dim=0, keepdim=True) + 1e-12
        decide = torch.matmul(affinity, P) / speaker_weight
        training_Emask = torch.argmax(decide, dim=-1)

        Emask_onehot = torch.zeros((1, self.F, self.T, self.J), device=CFG.device)
        Emask_onehot[0, np.arange(self.F)[:, None], np.arange(self.T), training_Emask.long()] = 1
        soft_training_Emask = None
        # soft_training_Emask = Functional.softmax(decide/0.01, dim=-1)

        return Emask_onehot, soft_training_Emask

    def NN_loss(self, mask_output, mask_input):

        # diff_s = mask_output - mask_input
        L2_loss = Functional.mse_loss(mask_output, mask_input)
        # ce = -mask_input.clamp(min=1e-8, max=1.0) * torch.log(mask_output.clamp(min=1e-8, max=1.0))

        return L2_loss



    def similarity_loss(self, mask_output, R, sigma = 1.0):
        B, F, T, C = R.shape
        R_norm = (R ** 2).sum(dim=-1, keepdims=True)
        R_dists = R_norm + R_norm.transpose(-2, -1) - 2 * torch.matmul(R, R.transpose(-2, -1))

        M_norm = (mask_output ** 2).sum(dim=-1, keepdims=True)
        M_dists = M_norm + M_norm.transpose(-2, -1) - 2 * torch.matmul(mask_output, mask_output.transpose(-2, -1))

        loss = Functional.mse_loss(torch.exp(-R_dists / (sigma ** 2 + 1e-10)), torch.exp(-M_dists / (sigma ** 2 + 1e-10)))
        return loss

    def smoothness_loss(self, mask_output):
        grad = mask_output[:, :, :, 1:] - mask_output[:, :, :, :-1]

        return torch.abs(grad).mean()

    def confidence_loss(self, mask_output):
        probs = mask_output / (mask_output.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy.mean()

    def plot_loss(self):
        """Plot the progression of total loss, RTF loss, and global loss."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.losses, label="Total Loss", linestyle="-")
        plt.plot(np.array(self.RTF_losses) * self.RTF_factor, label=f"RTF Loss * {self.RTF_factor}", linestyle="--")
        plt.plot(np.array(self.global_losses) * self.global_factor, label=f"Global Loss * {self.global_factor}", linestyle=":")

        plt.xlabel("Iterations")
        plt.ylabel("Loss Value")
        plt.title("Loss Progression")
        plt.legend()
        plt.grid(True)
        plt.show()




class GNNLoss(nn.Module):
    def __init__(self, F=CFG.lenF0, T=CFG.N_frames, C=(CFG.M - 1) * 2, J=CFG.Q, Emask=None,
                 soft_Emask=None, global_factor=CFG.global_factor, globalAVG_factor=CFG.globalAVG_factor, weight_decay=1e-8,
                 epochs=CFG.epochs_gnn, first_non0=0,  P_method=CFG.P_method, loss_names=['global', 'globalAVG', 'RTF', 'NN']):
        super(GNNLoss, self).__init__()

        # self.global_factor = global_factor
        # self.globalAVG_factor = globalAVG_factor
        self.L2_loss = nn.MSELoss(reduction='mean')
        self.P_method = P_method
        self.first_non0 = first_non0
        self.Emask = Emask

        self.losses = []
        self.F, self.T, self.C, self.J = F, T, C, J
        self.epochs = epochs

        self.loss_names = loss_names
        self.name = 'loss' + ''.join(f'_{name}' for name in loss_names)


    def forward(self, mask_output, P, R, soft_Emask=None, epoch=None):

        losses = {}
        if 'global' in self.loss_names:
            losses['global'] = self.global_loss(mask_output, P)
        if 'globalAVG' in self.loss_names:
            losses['globalAVG'] = self.globalAVG_loss(mask_output, P)
        if 'NN' in self.loss_names:
            losses['NN'] = self.NN_loss(mask_output, self.Emask[:,:,:,:-1])
        if 'RTF' in self.loss_names:
            losses['RTF'] = self.RTF_loss(mask_output, R)
        if 'pseudo' in self.loss_names:
            losses['pseudo'] = self.NN_loss(mask_output, soft_Emask)

        if epoch == 0:
            self.factors = {}
            for name, val in losses.items():
                self.factors[name] = 10.0 / val.item()

        loss = sum(self.factors[name] * val for name, val in losses.items())

        if not CFG.param_search_flag and epoch is not None and epoch % 10 == 0:
            msg = f"Epoch {epoch + 1}/{self.epochs}, Total Loss: {loss.item():.4f}"
            for name, val in losses.items():
                msg += f", {name}: {val.item():.4f})"
            print(msg)
        return loss

    def global_loss(self, mask_output, P):
        loss = -torch.einsum('bjt,bftj->bfj', P.transpose(1, 2), torch.log(mask_output + 1e-10)).sum()
        # diff = mask_output - P[:,None,:,:]
        # loss = torch.linalg.norm(diff) / (self.F)

        return loss

    def globalAVG_loss(self, mask_output, P):
        loss = -(P * torch.log(mask_output.mean(dim=1) + 1e-10)).sum(dim=(1, 2))
        return loss

    def NN_loss(self, mask_output, mask_input):

        # diff_s = mask_output - mask_input
        L2_loss = Functional.mse_loss(mask_output, mask_input)
        # ce = -mask_input.clamp(min=1e-8, max=1.0) * torch.log(mask_output.clamp(min=1e-8, max=1.0))

        return L2_loss

    def RTF_loss(self, mask_output, R):

        mask_clone = mask_output.clone()
        if CFG.local_noise_col:
            mask_clone[:, :, :, -1] = 0

        mask_cov = torch.einsum('bflj,bfji->bfli', mask_clone, mask_clone.transpose(2, 3))
        R_cov = torch.einsum('bflh,bfhi->bfli', R, R.transpose(2, 3))

        # Enforce diag(M Mᵀ) = 1
        diag_idx = torch.arange(mask_cov.size(-1), device=mask_cov.device)
        mask_cov[:, :, diag_idx, diag_idx] = 1.0

        # diff_s = mask_cov.mean(dim=(0,1)) - R_cov.mean(dim=(0,1))
        # loss = torch.linalg.norm(diff_s, ord='fro')/ (self.T * self.T)

        diff_s = mask_cov - R_cov
        L2_loss = torch.linalg.norm(diff_s, ord='fro', dim=(2, 3)).sum()


        return L2_loss


class GNNLoss_cleaner(nn.Module):
    def __init__(self, F=CFG.lenF0, T=CFG.N_frames, C=(CFG.M - 1) * 2, J=CFG.Q, Emask=None,
                 soft_Emask=None, global_factor=CFG.global_factor, globalAVG_factor=CFG.globalAVG_factor, weight_decay=1e-8,
                 epochs=CFG.epochs_gnn, first_non0=0,  P_method=CFG.P_method, loss_names=['global', 'globalAVG', 'RTF', 'NN']):
        super(GNNLoss_cleaner, self).__init__()

        # self.global_factor = global_factor
        # self.globalAVG_factor = globalAVG_factor
        self.L2_loss = nn.MSELoss(reduction='mean')
        self.P_method = P_method
        self.first_non0 = first_non0
        self.Emask = Emask

        self.losses = []
        self.F, self.T, self.C, self.J = F, T, C, J
        self.epochs = epochs

        self.loss_names = loss_names
        self.name = 'loss' + ''.join(f'_{name}' for name in loss_names)


    def forward(self, mask_output, P, R, soft_Emask=None, epoch=None):

        losses = {}
        if 'global' in self.loss_names:
            losses['global'] = self.global_loss(mask_output, P)
        if 'globalAVG' in self.loss_names:
            losses['globalAVG'] = self.globalAVG_loss(mask_output, P)
        if 'NN' in self.loss_names:
            losses['NN'] = self.NN_loss(mask_output, self.Emask[:,:,:,:-1])
        if 'RTF' in self.loss_names:
            losses['RTF'] = self.RTF_loss(mask_output, R)

        # if epoch == 0:
        #     self.factors = {}
        #     for name, val in losses.items():
        #         self.factors[name] = (10.0 / (val.detach() + 1e-10))
        loss = 0
        for name, val in losses.items():
            loss += val# * self.factors[name]


        # if not CFG.param_search_flag and epoch is not None and epoch % 10 == 0:
        #     msg = f"Epoch {epoch + 1}/{self.epochs}, Total Loss: {loss.sum().item():.4f}"
        #     for name, val in losses.items():
        #         msg += f", {name}: {val.sum().item():.4f})"
        #     print(msg)
        return loss

    def global_loss(self, mask_output, P):
        # loss = -(P * torch.log(mask_output + 1e-10)).sum(-1)
        # loss = Functional.mse_loss(mask_output, P, reduction='none').sum(-1)
        loss = Functional.kl_div(mask_output.log(), P.unsqueeze(1).expand(-1, mask_output.shape[1], -1, -1), reduction="none")
        return loss.sum(-1)

    def globalAVG_loss(self, mask_output, P):
        loss = Functional.kl_div(mask_output.mean(dim=1).log(), P, reduction="none").unsqueeze(1).expand(-1, mask_output.shape[1], -1, -1)
        return loss.sum(-1)

    def RTF_loss(self, mask_output, R):

        mask_clone = mask_output.clone()
        if CFG.local_noise_col:
            mask_clone[:, :, :, -1] = 0

        mask_cov = torch.einsum('bflj,bfji->bfli', mask_clone, mask_clone.transpose(2, 3))
        R_cov = torch.einsum('bflh,bfhi->bfli', R, R.transpose(2, 3))

        # Enforce diag(M Mᵀ) = 1
        diag_idx = torch.arange(mask_cov.size(-1), device=mask_cov.device)
        mask_cov[:, :, diag_idx, diag_idx] = 1.0


        loss = Functional.mse_loss(mask_cov, R_cov, reduction="none")

        return loss.sum(-1)

    def NN_loss(self, mask_output, mask_input):

        L2_loss = Functional.mse_loss(mask_output, mask_input, reduction='none').sum(-1)

        return L2_loss


        return L2_loss
def add_noise(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Adds white Gaussian noise to a signal at a specific SNR.

    Args:
        signal (torch.Tensor): Input signal, shape (..., T)
        snr_db (float): Desired signal-to-noise ratio in decibels

    Returns:
        torch.Tensor: Noisy signal
    """
    signal = signal.float()
    rms_signal = torch.sqrt(torch.mean(signal ** 2))
    snr_linear = 10 ** (snr_db / 10)
    rms_noise = rms_signal / torch.sqrt(torch.tensor(snr_linear, dtype=signal.dtype, device=signal.device))
    noise = torch.randn_like(signal) * rms_noise
    return signal + noise
