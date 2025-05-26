import CFG
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr

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
    def __init__(self, F=CFG.lenF0, L=CFG.N_frames, C=(CFG.M-1)*2, J=CFG.Q, RTF_factor=CFG.RTF_factor, global_factor=CFG.global_factor, weight_decay=1e-8):
            super(LocalLoss, self).__init__()
            self.name = 'RTF_L2 + global_CE'
            self.RTF_factor = RTF_factor
            self.global_factor = global_factor
            self.weight_decay = weight_decay

            self.losses = []
            self.RTF_losses = []
            self.global_losses = []
            self.F, self.L, self.C, self.J = F, L, C, J

    def forward(self, mask_output, R, P):
        """
                Compute combined local loss.

                Args:
                    mask_output (torch.Tensor): [B, F, L, J] estimated masks.
                    R (torch.Tensor): [B, F, L, C] RTF features.
                    P (torch.Tensor): [B, L, J] global speaker probabilities.

                Returns:
                    Tuple[loss, RTF_loss, global_loss]
        """
        loss1 = self.RTF_loss(mask_output, R)
        loss2 = self.global_loss(mask_output, P)

        loss = self.RTF_factor * loss1 + self.global_factor * loss2
        self.RTF_losses.append(loss1.item())
        self.global_losses.append(loss2.item())
        self.losses.append(loss.item())
        return loss, loss1, loss2

    def RTF_loss(self, mask_output, R):

        mask_clone = mask_output.clone()
        if CFG.local_noise_col:
            mask_clone[:, :, :, -1] = 0

        mask_cov = torch.einsum('bflj,bfji->bfli', mask_clone, mask_clone.transpose(2, 3))
        R_cov = torch.einsum('bflh,bfhi->bfli', R, R.transpose(2, 3))

        # Enforce diag(M Mᵀ) = 1
        diag_idx = torch.arange(mask_cov.size(-1), device=mask_cov.device)
        mask_cov[:, :, diag_idx, diag_idx] = 1.0

        diff_s = mask_cov - R_cov
        loss = torch.linalg.norm(diff_s, ord='fro', dim=(2, 3)).sum() / (self.F * self.L * self.C)
        return loss

    def global_loss(self, mask_output, P):
        loss = -(P * torch.log(mask_output.mean(dim=1) + 1e-10)).sum(dim=(1, 2)).mean() # / (self.F * self.L * self.J)
        return loss

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


