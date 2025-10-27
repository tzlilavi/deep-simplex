import os
import random

import joblib
import numpy as np
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.func import functional_call
from tqdm import tqdm

import CFG
from local_models import SpatialNet, GCN, Lambda_MLP
from global_models import BiLSTM_Att, MiSiCNet2, AutoEncoder
from custom_losses import (LocalLoss, SupervisedLoss, Unsupervised_Loss, find_best_permutation_supervised, CombinedLoss, GNNLoss, GNNLoss_cleaner,
                           find_best_permutation_supervised_local)
from functions import audio_scores, local_mapping, plot_masks, throwlow, local_mapping, plot_adjacency_matrices, plot_mask_speakers
from torch_geometric.utils import to_undirected

torch.manual_seed(42)


def run_global_model(model, input, W, loss_function, num_epochs, lr=CFG.lr, max_norm=CFG.clip_grad_max,
                     betas=(0.9, 0.999), dropout=CFG.dropout, K_dropout=None,
                     param_search=CFG.param_search_flag, plot_loss=False,
                     noise_col=CFG.noise_col, add_noise=CFG.add_noise):
    """
        Train the global deep model using unsupervised loss. Supports optional dropout inference and masking.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=CFG.weight_decay, betas=betas)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)


    input = input.to(CFG.device)
    W_target = W.to(CFG.device)


    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    losses = []
    mask = None
    if CFG.pad_flag:
        W_target = F.pad(W_target, (0, CFG.pad_tfs, 0, CFG.pad_tfs))
        input = F.pad(input, (0, CFG.pad_tfs, 0, CFG.pad_tfs))

    for epoch in range(num_epochs):
        model.train()

        P_output, W_output, E_output, A_output = model(input, epoch)


        loss, loss_SAD, loss_RE, PPt_output = (
            loss_function(P_output, W_target, mask, W_output, E_output, epoch=epoch))


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=1)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if not param_search:
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, SAD_loss: {loss_SAD.item()},"
                      f" loss_RE: {loss_RE.item()}")


        if best_loss > loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test-time dropout averaging if configured
    if K_dropout and dropout > 0:
        P_output = MC_dropout_averaging(model, input, K = K_dropout)

    else:
        model.eval()
        P_output, W_output, E_output, A_output = model(input, epoch)
    if CFG.pad_flag:
        P_output = P_output[:, :-CFG.pad_tfs,:]
    model.eval()
    P_noise = None
    if noise_col:
        P_noise = P_output[:, :,-1].detach().cpu().numpy().squeeze(0)
        if not add_noise:
            P_output = P_output[:, :, :-1]
    return {
        'model_name': model.name,
        'lr': CFG.lr,
        'epochs': num_epochs,
        'loss_name': loss_function.name,
        'loss': round(loss.item(), 4),
        'output_mat': P_output.detach(),
        'P_torch': P_output.detach(),
        'A': A_output.detach(),
        'P_noise': P_noise
    }

def MC_dropout_averaging(model, input, K=5):
    """Average multiple forward passes with dropout enabled."""
    model.train()
    print(f'Running MC dropout averaging for K = {K}')
    with torch.no_grad():
        P_samples = [model(input)[0] for _ in range(K)]
    return torch.stack(P_samples).mean(dim=0)

def mask_input(input, mask_ratio=0.15):
    """Random binary mask for symmetric 1xLxL input matrix."""
    B, L, L = input.shape
    mask = torch.rand(B, L, L, device=input.device) > mask_ratio  # Randomly mask values
    return mask

def global_method(input_mat, W_torch, first_non0, pr2, low_energy_mask_time, J=CFG.Q, lr=CFG.lr, SAD_factor=CFG.SAD_factor,
                  L2_factor=CFG.L2_factor, P_method=CFG.P_method, param_search_flag=CFG.param_search_flag,
                  epochs=CFG.epochs, betas=CFG.betas, n_repeat_last_lstm=CFG.n_repeat_last_lstm, n_heads=CFG.n_heads,
                  seed=CFG.global_seed, Hlf=None, low_energy_mask=None,
                  dropout=CFG.dropout, K_dropout=None, run_multiple_initializations=CFG.run_multiple_initializations
                  , fixed_mask_input_ratio=CFG.fixed_mask_input_ratio,
                  noise_col=CFG.noise_col, add_noise=CFG.add_noise):
    """
        Initialize and train a global deep model. Optionally run multiple initializations and reorder outputs.
    """
    mask = None
    if fixed_mask_input_ratio:
        print('Masking input with fixed ratio...')
        mask = mask_input(input_mat, mask_ratio=fixed_mask_input_ratio)
        mask = mask.to(CFG.device)
        input_mat = input_mat * mask
        W_torch = W_torch * mask

    dim_output = J
    if noise_col:
        dim_output = dim_output + 1
        print('Adding a noise column...')
    if not run_multiple_initializations:
        model = BiLSTM_Att(dim_output=dim_output, P_method=P_method, n_repeat_last_lstm=n_repeat_last_lstm,
                        n_heads=n_heads, seed=seed ,low_energy_mask = low_energy_mask_time, dropout=dropout).to(CFG.device)
        loss_function = Unsupervised_Loss(first_non0=first_non0, P_method=P_method, input_mask=mask, noise_col=noise_col)
        deep_dict = run_global_model(model, input_mat, W_torch, loss_function, epochs, lr, param_search=param_search_flag, betas=betas,
                                     dropout=dropout, K_dropout=K_dropout,
                                     noise_col=noise_col, add_noise=add_noise)
    else:
        seeds = [seed, 0, 42]
        print(f'Running different initializations for {len(seeds)} times')
        dicts = []
        losses = []
        for seed_k in seeds:
            model = BiLSTM_Att(dim_output=dim_output, P_method=P_method, n_repeat_last_lstm=n_repeat_last_lstm,
                               n_heads=n_heads, seed=seed_k, low_energy_mask=low_energy_mask_time, dropout=dropout).to(CFG.device)
            loss_function = Unsupervised_Loss(first_non0=first_non0, P_method=P_method, input_mask=mask, noise_col=noise_col)
            print(f'Running Seed {seed_k}')
            d = run_global_model(model, W_torch, W_torch, loss_function, epochs, lr, param_search=param_search_flag,
                                         betas=betas, dropout=dropout, K_dropout=K_dropout, noise_col=noise_col)
            dicts.append(d)
            losses.append(d['loss'])
        print(f'Different init losses: {losses}')
        deep_dict = dicts[np.argmin(losses)]

    # Best output
    P = deep_dict['output_mat']

    # Align output speakers to compare to ideal pr2
    loss_function = SupervisedLoss(L2_factor=L2_factor, SAD_factor=SAD_factor)
    _, P, best_permutation, _, _ = find_best_permutation_supervised(loss_function, P.to(CFG.device), torch.from_numpy(pr2).unsqueeze(0).float().to(CFG.device))

    # Remove negative values and normalize rows that exceed simplex constraints
    P = P.cpu().numpy().squeeze(0)
    P = throwlow(P)
    P[P.sum(1) > 0, :] = P[P.sum(1) > 0, :] / P[P.sum(1) > 0, :].sum(1, keepdims=True)

    A = deep_dict['A'].detach().cpu().numpy()
    A = A[:, best_permutation]


    return deep_dict, P, A


def run_local_model(model, P, R, loss_function, num_epochs=CFG.epochs_local, lr=CFG.lr_local, max_norm=CFG.clip_grad_max, betas=CFG.betas,
                    param_search=CFG.param_search_flag, plot_loss=False):
    """
        Train the local model to predict soft masks from RTF pre frequency (Hlf) and global P.

        Args:
            model (nn.Module): Local model (e.g., SpatialNet).
            P (torch.Tensor): Global speaker probabilities [1, L, J].
            R (torch.Tensor): Real-valued RTF-like features [1, F, L, 2(M-1)].
            loss_function (nn.Module): Local loss.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate.

        Returns:
            dict: Dictionary with model outputs and metadata.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=CFG.weight_decay, betas=betas)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)

    F, T, C = R.shape
    _, J = P.shape
    P = P.to(CFG.device)
    expanded_P = P.expand(F, T, J)
    R = R.real.float().to(CFG.device)

    if CFG.random_local_input:
        uniform_noise = torch.distributions.Uniform(-0.1 ** 0.5, 0.1 ** 0.5).sample(R.shape)
        gaussian_noise = torch.normal(mean=0, std=0.03 ** 0.5, size=R.shape)
        input = (uniform_noise + gaussian_noise).to(CFG.device)
        print('Using random local input...')
    else:
        input = R.clone()

    patience = 30
    patience_counter = 0

    best_loss = float('inf')

    for epoch in range(num_epochs):

        model.train()

        _, mask_output = model(input)


        loss = loss_function(mask_output, R=R.unsqueeze(0), P=P.unsqueeze(0), expanded_P=expanded_P.unsqueeze(0),
                            epoch=epoch)


        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=1)
        optimizer.step()

        scheduler.step()


        if best_loss > loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    model.eval()
    _, mask_output = model(input)

    if plot_loss:
        loss_function.plot_loss()

    d = {}
    d['model_name'] = model.name
    d['lr'] = lr
    d['epochs'] = num_epochs
    d['loss_name'] = loss_function.name
    d['loss'] = round(loss.item(), 4)
    # d['early_loss'] = round(early_loss, 4)
    d['deep_mask'] = mask_output.detach()
    d['P_signals'] = loss_function.y_P.detach().cpu()
    d['P_signals_fh'] = loss_function.fh_p
    d['P_signals_mask'] = loss_function.P_mask.detach().cpu().numpy()
    return d

def deep_local_masking(Xt, P, Hlf, Tmask, Emask=None, soft_Emask=None, P_method=CFG.P_method, J=CFG.Q, plot_mask=False, plot_loss=False, lr=CFG.lr_local, betas=CFG.betas, RTF_factor=CFG.RTF_factor, global_factor=CFG.global_factor, epochs=CFG.epochs_local,
                       num_layers=CFG.num_layers, dim_squeeze=CFG.dim_squeeze, encoder_kernel_size=CFG.encoder_kernel_size, kernel_size=CFG.kernel_size, conv_groups=CFG.conv_groups,
                       param_search=CFG.param_search_flag, local_init_seed=CFG.local_init_seed, low_energy_mask=None):
    """
        Train SpatialNet to predict local TF masks from RTF features and global speaker probabilities.

        Returns:
            - dict: Training metadata and mask output.
            - np.ndarray: Soft mask [L, J].
            - np.ndarray: Hard mask [L] with speaker labels.
    """
    print("Running Deep Local Mapping...")
    Emask_onehot = torch.zeros((1, CFG.lenF0, CFG.N_frames, J + 1), device=CFG.device)
    Emask_onehot[0, np.arange(CFG.lenF0)[:, None], np.arange(CFG.N_frames), torch.from_numpy(Emask).long()] = 1
    if soft_Emask is not None:
        soft_Emask = torch.from_numpy(soft_Emask).float().to(CFG.device).unsqueeze(0)
    Xt = torch.from_numpy(Xt).to(CFG.device).unsqueeze(0)
    P = torch.from_numpy(P).to(CFG.device)
    local_loss = LocalLoss(RTF_factor=RTF_factor, global_factor=global_factor, Emask=Emask_onehot, soft_Emask=soft_Emask,
                           P=P, Xt=Xt, loss_names=['BF', 'RTF', 'globalAVG', 'global'], low_energy_mask=low_energy_mask).to(CFG.device)
    local_model = SpatialNet(num_layers=num_layers, dim_squeeze=dim_squeeze, encoder_kernel_size=encoder_kernel_size,
                             kernel_size=kernel_size, conv_groups=conv_groups, seed=local_init_seed, low_energy_mask=low_energy_mask).to(CFG.device)
    deep_dict_local = run_local_model(local_model, P, torch.from_numpy(Hlf), local_loss, lr=lr, betas=betas, plot_loss=plot_loss,
                                      param_search=param_search)

    deep_mask_soft = deep_dict_local['deep_mask'].squeeze(0).detach().cpu().numpy()
    deep_mask_hard = deep_mask_soft.argmax(axis=-1)

    deep_mask_hard[low_energy_mask] = J

    if plot_mask:
        plot_masks(Tmask, deep_mask_hard, Emask, P_method=P_method)

    return deep_dict_local, deep_mask_soft, deep_mask_hard, local_loss.name

def run_combined_model(W_torch, Hlf, P_global_final, low_energy_mask_time, low_energy_mask, pr2, Tmask, first_non0=0, num_epochs=CFG.epochs_combined, lr=CFG.lr,
                       max_norm=CFG.clip_grad_max, betas=(0.9, 0.999), Emask=None, soft_Emask=None, P_method=CFG.P_method, pe=None,
                       t=None, f=None, J=CFG.Q, Xt=None,
                       plot_mask=False):

    dim_output = J + 1 if CFG.noise_col else J

    global_model = BiLSTM_Att(dim_output=dim_output, P_method=P_method, n_repeat_last_lstm=CFG.n_repeat_last_lstm,
                              n_heads=CFG.n_heads, seed=CFG.global_seed, low_energy_mask=low_energy_mask_time,
                              dropout=CFG.dropout).to(CFG.device)

    local_model = SpatialNet(num_layers=CFG.num_layers, dim_squeeze=CFG.dim_squeeze, encoder_kernel_size=CFG.encoder_kernel_size,
                             kernel_size=CFG.kernel_size, conv_groups=CFG.conv_groups, seed=CFG.local_init_seed,
                             low_energy_mask=low_energy_mask).to(CFG.device)

    Emask_onehot = torch.zeros((1, CFG.lenF0, CFG.N_frames, J + 1), device=CFG.device)
    Emask_onehot[0, np.arange(CFG.lenF0)[:, None], np.arange(CFG.N_frames), torch.from_numpy(Emask).long()] = 1
    if soft_Emask is not None:
        soft_Emask = torch.from_numpy(soft_Emask).float().to(CFG.device).unsqueeze(0)
    combined_loss = CombinedLoss(F=CFG.lenF0, T=CFG.N_frames, C=(CFG.M - 1) * 2, J=CFG.Q, RTF_factor=CFG.RTF_factor,
                                 Emask=Emask_onehot, soft_Emask=soft_Emask, global_factor=CFG.global_factor,
                                 globalAVG_factor=CFG.globalAVG_factor, weight_decay=CFG.weight_decay,
                                 epochs=num_epochs, first_non0=first_non0, SAD_factor=CFG.SAD_factor,
                                 L2_factor=CFG.L2_factor, P_method=P_method, input_mask=None,
                                 noise_col=CFG.noise_col, noise_col_weight=CFG.noise_col_weight).to(CFG.device)

    optimizer = optim.Adam(list(global_model.parameters()) + list(local_model.parameters()),
                           lr=lr, weight_decay=CFG.weight_decay, betas=betas)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)

    W_torch = W_torch.to(CFG.device)
    Hlf = Hlf.unsqueeze(0).real.float().to(CFG.device)

    best_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(num_epochs):
        global_model.train(); local_model.train()
        P_output, W_output, E_output, A_output = global_model(W_torch, epoch)
        mask_output = local_model(Hlf)
        loss, loss_global, loss_local = combined_loss(mask_output, Hlf, W_torch, P_output, epoch=epoch)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(global_model.parameters()) + list(local_model.parameters()), max_norm)
        optimizer.step(); scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item(); patience_counter = 0
        else:
            patience_counter += 1;
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    global_model.eval(); local_model.eval()
    with torch.no_grad():
        P_output, _, _, A_output = global_model(W_torch)
        mask_output = local_model(Hlf)


    loss_function_global = SupervisedLoss(L2_factor=CFG.L2_factor, SAD_factor=CFG.SAD_factor)
    _, P_output, best_permutation, _, _ = find_best_permutation_supervised(loss_function_global, P_output.detach(),
                                                                    torch.from_numpy(pr2).unsqueeze(0).float().to(CFG.device))

    Tmask_onehot = torch.zeros((1, CFG.lenF0, CFG.N_frames, J + 1), device=CFG.device)
    Tmask_onehot[0, np.arange(CFG.lenF0)[:, None], np.arange(CFG.N_frames), torch.from_numpy(Tmask).long()] = 1
    _, mask_output, best_permutation_mask = find_best_permutation_supervised_local(mask_output, Tmask_onehot[:, :, :, :-1])


    # Remove negative values and normalize rows that exceed simplex constraints
    P_output = P_output.cpu().numpy().squeeze(0)
    P_output = throwlow(P_output)
    P_output[P_output.sum(1) > 0, :] = P_output[P_output.sum(1) > 0, :] / P_output[P_output.sum(1) > 0, :].sum(1, keepdims=True)

    deep_mask_soft = mask_output.squeeze(0).detach().cpu().numpy()
    deep_mask_hard = deep_mask_soft.argmax(axis=-1)
    # deep_mask_hard, _, _, deep_mask_soft = local_mapping(
    #     pe, P_output, None, pr2, Hlf.squeeze(0).detach().cpu().numpy(), Xt, low_energy_mask, J, f, t, P_method, 'asd', Tmask)
    # deep_mask_hard = deep_mask_hard.astype(int)
    if plot_mask:
        plot_masks(Tmask, deep_mask_hard, Emask, P_method=P_method)

    return P_output, A_output, deep_mask_soft, deep_mask_hard, loss


def calculate_adjancy_mat(feature_mat, Hlf=None, P=None, Tmask=None, k=10, sigma=1.0,
                          plot_flag=CFG.plot_flag, TH=0.5, method='KNN'):

    F, T, dim_input = feature_mat.shape
    batch_size = F
    count = F * T * T

    dist = torch.cdist(feature_mat, feature_mat, p=2)

    A_gauss = torch.exp(-(dist**2) / (2 * (sigma**2)))

    eye = torch.eye(T, dtype=torch.bool).unsqueeze(0)
    A_gauss = A_gauss.masked_fill(eye, 0)

    if method == 'KNN':
        vals, idx = A_gauss.topk(k, dim=-1)
        Kp = idx.shape[-1]
        expanded_T = (torch.arange(T)).expand(F, T)
        updated_idx = torch.cat([idx, (expanded_T + 1)[:, :, None], (expanded_T - 1)[:, :, None]], dim=-1)
        updated_idx = updated_idx.clamp(0, T - 1)
        updated_vals = torch.cat([vals, A_gauss.gather(-1, updated_idx[:, :, -2:])], dim=-1)


        A = torch.zeros_like(A_gauss)
        A.scatter_(dim=-1, index=updated_idx, src=updated_vals)

        f_idx = torch.arange(F)[:, None, None].expand_as(updated_idx)  # (F,T,Kp)
        t_idx = torch.arange(T)[None, :, None].expand_as(updated_idx)  # (F,T,Kp)

        src = (f_idx * T + t_idx).reshape(-1)  # (N_edges,)
        dst = (f_idx * T + updated_idx).reshape(-1)  # (N_edges,)
        edge_index = torch.stack([src, dst], dim=0)

        edge_weight = updated_vals.reshape(-1)



    elif method == 'TH':
        device = A_gauss.device
        num_nodes = F * T

        mask = A_gauss > TH  # (F,T,T)
        t = torch.arange(T, device=device)
        mask[:, t, (t + 1).clamp(0, T - 1)] = True
        mask[:, t, (t - 1).clamp(0, T - 1)] = True
        A = A_gauss * mask.float()

        idx = mask.nonzero(as_tuple=False)  # (N,3): [f, i, j]
        vals = A_gauss[idx[:, 0], idx[:, 1], idx[:, 2]]  # (N,)

        src = idx[:, 0] * T + idx[:, 1]
        dst = idx[:, 0] * T + idx[:, 2]
        edge_index = torch.stack([src, dst], dim=0)  # (2, N)
        edge_weight = vals  # (N,)

        # drop self-loops (can appear at boundaries due to clamp)
        keep = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, keep]
        edge_weight = edge_weight[keep]

        edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")

        order = edge_index[0] * num_nodes + edge_index[1]
        perm = torch.argsort(order)
        edge_index = edge_index[:, perm]
        edge_weight = edge_weight[perm]


    if plot_flag:
        A_tmask = (Tmask.unsqueeze(-1) == Tmask.unsqueeze(-2)).float()
        f = 100
        plot_adjacency_matrices(
            [A_tmask[[0, 300, 600, 900], :, :].mean(0).cpu().numpy(),
             A_gauss[[0, 300, 600, 900], :, :].mean(0).cpu().numpy(),
             A[[0, 300, 600, 900], :, :].mean(0).cpu().numpy()],
            ["Real (Tmask)", "Gauss", f"A_>{TH}"],
            suptitle=f"mean over a few freqs adjacency matrices"
        )
    return edge_index, edge_weight, A_gauss, A




def GNN_local_masking(Xt, P, Hlf, Tmask, Emask=None, soft_Emask=None, P_method=CFG.P_method, J=CFG.Q, plot_mask=False, plot_loss=False, gnn_lr=CFG.gnn_lr, lambda_lr=CFG.lambda_lr,
                      betas=CFG.betas, num_epochs=CFG.epochs_gnn, max_norm=CFG.clip_grad_max,
                       param_search=CFG.param_search_flag, local_init_seed=CFG.local_init_seed, low_energy_mask=None,
                      hidden_size=CFG.hidden_gnn, batch_size=CFG.batch_gnn):

    print("Running GNN Local Mapping...")
    Emask_onehot = torch.zeros((1, CFG.lenF0, CFG.N_frames, J + 1), device=CFG.device)
    Emask_onehot[0, np.arange(CFG.lenF0)[:, None], np.arange(CFG.N_frames), torch.from_numpy(Emask).long()] = 1
    if soft_Emask is not None:
        soft_Emask = torch.from_numpy(soft_Emask).float().to(CFG.device)
    Xt = torch.from_numpy(Xt).to(CFG.device).unsqueeze(0)
    onehot_Tmask = Functional.one_hot(torch.from_numpy(Tmask), num_classes=J + 1)[:, :, :J]

    P = torch.from_numpy(P)
    Hlf = torch.from_numpy(Hlf).real.float()


    F, T, C = Hlf.shape
    batch_size = F

    # feature_mat = torch.cat([Hlf,  P.expand(F, T, J)], dim=-1)
    # feature_mat = torch.cat([Hlf, soft_Emask.cpu()], dim=-1)
    feature_mat = Hlf
    dim_input = feature_mat.shape[-1]
    edge_idx, edge_weights, A_gauss, A = calculate_adjancy_mat(feature_mat, Tmask=torch.from_numpy(Tmask), k=10, sigma=1,
                                                                         method='KNN')

    edge_idx = edge_idx.to(CFG.device)
    edge_weights = edge_weights.to(CFG.device)
    P = P.to(CFG.device)
    Hlf = Hlf.to(CFG.device)
    feature_mat = feature_mat.to(CFG.device)



    gnn_model = GCN(input_adjancy_mat=(edge_idx, edge_weights), batch_size=F, num_nodes=T ,in_feats=dim_input, out_feats=J, hidden=hidden_size)
    gnn_model = gnn_model.to(CFG.device)


    # lambda_model = Lambda_MLP(n_losses_feats=2)
    # lambda_model = lambda_model.to(CFG.device)

    loss_function = GNNLoss(F, T, C=dim_input, J=J, Emask=Emask_onehot, epochs=num_epochs,
                            loss_names=['global', 'globalAVG'])

    optimizer = torch.optim.Adam([{'params': gnn_model.parameters(),   'lr': gnn_lr, 'weight_decay': 5e-4}])
                                # {'params': lambda_model.parameters(),  'lr': lambda_lr, 'weight_decay': 0.0},])
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)

    gnn_model.train()
    # lambda_model.train()

    patience = 30
    patience_counter = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):

        gnn_model.train()
        out_features, mask_output = gnn_model(feature_mat)

        loss = loss_function(mask_output, P.unsqueeze(0), R=Hlf.unsqueeze(0), epoch=epoch)

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=max_norm, norm_type=1)
        optimizer.step()

        scheduler.step()

        if best_loss > loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    gnn_model.eval()
    out_features, deep_mask_soft = gnn_model(feature_mat)


    deep_mask_soft = deep_mask_soft.squeeze(0).detach().cpu().numpy()
    deep_mask_hard = deep_mask_soft.argmax(axis=-1)

    deep_mask_hard[low_energy_mask] = J

    if plot_mask:
        plot_masks(Tmask, deep_mask_hard, Emask, P_method=P_method)

    deep_dict_local = {}
    return deep_dict_local, deep_mask_soft, deep_mask_hard, loss_function.name


def GNN_cleaner_masking(Xt, P, Hlf, Tmask, Emask=None, soft_Emask=None, P_method=CFG.P_method, J=CFG.Q, plot_mask=False, plot_loss=False,
                        gnn_lr=CFG.gnn_lr, lambda_lr=CFG.lambda_lr, K_epochs=CFG.K_epochs, sigma=1.0, k_top=CFG.k_neighbours,
                      betas=CFG.betas, num_epochs=CFG.epochs_gnn, max_norm=CFG.clip_grad_max,
                       param_search=CFG.param_search_flag, local_init_seed=CFG.local_init_seed, low_energy_mask=None,
                      hidden_size=CFG.hidden_gnn, batch_size=CFG.batch_gnn):

    print("Running GNN Cleaner Local Mapping...")
    Emask_onehot = torch.zeros((1, CFG.lenF0, CFG.N_frames, J + 1), device=CFG.device)
    Emask_onehot[0, np.arange(CFG.lenF0)[:, None], np.arange(CFG.N_frames), torch.from_numpy(Emask).long()] = 1
    if soft_Emask is not None:
        soft_Emask = torch.from_numpy(soft_Emask).float().to(CFG.device)
    Xt = torch.from_numpy(Xt).to(CFG.device).unsqueeze(0)


    P = torch.from_numpy(P)
    Hlf = torch.from_numpy(Hlf).real.float()


    F, T, C = Hlf.shape
    batch_size = F

    # feature_mat = torch.cat([Hlf,  P.expand(F, T, J)], dim=-1)
    # feature_mat = torch.cat([Hlf, soft_Emask.cpu()], dim=-1)
    feature_mat = Hlf
    dim_input = feature_mat.shape[-1]
    edge_idx, edge_weights, A_gauss, A = calculate_adjancy_mat(feature_mat, Tmask=torch.from_numpy(Tmask), k=k_top, sigma=sigma,
                                                                         method='KNN')

    edge_idx = edge_idx.to(CFG.device)
    edge_weights = edge_weights.to(CFG.device)
    P = P.to(CFG.device)
    Hlf = Hlf.to(CFG.device)
    feature_mat = feature_mat.to(CFG.device)
    A = A.to(CFG.device)


    # gnn_model = GCN(input_adjancy_mat=(edge_idx, edge_weights), batch_size=F, num_nodes=T ,in_feats=dim_input, out_feats=J, hidden=hidden_size)
    gnn_model = SpatialNet(num_layers=CFG.num_layers, dim_squeeze=CFG.dim_squeeze,
                             encoder_kernel_size=CFG.encoder_kernel_size,
                             kernel_size=CFG.kernel_size, conv_groups=CFG.conv_groups, seed=CFG.local_init_seed,
                             low_energy_mask=low_energy_mask).to(CFG.device)
    gnn_model = gnn_model.to(CFG.device)

    lambda_model = Lambda_MLP(n_losses_feats=2)
    lambda_model = lambda_model.to(CFG.device)


    loss_function = GNNLoss_cleaner(F, T, C=dim_input, J=J, Emask=Emask_onehot, epochs=num_epochs, loss_names=['global', 'globalAVG', 'RTF'])



    optimizer = torch.optim.Adam([{'params': gnn_model.parameters(),   'lr': gnn_lr, 'weight_decay': 5e-4},
                                {'params': lambda_model.parameters(),  'lr': lambda_lr, 'weight_decay': 5e-4},])
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)

    gnn_model.train()
    lambda_model.train()

    patience = 30
    patience_counter = 0
    best_loss = float('inf')
    inner_lr = 1e-4
    expanded_P = P.expand(F, T, J)
    for epoch in range(num_epochs):

        out_features, mask_output = gnn_model(feature_mat)  # h, y_hat

        # 1. run label propagation
        with torch.no_grad():
            Y_pseudo = label_propagation(out_features, A, K_epochs, soft_Emask, soft_Emask=soft_Emask, epoch=epoch)  # [F, T, J]
            Y_pseudo = Y_pseudo / (Y_pseudo.sum(dim=-1, keepdim=True) + 1e-10)
        # 2. compute per-node losses
        # loss1 = Functional.mse_loss(mask_output.squeeze(), Y_pseudo)
        # loss2 = Functional.mse_loss(mask_output.squeeze(), expanded_P)


        # loss1 = -(Y_pseudo * torch.log(mask_output.squeeze() + 1e-10)).sum(-1)
        # loss1 = Functional.kl_div(mask_output.log(), Y_pseudo, reduction="batchmean")
        loss1 = loss_function(mask_output, P.unsqueeze(0), R=Hlf.unsqueeze(0), epoch=epoch).squeeze()
        loss2 = Functional.kl_div(mask_output.log(), Y_pseudo, reduction="none").squeeze().sum(-1)

        # loss2 = -(expanded_P * torch.log(mask_output.squeeze() + 1e-10)).sum()
        # loss1 = -(P.unsqueeze(0) * torch.log(mask_output.mean(dim=1) + 1e-10)).sum(dim=(1, 2))



        lambda_output = lambda_model(loss1, loss2)  # [F, T]
        #
        # # 4. corrected labels
        Y_output = lambda_output[:, :, None] * P[None, :, :] + (1 - lambda_output[:, :, None]) * Y_pseudo  # [F, T, J]
        #

        # L_tr = loss1.mean() + loss2.mean()
        L_tr = Functional.kl_div(mask_output.log(), Y_output, reduction="batchmean").squeeze().sum(-1)
        optimizer.zero_grad()
        L_tr.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, l1(mask_out,Y_pseudo): {loss1.mean().item():.4f}, l2(mask_out,P): {loss2.mean().item():.4f}, L_tr(mask_out,Y_out): {L_tr.item():.4f}")

        if best_loss > L_tr:
            best_loss = L_tr
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    gnn_model.eval()
    out_features, deep_mask_soft = gnn_model(feature_mat)


    deep_mask_soft = deep_mask_soft.squeeze(0).detach().cpu().numpy()
    deep_mask_hard = deep_mask_soft.argmax(axis=-1)

    deep_mask_hard[low_energy_mask] = J

    if plot_mask:
        plot_masks(Tmask, deep_mask_hard, Emask, P_method=P_method)

    deep_dict_local = {}
    return deep_dict_local, deep_mask_soft, deep_mask_hard, 'GNN_cleaner'

def label_propagation(out_features, A, K_epochs, noisy_Y=None, soft_Emask=None, epoch=None):
    out_features = out_features.squeeze(0)
    F, T, J = out_features.shape
    Y = noisy_Y

    # dist = torch.cdist(out_features, out_features, p=2)
    W = A #/ (dist + 1e-10)

    for k in range(K_epochs):
        d = W.sum(-1) + 1e-6
        d_inv_sqrt = torch.rsqrt(d)
        S = d_inv_sqrt[:, :, None] * W * d_inv_sqrt[:, None, :]

        Y = torch.bmm(S, Y)
    if CFG.plot_flag and epoch % 100 == 0:
        plot_mask_speakers([soft_Emask, Y], ['Emask', 'Pseudo_Y'], title=f'epoch={epoch}')
    return Y

