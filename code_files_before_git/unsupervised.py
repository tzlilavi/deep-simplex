import torch
import os
import joblib
import random
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import scipy
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CyclicLR
from itertools import product
import random
import matplotlib.pyplot as plt
import CFG
CFG.set_mode('unsupervised')
import optuna
from tqdm import tqdm
from MiSiCNet import MiSiCNet2
from LSTMs import BiLSTM_Att
from Transformer_model import AutoEncoder
from functions import compute_rmse, plot_metrics, plot_heat_mat, cosine_sim, throwlow, plot_results, audio_scores, plot_masks, local_mapping
from custom_losses import SAD, NonZeroClipper, Unsupervised_Loss, Unsupervised_Loss_try, Unsupervised_Loss_tryyy, find_best_permutation_unsupervised, LocalLoss, find_best_permutation_supervised, SupervisedLoss
from LocalNets import SpatialNet
import itertools
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import copy
torch.manual_seed(42)


def run_global_model(model, input, W, loss_function, num_epochs, lr=CFG.lr, max_norm=CFG.clip_grad_max,
                     betas=(0.9, 0.999), dropout=CFG.dropout, K_dropout=None,
                     param_search=CFG.param_search_flag, plot_loss=False, mask_input_ratio=None,
                     noise_col=CFG.noise_col, add_noise=CFG.add_noise):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=CFG.weight_decay, betas=betas)


    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)


    input = input.to(CFG.device)
    W_target = W.to(CFG.device)


    best_loss = float('inf')
    patience = 30
    patience_counter = 0

    losses = []
    mask=None
    if CFG.pad_flag:
        W_target = F.pad(W_target, (0, CFG.pad_tfs, 0, CFG.pad_tfs))
        input = F.pad(input, (0, CFG.pad_tfs, 0, CFG.pad_tfs))

    for epoch in range(num_epochs):

        model.train()

        if mask_input_ratio and (epoch % 10)==0:
            mask = mask_input(input, mask_ratio=mask_input_ratio)
            mask = mask.to(CFG.device)
            input = input * mask
            # W_target = W_target * mask

        P_output, W_output, E_output, A_output = model(input, epoch)


        loss, loss_SAD, loss_RE, PPt_output = (
            loss_function(P_output, W_target, mask, W_output, E_output, epoch=epoch))

        # if epoch == CFG.speakers_epoch_TH-1 or epoch ==0:
        #     early_loss = loss.item()

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=1)
        # if epoch % 10 == 0:
        #     print("=== Gradient Norms ===")
        #     for name, param in model.named_parameters():
        #         if param.grad is not None and ("blstm1" in name or "Conv1" in name):
        #             grad_norm = param.grad.norm().item()
        #             print(f"{name}: {grad_norm:.4e}")
        optimizer.step()

        losses.append(loss.item())

        scheduler.step()


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
    d = {}
    d['model_name'] = model.name
    d['lr'] = CFG.lr
    d['epochs'] = num_epochs
    d['loss_name'] = loss_function.name
    d['loss'] = round(loss.item(), 4)
    d['output_mat'] = P_output.detach()
    d['P_torch'] = P_output.detach()
    d['A'] = A_output.detach()
    d['P_noise'] = P_noise
    return d

def MC_dropout_averaging(model, input, K=5):
    P_sum = 0
    model.train()
    print(f'Running MC dropout averaging for K = {K}')
    with torch.no_grad():
        P_samples = [model(input)[0] for _ in range(K)]

    return torch.stack(P_samples).mean(dim=0)
def mask_input(input, mask_ratio=0.15):
    B, L, L = input.shape
    mask = torch.rand(B, L, L, device=input.device) > mask_ratio  # Randomly mask values
    return mask

def global_method(input_mat, W_torch, first_non0, pr2, low_energy_mask_time, J=CFG.Q, lr=CFG.lr, SAD_factor=CFG.SAD_factor,
                  L2_factor=CFG.L2_factor, P_method=CFG.P_method, param_search_flag=CFG.param_search_flag,
                  epochs=CFG.epochs, betas=CFG.betas, n_repeat_last_lstm=CFG.n_repeat_last_lstm, n_heads=CFG.n_heads,
                  seed=CFG.global_seed, Hlf=None, low_energy_mask=None,
                  dropout=CFG.dropout, K_dropout=None, run_multiple_initializations=CFG.run_multiple_initializations
                  , mask_input_ratio=CFG.mask_input_ratio, fixed_mask_input_ratio=CFG.fixed_mask_input_ratio,
                  noise_col=CFG.noise_col, add_noise=CFG.add_noise):

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
                                     dropout=dropout, K_dropout=K_dropout, mask_input_ratio=mask_input_ratio,
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


    P = deep_dict['output_mat']

    loss_function = SupervisedLoss(L2_factor=L2_factor, SAD_factor=SAD_factor)
    _, P, best_permutation, _, _ = find_best_permutation_supervised(loss_function, P.to(CFG.device), torch.from_numpy(pr2).unsqueeze(0).float().to(CFG.device))


    P = P.cpu().numpy().squeeze(0)
    P = throwlow(P)
    P[P.sum(1) > 0, :] = P[P.sum(1) > 0, :] / P[P.sum(1) > 0, :].sum(1, keepdims=True)

    A = deep_dict['A'].detach().cpu().numpy()
    A = A[:, best_permutation]


    return deep_dict, P, A


def run_local_model(model, P, R, loss_function, num_epochs=CFG.epochs_local, lr=CFG.lr_local, max_norm=CFG.clip_grad_max, betas=CFG.betas, param_search=CFG.param_search_flag, plot_loss=False):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=CFG.weight_decay, betas=betas)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)


    P = P.to(CFG.device)
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

        mask_output = model(input)


        loss, loss_1, loss_2 = loss_function(mask_output, R=R.unsqueeze(0), P=P.unsqueeze(0))

        # if epoch == CFG.speakers_epoch_TH-1 or epoch ==0:
        #     early_loss = loss.item()

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=1)
        optimizer.step()

        scheduler.step()


        if not param_search:
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, RTF_loss: {loss_1.item()},"
                      f" global_loss: {loss_2.item()}")


        if best_loss > loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    model.eval()

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


    return d

def deep_local_masking(Xt, P, Hlf, Tmask, Emask=None, P_method=CFG.P_method, J=CFG.Q, plot_mask=False, plot_loss=False, lr=CFG.lr_local, betas=CFG.betas, RTF_factor=CFG.RTF_factor, global_factor=CFG.global_factor, epochs=CFG.epochs_local,
                       num_layers=CFG.num_layers, dim_squeeze=CFG.dim_squeeze, encoder_kernel_size=CFG.encoder_kernel_size, kernel_size=CFG.kernel_size, conv_groups=CFG.conv_groups,
                       param_search=CFG.param_search_flag, local_init_seed=CFG.local_init_seed, low_energy_mask=None):
    print("Running Deep Local Mapping...")
    local_loss = LocalLoss(RTF_factor=RTF_factor, global_factor=global_factor).to(CFG.device)
    local_model = SpatialNet(num_layers=num_layers, dim_squeeze=dim_squeeze, encoder_kernel_size=encoder_kernel_size,
                             kernel_size=kernel_size, conv_groups=conv_groups, seed=local_init_seed, low_energy_mask=low_energy_mask).to(CFG.device)
    deep_dict_local = run_local_model(local_model, torch.from_numpy(P), torch.from_numpy(Hlf), local_loss, lr=lr, betas=betas, plot_loss=plot_loss, param_search=param_search)

    deep_mask_soft = deep_dict_local['deep_mask'].squeeze(0).detach().cpu().numpy()
    deep_mask_hard = deep_mask_soft.argmax(axis=-1)
    # local_ths = find_local_thresholds(deep_mask_soft, P, num_ths=30)
    # deep_mask_hard = J * np.ones((CFG.lenF0, CFG.N_frames))
    # for j in range(J):
    #     deep_mask_hard[deep_mask_soft[:,:,j] > local_ths[j]] = j

    deep_mask_hard[low_energy_mask] = J

    if plot_mask:
        plot_masks(Tmask, deep_mask_hard, Emask, P_method=P_method)

    return deep_dict_local, deep_mask_soft, deep_mask_hard

def find_local_thresholds(deep_mask_soft, P, num_ths=30, F=CFG.lenF0, L=CFG.N_frames):
    th_arr = np.linspace(0.01, 0.99, num_ths)
    P_bigger_than_th = np.sum(P[:, :, None] > th_arr, axis=0) / L
    mask_bigger_than_th = np.sum(deep_mask_soft[:, :, :, None] > th_arr, axis=(0, 1)) / (L * F)
    diffs = abs(P_bigger_than_th - mask_bigger_than_th)
    best_ths = th_arr[np.argmin(diffs, axis=1)]
    return best_ths

def run_model_unknown_J(model, input_matrix, W, loss_function, num_epochs, lr=CFG.lr, U_target=None, max_norm=CFG.clip_grad_max, param_search=CFG.param_search_flag):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=CFG.weight_decay)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)
    if isinstance(U_target, dict):
        U_target = {key: value.to(CFG.device) for key, value in U_target.items()}
    else:
        U_target = U_target.to(CFG.device)
    loss_function = Unsupervised_Loss(U_target=U_target)

    input_matrix = input_matrix.to(CFG.device)
    W_target = W.to(CFG.device)


    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    losses = []
    head_losses = {3: [], 4: [], 5: []}  # To track losses for each head
    loss_weights = np.arange(3, 6)
    total_loss = 0
    init_losses = float('inf')
    tfs_diffs = []
    sil_scores = []
    coeffs_scores = []
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        optimizer.zero_grad()
        P_output, W_output, E_output, U_output = model(input_matrix)
        # Compute loss for each head if before speakers_epoch_TH
        if epoch <= CFG.speakers_epoch_TH:
            total_loss = 0
            loss_per_head = []
            lossU_per_head = []
            SAD_loss_per_head = []
            MSE_loss_per_head = []
            SAD_loss_per_head2 = []
            MSE_loss_per_head2 = []
            abs_diffs = []
            P_heads = []
            PPts = []

            for head_idx, weight in zip(range(3, 6), [1.3, 1.4, 1.5]):
                P_head = P_output[head_idx]
                U_head = U_output[head_idx]
                loss, loss2, _, sad_loss, loss_RE, sad_loss2, loss_RE2, _, PPt_output = find_best_permutation_unsupervised(
                    loss_function,
                    P_head,
                    W_target,
                    W_output=W_output,
                    E_output=E_output,
                    U=U_head
                )
                total_loss += loss
                PPts.append(PPt_output)
                abs_diffs.append(torch.abs(PPt_output-W_target))
                P_heads.append(P_head.squeeze(0))
                lossU_per_head.append(loss2.item())
                loss_per_head.append(loss.item())
                SAD_loss_per_head.append(sad_loss.item())
                MSE_loss_per_head.append(loss_RE.item())
                SAD_loss_per_head2.append(sad_loss2.item())
                MSE_loss_per_head2.append(loss_RE2.item())


                head_losses[head_idx].append(loss.item())
            if epoch == 0:
                init_losses = np.array(loss_per_head.copy())

            total_loss.backward()
            loss_per_head = np.array(loss_per_head)
            loss_diffs_head = init_losses-loss_per_head
            selected_head_idx = np.argmin(np.array(loss_per_head * loss_weights))
            max_diff_idx = np.argmax(loss_diffs_head)
            max_diff_weighted_idx = np.argmax( loss_diffs_head * loss_weights)


            # print(f"Epoch {epoch}, Min weighted loss is for {selected_head_idx + 3} speakers with loss: {loss_per_head[selected_head_idx]}")
            # print(f"Epoch {epoch}, Min weighted loss is for {selected_head_idx + 3} speakers with loss diff: {(init_losses-loss_per_head)[selected_head_idx]}")
            if epoch == CFG.speakers_epoch_TH:
                for P, n_clusters in zip(P_heads, [3,4,5]):
                    P_np = P.squeeze(0).detach().cpu().numpy()

                # print(f"Selected speakers num: {selected_head_idx + 3}")
                losses = head_losses[head_idx]

                return selected_head_idx
        else:
            selected_head = P_heads[selected_head_idx]
            loss, P_output, best_sad_loss, best_loss_RE, best_diagonal_loss, PPt_output = find_best_permutation_unsupervised(loss_function, selected_head,
                                                                                                                             W_target, W_output=W_output,
                                                                                                                             E_output=E_output, U=U_output)
            losses.append(loss.item())
            loss.backward()


        optimizer.step()
        scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=1)



        if not param_search and epoch % 10 == 0 and epoch > CFG.speakers_epoch_TH:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, "
                f"SAD_loss: {best_sad_loss.item()}, loss_RE: {best_loss_RE.item()}"
            )
        # Early stopping
        if best_loss > loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation
    model.eval()

    # Return details
    d = {}
    d['model_name'] = model.name
    d['lr'] = CFG.lr
    d['epochs'] = num_epochs
    d['loss_name'] = loss_function.name
    d['loss'] = round(loss.item(), 4)
    d['output_mat'] = P_output.detach()
    d['P_torch'] = P_output.detach()

    return d




def load_all_dicts(folder_path="array_data"):
    """Load all .pkl files and return a list of dictionaries."""
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pkl")])
    return [joblib.load(os.path.join(folder_path, f)) for f in all_files]


def objective_local(trial):
    """Objective function for Optuna hyperparameter tuning."""

    # Define hyperparameter search space
    seed = trial.suggest_int("seed", 0, 9999)
    lr_local = trial.suggest_categorical("lr_local", [5 * 1e-5, 5 * 1e-4, 5 * 1e-3, 5 * 1e-2, 5 * 1e-1])
    RTF_factor = trial.suggest_categorical("RTF_factor", [10, 100, 1000, 10000])
    global_factor = trial.suggest_categorical("global_factor", [100, 1000, 10000, 100000])
    num_layers = trial.suggest_categorical("num_layers", [3, 5, 7, 9])
    encoder_kernel_size = trial.suggest_categorical("encoder_kernel_size", [3, 5])
    kernel_size_options = [(5, 3), (3, 3), (5, 5), (3, 5)]
    conv_groups_options = [(8, 8), (6, 6), (4, 4), (2, 2)]
    kernel_size = trial.suggest_categorical("kernel_size", [str(k) for k in kernel_size_options])
    conv_groups = trial.suggest_categorical("conv_groups", [str(c) for c in conv_groups_options])
    kernel_size = eval(kernel_size)
    conv_groups = eval(conv_groups)

    # Fixed parameters
    epochs_local = 100
    dim_squeeze = 8  # Always fixed

    # Load all datasets and randomly select 10
    all_dicts = load_all_dicts('array_data_global_tuning_full')
    selected_dicts = random.sample(all_dicts, 5)

    sisdr_scores = []

    for data_dict in tqdm(selected_dicts):

        # Extract necessary data
        Xt = data_dict['Xt']
        P = data_dict['P']
        Hlf = data_dict['Hlf']
        pe = data_dict['pe']
        pr2 = data_dict['SDR_prob']  # Assuming this is the correct key
        P2 = data_dict['P2']
        Tmask = data_dict['Tmask']
        Emask = data_dict['Emask']
        Emask2 = data_dict['Emask2']
        Emask_pe = data_dict['Emask_pe']
        Hl = data_dict['Hlf']
        Hq = data_dict['Hlf']  # Placeholder, adjust if needed
        Xq = data_dict['Xq']
        xqf = data_dict['xqf']
        fh2 = data_dict['fh2']
        fh22 = data_dict['fh22']
        fh2_pe = data_dict['fh2_pe']

        # Call deep_local_masking with sampled hyperparameters
        _, _, deep_mask_hard = deep_local_masking(
            Xt, P, Hlf, Emask, Tmask,
            lr=lr_local,
            RTF_factor=RTF_factor,
            global_factor=global_factor,
            epochs=epochs_local,
            num_layers=num_layers,
            dim_squeeze=dim_squeeze,
            encoder_kernel_size=encoder_kernel_size,
            kernel_size=kernel_size,
            conv_groups=conv_groups,
            seed=seed,
            param_search=True
        )

        # Compute SDR score
        scores = audio_scores(pe, pr2, P2, None, Tmask, deep_mask_hard, None, Emask_pe, Hl, Xt, Hq, Xq, xqf, fh2, None,
                              fh2_pe,
                              P_method='prob', calc_SPA_scores=False, print_scores=False)
        sisdr_prob_NN = scores['sisdr_prob_NN']
        print(sisdr_prob_NN)
        sisdr_scores.append(sisdr_prob_NN)

    # Return mean SDR as the metric to maximize
    return np.mean(sisdr_scores)

def objective_global(trial):
    """Objective function for Optuna hyperparameter tuning."""

    # Define hyperparameter search space
    seed = trial.suggest_int("seed", 0, 9999)
    lr = trial.suggest_categorical("lr", [5 * 1e-5, 5 * 1e-4, 5 * 1e-3, 5 * 1e-2, 5 * 1e-1])
    SAD_factor = trial.suggest_categorical("SAD_factor", [10, 100, 1000, 10000])
    L2_factor = trial.suggest_categorical("L2_factor", [100, 1000, 10000, 100000])
    n_heads = trial.suggest_categorical("n_heads", [4,8,12])
    n_repeat_last_lstm = trial.suggest_categorical("n_repeat_last_lstm", [1,2,3])
    betas_options = [(0.9, 0.999), (0.85, 0.98), (0.5, 0.99), (0.9, 0.95)]
    betas = trial.suggest_categorical("betas", [str(k) for k in betas_options])
    betas = eval(betas)
    # Fixed parameters
    epochs_local = 100


    # Load all datasets and randomly select 10
    all_dicts = load_all_dicts('array_data_global_tuning_full')
    selected_dicts = random.sample(all_dicts, 15)

    sisdr_scores = []

    for data_dict in tqdm(selected_dicts):

        # Extract necessary data
        Xt = data_dict['Xt']
        P = data_dict['P2']
        Hlf = data_dict['Hlf']
        pe = data_dict['pe']
        pr2 = data_dict['pr2']  # Assuming this is the correct key
        # P2 = data_dict['P2']
        Tmask = data_dict['Tmask']
        Emask = data_dict['Emask']
        Emask2 = data_dict['Emask2']
        Emask_pe = data_dict['Emask_pe']
        Hl = data_dict['Hlf']
        Hq = data_dict['Hlf']  # Placeholder, adjust if needed
        Xq = data_dict['Xq']
        xqf = data_dict['xqf']
        fh2 = data_dict['fh2']
        fh22 = data_dict['fh22']
        fh2_pe = data_dict['fh2_pe']
        W = data_dict['W']
        first_non0 = data_dict['first_non0']
        low_energy_mask = data_dict['low_energy_mask']
        low_energy_mask_time = data_dict['low_energy_mask_time']

        # Call deep_local_masking with sampled hyperparameters
        _, P, _ = global_method(W, first_non0, pr2, low_energy_mask_time, J=CFG.Q, lr=lr, SAD_factor=SAD_factor,
                  L2_factor=L2_factor, P_method='prob', param_search_flag=True,
                  epochs=100, betas=betas, n_repeat_last_lstm=n_repeat_last_lstm, n_heads=n_heads, seed=seed)

        Emask, fh2, _, _, _, _ = local_mapping(pe, P, None, Hlf, Xt, low_energy_mask, CFG.Q, None, None,
                                                                   'prob', 'prob', plot_Emask=False)

        scores = audio_scores(pr2, P, Tmask, Emask,
                 Hl, Xt, Hq, Xq, xqf, fh2, J=CFG.Q, compute_ideal=False,
                 P_method='prob', local_method='NN', print_scores=False)

        sisdr_prob_NN = scores['si-sdr_prob_NN']
        print(sisdr_prob_NN)
        sisdr_scores.append(sisdr_prob_NN)
        if np.mean(sisdr_scores) < 0:
            return np.mean(sisdr_scores)

        # Return mean SDR as the metric to maximize
    return np.mean(sisdr_scores)
def optuna_param_search(n_trials=50, excel_file="optuna_global_results.xlsx"):
    """Run Optuna to find the best hyperparameters."""

    # Create Optuna study
    study = optuna.create_study(direction='maximize')  # SDR should be maximized
    study.optimize(objective_global, n_trials=n_trials)

    # Save results to Excel
    df_results = pd.DataFrame(study.trials_dataframe())
    df_results.to_excel(excel_file, index=False)

    # Print best result
    print("\nBest hyperparameters:", study.best_params)
    print("Best SI-SDR value:", study.best_value)

    return study.best_params

if __name__ == "__main__":
    a=5
    # d_list = load_all_dicts()
    # d = d_list[0]
    # model = SpatialNet()
    # loss_function = LocalLoss()
    # Xt = d['Xt']
    # P = d['P']
    # P2 = d['P2']
    # pe = d['pe']
    # Tmask = d['Tmask']
    # Emask_pe = d['Emask_pe']
    # fh2 = d['fh2']
    # fh2_pe = d['fh2_pe']
    # fh22 = d['fh22']
    # Xq=d['Xq']
    # Hlf = d['Hlf']
    # Emask = d['Emask']
    # Emask2 = d['Emask2']
    # deep_dict_local, deep_mask_soft, deep_mask_hard = deep_local_masking(Xt, P2, Hlf, J=CFG.Q, P_method='prob')
    # plot_masks(Tmask, Emask, 1 - deep_mask_soft[:, :, 0], 'prob')

    best_params = optuna_param_search(n_trials=30)