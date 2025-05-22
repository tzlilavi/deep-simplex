import os
import random

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import CFG
from global_models import BiLSTM_Att, MiSiCNet2, AutoEncoder
from custom_losses import SupervisedLoss, Unsupervised_Loss, find_best_permutation_supervised
from functions import audio_scores, local_mapping, plot_masks, throwlow

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
                  , mask_input_ratio=CFG.mask_input_ratio, fixed_mask_input_ratio=CFG.fixed_mask_input_ratio,
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