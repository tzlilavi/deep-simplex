import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
import random
import matplotlib.pyplot as plt
import torch.nn.init as init
import torch.nn.functional as F
import CFG
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from unsupervised import SAD, NonZeroClipper, run_1_model
CFG.set_mode('supervised')
from Transformer_model import AutoEncoder
import pickle
from create_synth_data import create_W_P_efficient, create_W_P
from tqdm import tqdm
import time
from MiSiCNet import MiSiCNet2
from functions import plot_metrics, compute_rmse, plot_P_speakers, calculate_SPA_simplex, cosine_sim, \
    feature_extraction, calculate_W_U_realSimplex, initialize_arrays, local_mapping, dist_scores, audio_scores
from custom_losses import SupervisedLoss, find_best_permutation_supervised, center_reg, Unsupervised_Loss
from data_simulator import get_speaker_signals, combine_speaker_signals_no_noise, generate_RIRs
import gzip
import joblib
import itertools
import optuna
from functools import partial
from scipy.stats import spearmanr
from LSTMs import BiLSTM, BiLSTM_Att
random.seed(CFG.seed0)
# torch.backends.cudnn.enabled = False


class Synthetic_Dataset(Dataset):
    def __init__(self, data_dict):
        self.num_samples = len(data_dict['Ws'])
        self.data_dict = data_dict

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(np.float32(self.data_dict['Ws'][idx]), requires_grad=True).to(CFG.device), \
               torch.tensor(np.float32(self.data_dict['Ps'][idx]), requires_grad=True).to(CFG.device)#,\
               # torch.tensor(np.float32(self.data_dict['Ws_real'][idx]), requires_grad=True).to(CFG.device)


def create_loaders(data_dict, batch_size, num_samples=CFG.num_samples, train_size_split=0.8):

    if not isinstance(data_dict, list):
        dataset = Synthetic_Dataset(data_dict)
        indices = list(range(num_samples))
        train_indices, val_indices = train_test_split(indices, train_size=train_size_split, random_state=CFG.seed0)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
    else:
        train_dataset = Synthetic_Dataset(data_dict[0]) ## Training data_dict
        val_dataset = Synthetic_Dataset(data_dict[1]) ## Val data_dict

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def training_step(model, loss_class, batch ,center_factor=CFG.center_factor,
                  cos_sim_factor = CFG.cos_sim_factor):
    # W_input, P_target, W_target = batch
    W_input, P_target = batch

    W_input = W_input.to(CFG.device)
    P_target = P_target.to(CFG.device)

    P_output, W_output, _, _ = model(W_input)

    P_output = P_output.to(CFG.device)
    W_output = W_output.to(CFG.device)

    # P_output = P_output[:, :, :-1]   ### Remove noise dim
    # P_target = P_target[:, :, :-1]

    
    W_output = torch.bmm(P_output, torch.transpose(P_output,1,2))
    W_output[:, range(CFG.N_frames), range(CFG.N_frames)] = 1

    W_target = torch.bmm(P_target, torch.transpose(P_target, 1, 2))
    W_target[:, range(CFG.N_frames), range(CFG.N_frames)] = 1

    loss_P, P_output, _ = find_best_permutation_supervised(loss_class, P_output, P_target)
    # loss_W = loss_class(W_output, W_target)

    reg = center_factor * center_reg(P_output, W_target)

    cols_cos_sim = cos_sim_factor * cosine_sim(P_output)
    loss = loss_P + reg + cols_cos_sim

    # P_output, P_target = P_output[:, :, :-1], P_target[:, :, :-1]  ### Remove noise dim
    RMSE_train = compute_rmse(P_output, P_target)
    SAD_construct = SAD()
    SAD_train = SAD_construct(P_output, P_target)
    return loss, RMSE_train.to(CFG.device), SAD_train.to(CFG.device), cols_cos_sim, reg.item()

def train_model(train_loader, val_loader, model, loss_class, lr=CFG.lr, epochs=CFG.epochs, center_factor = CFG.center_factor, cos_sim_factor = CFG.cos_sim_factor,
                max_norm=CFG.clip_grad_max, param_search=CFG.param_search_flag, num_samples=CFG.num_samples, patience=5, gpu2=False):
    model.to(CFG.device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=CFG.weight_decay)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)
    apply_clamp_inst1 = NonZeroClipper()

    train_losses = []
    val_losses = []
    train_rmses = []
    val_rmses = []
    train_sads = []
    val_sads = []

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):

        model.train()  # Set model to training mode
        train_loss = 0
        train_rmse = 0
        train_sad = 0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as progress_bar:
            for batch_idx, batch in enumerate(train_loader):
                batch = [tensor.to(CFG.device) for tensor in batch]
                loss, RMSE, SAD, train_cols_cos_sim , center_reg= training_step(model, loss_class, batch, center_factor=center_factor,
                                                              cos_sim_factor = cos_sim_factor)

                optimizer.zero_grad()
                loss.backward()
                # model.decoder.apply(apply_clamp_inst1)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=1)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_rmse += RMSE.item()
                train_sad += SAD.item()

                if batch_idx % 10 == 0:
                    progress_bar.update(10)
                    if not param_search:
                        print(f"\nTrain loss = {loss.item()}, Train RMSE = {RMSE.item()}, Train SAD = {SAD.item()},"
                              f" center_reg = {center_reg}, cos_sim = {train_cols_cos_sim.item()}")

            avg_train_loss = train_loss / len(train_loader)
            avg_train_rmse = train_rmse / len(train_loader)
            avg_train_sad = train_sad / len(train_loader)

            train_losses.append(avg_train_loss)
            train_rmses.append(avg_train_rmse)
            train_sads.append(avg_train_sad)

            # Validation
            model.eval()  # Set model to evaluation mode
            val_loss = 0
            val_rmse = 0
            val_sad = 0
            for batch in val_loader:
                vloss, RMSE_val, SAD_val, val_cols_cos_sim, center_reg = training_step(model, loss_class, batch, center_factor=center_factor, cos_sim_factor = cos_sim_factor)
                val_loss += vloss.item()
                val_rmse += RMSE_val.item()
                val_sad += SAD_val.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_rmse = val_rmse / len(val_loader)
            avg_val_sad = val_sad / len(val_loader)

            val_losses.append(avg_val_loss)
            val_rmses.append(avg_val_rmse)
            val_sads.append(avg_val_sad)

            print(
                f"\nEpoch {epoch + 1}, Train Loss: {avg_train_loss}, Train RMSE: {avg_train_rmse}, Train SAD: {avg_train_sad},\n "
                f"Val Loss: {avg_val_loss}, Val RMSE: {avg_val_rmse}, Val SAD: {avg_val_sad}")
            state_name = f'{CFG.supervised_model_path}/dirichlet_J{CFG.Q}_{model.name}_epoch_{epoch + 1}_valRMSE{avg_val_rmse:.3f}_augment1_samples{num_samples}.pth'

            if best_loss > val_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    if not param_search:
                        if gpu2:
                            torch.save(model.module.state_dict(), state_name)
                        else:
                            torch.save(model.state_dict(), state_name)
                    break
            if not param_search:
                if (epoch + 1) % 5 == 0:
                    if gpu2:
                        torch.save(model.module.state_dict(), state_name)
                    else:
                        torch.save(model.state_dict(), state_name)

    if not param_search:
        plot_metrics(train_losses, train_rmses, train_sads, val_losses, val_rmses, val_sads)
        return model.state_dict()
    else:
        return avg_val_rmse, model.state_dict()

def objective(trial, data_dict):
    print(f"Trial {trial.number}")
    lr = trial.suggest_float('lr', 0.0001, 0.5, log=True)
    SAD_factor = trial.suggest_float('SAD_factor', 1e-4, 1, log=True)
    L2_factor = trial.suggest_float('L2_factor', 1, 8000, log=False)
    center_factor = trial.suggest_float('center_factor', 0, 500, log=False)
    cos_sim_factor = trial.suggest_float('cos_sim_factor', 1e-5, 1, log=True)
    dropout = trial.suggest_float('dropout', 0, 0.5, log=False)
    momentum = trial.suggest_float('momentum', 0.1, 0.9, log=False)
    batch_size = trial.suggest_int('batch_size', 32,100, log=False)
    max_norm = trial.suggest_int('max_norm', 5, 50, log=False)


    train_loader, val_loader = create_loaders(data_dict=data_dict, num_samples=CFG.num_samples,
                                              batch_size=batch_size)
    loss_class = SupervisedLoss(L2_factor=L2_factor, SAD_factor=SAD_factor)
    model = MiSiCNet2(CFG.N_frames, out_dim=CFG.Q, dropout=dropout, momentum=momentum).to(CFG.device)
    score, state_dict = train_model(train_loader, val_loader, model, loss_class, lr=lr, center_factor=center_factor,
                cos_sim_factor=cos_sim_factor,  max_norm=max_norm, param_search=True)


    model = MiSiCNet2(CFG.N_frames, out_dim=CFG.Q).to(CFG.device)
    model.load_state_dict(state_dict)
    W, P, _ = create_W_P_efficient(2, CFG.N_frames, CFG.H_freqbands, noise_dim=1)
    P_output, _, _ = model(torch.from_numpy(W).float().unsqueeze(0).to(CFG.device))
    P_output_np = P_output.squeeze(0).detach().cpu().numpy()

    spearman_corr_1st_speaker, _ = spearmanr(P[:, 0], P_output_np[:, 0])
    spearman_corr_2nd_speaker, _ = spearmanr(P[:, 1], P_output_np[:, 1])
    score = torch.mean(torch.abs(spearman_corr_1st_speaker), torch.abs(spearman_corr_2nd_speaker))
    return score

def optuna_param_search(data_dict, n_trials):
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(objective, data_dict=data_dict), n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)

def test_model(model, loss_func, test=True, method='dirichlet'): ## Test model on new synthetic data

    if test=='real':
        W = np.load(f'example_data/W_J{CFG.Q}_L626_SNR20.npy')
        pr2 = np.load(f'example_data/REAL-P_J{CFG.Q}_L626_SNR20.npy')
    elif test=='simulated':
        signals, previous_combinations, speakers = get_speaker_signals('dev-wav-2/train', None)
        J = CFG.Q + CFG.noise
        RIRs, _ = generate_RIRs(room_length=6, room_width=6, mic_spacing=0.3, num_mics=6, min_angle_difference=30,
                             radius=2,
                             num_of_RIRs=J)
        Xt, Tmask, f, t, xqf = combine_speaker_signals_no_noise(signals, RIRs, num_mics=6, J=J)

        Hl, Fall, lenF, F = feature_extraction(Xt)
        Hln, W, E0, pr2 = calculate_W_U_realSimplex(Hl, Fall, Tmask, lenF, F)


    else:
        W, pr2, _ = create_W_P_efficient(CFG.Q, CFG.N_frames, CFG.H_freqbands, method=method, noise_dim=1)
    de, E0 = np.linalg.eig(W)
    pe, id0, ext0 = calculate_SPA_simplex(E0, pr2, CFG.Q)  ### Calculate SPA P on that data
    pe[:, :CFG.Q] = pe[:, id0]
    U_torch = torch.from_numpy(E0[:, :J]).float()

    unsupervised_loss = Unsupervised_Loss()

    W_torch = torch.from_numpy(W).float().unsqueeze(0)
    CFG.set_mode('unsupervised')
    deep_dict = run_1_model(model, W_torch.clone(), W_torch, unsupervised_loss, CFG.epochs, param_search=CFG.param_search_flag)
    P = deep_dict['output_mat']
    loss_function = SupervisedLoss(L2_factor=CFG.L2_factor, SAD_factor=CFG.SAD_factor)
    _, P, _ = find_best_permutation_supervised(loss_function, P.to(CFG.device),
                                            torch.from_numpy(pr2).unsqueeze(0).float().to(CFG.device))

    P = P.cpu().numpy().squeeze(0)

    speaker_data = [
            (pr2[:, :CFG.Q], 'real_P_speakers', 'real P speakers over L', pr2[:, -1]),
            (P[:, :CFG.Q], 'model_P_speakers', 'model_P_speakers', P[:, -1]),
            (pe[:, :CFG.Q], 'SPA_model_P_speakers', 'SPA P speakers over L', pe[:, -1])
        ]

    # Create a new figure for subplots
    plt.figure(figsize=(18, 6))

    # Loop through each speaker data and plot as subplot
    for i, (speakers, plot_name, title, noise) in enumerate(speaker_data, 1):
        if CFG.noise == 0:
            noise = None
        plt.subplot(1, 3, i)
        plot_P_speakers(speakers, plot_name, CFG.figs_dir, title=title, noise=noise, save_flag=False,
                    show_flag=False, need_fig=False)
    plt.tight_layout()
    plt.show()

    L, J = P.shape

    fig, axs = plt.subplots(J, 1, figsize=(10, 4 * J))  # Create subplots for each speaker + noise

    for i in range(J):
        if CFG.noise==1:
            if i < J - 1:
                label_real = f'Real P - Speaker {i + 1} (Col {i})'
                label_model = f'Model P - Speaker {i + 1} (Col {i})'
                title = f'Real vs Model P matrix - Speaker {i + 1}'
            else:
                label_real = f'Real P - Noise (Col {i})'
                label_model = f'Model P - Noise (Col {i})'
                title = 'Real vs Model P matrix - Noise'
        else:
            label_real = f'Real P - Speaker {i + 1} (Col {i})'
            label_model = f'Model P - Speaker {i + 1} (Col {i})'
            title = f'Real vs Model P matrix - Speaker {i + 1}'
        # Plot real and model P matrix for each speaker/noise
        axs[i].plot(range(50), pr2[:50, i], label=label_real, color='b')
        axs[i].plot(range(50), P[:50, i], label=label_model, color='r', linestyle='--')
        axs[i].set_title(title)
        axs[i].set_xlabel('Timeframes')
        axs[i].set_ylabel('Probability')
        axs[i].legend()
        axs[i].grid(True)

        # Calculate Pearson correlation coefficient
        pearson_corr = np.corrcoef(pr2[:, i], P[:, i])[0, 1]

        # Calculate Spearman correlation coefficient
        spearman_corr, _ = spearmanr(pr2[:, i], P[:, i])

        if noise==1:
            if i < J - 1:
                print(f"Speaker {i + 1}: Pearson Corr = {pearson_corr:.4f}, Spearman Corr = {spearman_corr:.4f}")
            else:
                print(f"Noise: Pearson Corr = {pearson_corr:.4f}, Spearman Corr = {spearman_corr:.4f}")
        else:
            print(f"Speaker {i + 1}: Pearson Corr = {pearson_corr:.4f}, Spearman Corr = {spearman_corr:.4f}")

    # Adjust layout
    # plt.suptitle('')
    plt.tight_layout()
    plt.show()



    for rr, rev in enumerate(CFG.revs):
        for ss, SNR in enumerate(CFG.SNRs):
            MD, FA, MD_pe, FA_pe, SDR, SIR, SDR_pe, SIR_pe, SDRi, SIRi, SDRiva, SIRiva, SDRp, SIRp, SDRp_pe, SIRp_pe, SNRin, den, sumin, spk, chr = initialize_arrays()
            Emask, fh2, Emask_pe, fh2_pe = local_mapping(pe, P, Tmask, Hl, Xt, f, t, deep_dict['model_name'], MD, FA, 0,
                                                         ss, rr)
            deep_L2, deep_mse, SPA_L2, SPA_mse = dist_scores(pe, P, pr2)
            model_tested = deep_dict['model_name']
            MD, FA, SDRp, SIRp, \
            SDRi, SIRi, SDR, SIR = audio_scores(pe, pr2, P, Tmask, Emask, Emask_pe, Hl, Xt, f, t, model_tested, MD, FA, MD_pe,
                                                FA_pe,
                                                0, ss, rr, SDRp, SIRp, SDRp_pe, SIRp_pe, SIRi, SDRi, SIR, SDR, SIR_pe, SDR_pe, xqf,
                                                fh2, fh2_pe)

nets = [MiSiCNet2(CFG.N_frames, out_dim=CFG.Q, dropout=CFG.dropout, momentum=CFG.momentum).to(CFG.device),
        AutoEncoder(J=CFG.Q, L=CFG.N_frames, size=CFG.N_frames, patch=CFG.patch, dim=CFG.dim).to(CFG.device),
        BiLSTM_Att().to(CFG.device)]
model = nets[0] ## Choose a model
loss_class = SupervisedLoss(L2_factor=CFG.L2_factor, SAD_factor=CFG.SAD_factor)
train_flag = True

if train_flag:
    #### Call one of the joblib or pkl data files:
    # filename = f'{CFG.synth_data_path}/DataDict_J{CFG.Q}_L{CFG.N_frames}_noise{1}_eff_updated{CFG.num_samples}.joblib'

    filename = f'{CFG.synth_data_path}/DataDict_J{CFG.Q}_L{CFG.N_frames}_noise{CFG.noise}_dirichlet{CFG.num_samples}_augmented1.joblib'
    # filename = f'{CFG.synth_data_path}/DataDict_J{CFG.Q}_L{CFG.N_frames}_noise{CFG.noise}_dirichlet_identical_sources{CFG.num_samples}_augmented1.joblib'
    data_dict = joblib.load(filename) ### takes time

    num_samples = 20000
    batch_size = CFG.batch_size
    train_size_split = 0.8
    train_loader, val_loader = create_loaders(data_dict=data_dict, batch_size = batch_size, num_samples=num_samples, train_size_split=train_size_split)

    ## Start training
    print(f'Starting training for {CFG.epochs} epochs, on {int(num_samples*train_size_split)} samples...')
    start = time.time()
    model = MiSiCNet2(CFG.N_frames, out_dim=CFG.Q, dropout=CFG.dropout, momentum=CFG.momentum).to(CFG.device)
    # if CFG.device.type == 'cuda' and torch.cuda.device_count() == 2:
    #     new_model = nn.DataParallel(model, [0, 1])
    #     new_model.name = model.name
    #     model = new_model
    gpu2=False
    patience=3
    state_dict = train_model(train_loader, val_loader, model, loss_class, lr=CFG.lr, epochs=CFG.epochs, center_factor=CFG.center_factor,
                cos_sim_factor = CFG.cos_sim_factor,max_norm=CFG.clip_grad_max, param_search=False, patience=patience, gpu2=gpu2)
    print(f'Done training in {(time.time() - start)/60} minutes')

##### TESTING


# model_file = f'models/supervised_J2_MiSiCNet2_epoch_10_valRMSE0.211.pth'
# model_file = f'models/supervised_J2_MiSiCNet2_epoch_11_valRMSE0.173.pth'  ### 2 best
# model_file = f'models/dirichlet_J2_MiSiCNet2_epoch_20_valRMSE0.094.pth'
model_file = f'models/dirichlet_J3_MiSiCNet2_epoch_40_valRMSE0.125_augment1.pth'
# model_file = f'models/dirichlet_J2_MiSiCNet2_epoch_20_valRMSE0.108.pth'
# model_file = f'models/dirichlet_J2_MiSiCNet2_epoch_20_valRMSE0.094.pth'

if not train_flag:
    state_dict = torch.load(model_file, map_location=torch.device(CFG.device), pickle_module=pickle)

model.load_state_dict(state_dict)


test_model(model, loss_class, test='simulated', method='dirichlet')

a=5

















