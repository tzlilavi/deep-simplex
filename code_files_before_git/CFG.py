import numpy as np
import os
from scipy import signal
import torch
import math


# Configuration Parameters
seed0 = 32#66#78 15  #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
synth_data_path = 'synthesized_data'
simulated_data_path = 'simulated_data'
supervised_model_path = 'models'
wav_folder = 'wav_examples/wsj0'
write = False
expert = False
P_methods = ['vertices', 'prob', 'both']
P_method = P_methods[1]
num_test_runs = 30

# Parameters
seconds = 20
use_local_deep=False
pad_flag=False
resample_flag=False
plot_flag = False
calibrate = False



data_mode = 'libri'
noise = 0
add_noise = 0
SNR = 20
Q = 3
if data_mode == 'wsj0':
    Q = 2
    seconds=4

SNRs = [20]  # [0,5,10,15,20]

Js = [Q]
run_two_models_flag = False
unknown_J = False


param_search_flag = False
param_tries = 40
loss_plot_flag = True
# if param_search_flag:
#     loss_plot_flag = False


# AutoEncoder parameters
patch = 5
dim = 200


num_samples = 20000
batch_size = 64
### Supervised\unsupervised params
def set_mode(mode):
    global supervised, epochs, lr, epochs, clip_grad_max, SAD_factor, L2_factor, diag_factor, cos_sim_factor, center_factor, \
        weight_decay, dropout, momentum, two_models_factor, two_models_epoch_TH, speakers_epoch_TH, lr_local, betas, epochs_local, RTF_factor, global_factor,\
        num_layers, dim_squeeze, encoder_kernel_size, kernel_size, conv_groups, local_init_seed, n_repeat_last_lstm, global_seed, n_heads, \
        mask_input_ratio, fixed_mask_input_ratio, K_dropout, run_multiple_initializations, random_input, \
        low_energy_mask_time, noise_col, noise_col_weight, local_noise_col, use_local_energy_mask, random_local_input

    if mode == 'supervised':
        supervised = True
        epochs = 40
        lr = 0.00005
        clip_grad_max = 10
        SAD_factor = 5e-2
        L2_factor = 5e3
        cos_sim_factor = 0
        center_factor = 0
        weight_decay = 4e-5
        dropout = 0
        momentum = 0.1

    elif mode == 'unsupervised':
        supervised = False
        epochs = 200
        speakers_epoch_TH = 2
        clip_grad_max = 10
        momentum = 0.1
        betas = (0.9, 0.999)
        # Optuna best:
        global_seed = 5239
        lr = 5 * 1e-05
        SAD_factor = 1
        L2_factor = 1000
        n_heads = 4
        n_repeat_last_lstm = 4
        betas = (0.5, 0.99)
        RTF_factor = 100
        global_factor = 1000

        low_energy_mask_time = True
        noise_col = False
        noise_col_weight = 0
        dropout = None
        K_dropout = None
        run_multiple_initializations = False
        fixed_mask_input_ratio = None
        mask_input_ratio = None
        random_input = False
        if random_input:
            epochs=1000

        local_noise_col = False
        use_local_energy_mask = True
        # Local params
        lr_local = 5 * 1e-4
        dim_squeeze = 8
        epochs_local = 400
        random_local_input = False
        # RTF_factor = 100
        # global_factor = 10000
        # num_layers = 5
        # encoder_kernel_size = 5
        # kernel_size = (5, 3)
        # conv_groups = (8, 8)
        # local_init_seed=seed0

        #### Optuna best:
        local_init_seed = 6665
        lr_local = 5 * 1e-4
        # RTF_factor = 10
        # global_factor = 10000
        num_layers = 5
        encoder_kernel_size = 5
        kernel_size = (3, 3)
        conv_groups = (4, 4)





        two_models_factor = 0.9
        two_models_epoch_TH = 200
        diag_factor = 0
        cos_sim_factor = 0
        center_factor = 5e-3
        weight_decay = 4e-5

        # Set other unsupervised-specific parameters here
    elif mode == 'optuna-supervised':
        epochs = 30
        batch_size = 75
        lr = 0.18406059275672598
        clip_grad_max = 10
        SAD_factor = 0.8578264764388032
        L2_factor = 76.7432665200115
        cos_sim_factor = 0.32058958237028545
        center_factor = 123.73844571425045
        weight_decay = 4e-5
        dropout = 0.3620203999523224
        momentum = 0.7716004014731446
    elif mode == 'optuna-unsupervised':
        epochs = 1000
        batch_size = 75
        lr = 0.00015387931615017762
        clip_grad_max = 17
        SAD_factor = 0.010163885090839295
        L2_factor = 6340.113858609719
        cos_sim_factor = 0.0007212125958031223
        center_factor = 381.80914665607577
        weight_decay = 4e-5
        dropout = 0.4834000083262657
        momentum = 0.5739620775090611




##### two models params
# if not run_two_models_flag:
    # epochs = 20
    # batch_size = 32
    # lr = 0.001
    # center_factor = 0.01
    # reg_factor = 0.1
    # clip_grad_max = 5
    # SAD_factor = 0.01
    # L2_factor = 1000
    # weight_decay = 0

# else:# 2 models factors
#
#     lr = 0.0005
#     # model1 loss factors
#     center_factor = 0.01
#     reg_factor = 10
#     SAD_factor = 0.001
#     L2_factor = 5000
#     # total loss factors
#     model1_factor = 10
#     Qcenter_factor = 1e6
#     SAD2_factor = 10
#     L22_factor = 50
#     model_2_epoch_TH = 10
#     clip_grad_max = 5
#     weight_decay = 0
#
#     # param search
#     lrs = [0.0001, 0.0002, 0.0005, 0.001, 0.002]
#     center_factors = [0.01, 0.1, 1, 10, 100]
#     reg_factors = [0.1, 1, 10, 100]
#     SAD_factors = [0.01, 0.001, 0.0001]
#     L2_factors = [1000, 2000, 3000, 5000]
#     model1_factors = [0.1, 0.5, 1, 5, 10]
#     Qcenter_factors = [1e3, 1e4, 1e5, 1e6]
#     SAD2_factors = [10, 50, 100, 500]
#     L22_factors = [1, 10, 20, 50]
#     model_2_epoch_THs = [10, 20, 30, 40]




#### Preprocessing params
fs_h = 48000
fs = 16000
NFFT = 2 ** 11
olap = 0.75
seconds_pad=0
pad_tfs = 0
pad_size = 0

if pad_flag:
    seconds_pad = 1
    pad_size = int(seconds_pad * fs)
    pad_tfs = math.ceil(pad_size / ((1 - olap) * NFFT))
    seconds += seconds_pad


sample_factor = 1
old_fs = fs
if resample_flag:
    target_fs = 20000
    sample_factor = target_fs / fs
    fs = target_fs

lens_h = seconds * fs_h
lens = int(seconds * fs)
sile = int(0.5 * fs)
sile_h = int(0.5 * fs_h)

len_win = 2 ** 11
320000 / ((1-0.75)*2048) +1
N_frames = math.ceil(lens / ((1 - olap) * NFFT) + 1)

window = signal.get_window("hann", NFFT)
stft_scaling = np.sum(window)

revs = ['RT60_Low']  # ,'RT60_High'
# RT60_Low_rev = 0.15
low_rev = 0.3
high_rev = 0.6

Noise_type = 'air_conditioner'  # 'Babble_noise';%'air_conditioner'; %'Babble_noise'
wav_folder = 'wav_examples'

F0 = np.arange(NFFT // 2 + 1)
lenF0 = len(F0)

f1 = 1000
f2 = 2000
sv = 343
k = np.arange(0, NFFT // 2 + 1)

Iter = 1  # 50

beamformer_type = 'lcmv' # 'lcmv_C', MVDR

att = 0.3
a_beamforming = 0.01
if data_mode=='real':
    mics = np.arange(24, 31)
    M = len(mics)
elif data_mode=='libri':
    M = 4
    room_length = 6
    room_width = 4
    mic_spacing = 0.3
    middle_x = room_length / 2
    middle_y = room_width / 2
    mics = [[middle_x - (M // 2 - i) * mic_spacing, middle_y, 1] for i in range(M)]

else:
    M=4
speakers = 'ABCDEFHIJKLMNOPQST'
chairs = np.arange(1, 7)
figs_dir = 'figures'
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)
highpass = signal.firwin(199, 150, fs=fs, pass_zero="highpass")


F = np.arange(int(np.ceil(f1 * NFFT / fs)), int(np.floor(f2 * NFFT / fs)) + 1)
lenF = len(F)


H_freqbands = lenF * (M - 1) * 2

# Set the random seed for reproducibility
np.random.seed(seed0)


