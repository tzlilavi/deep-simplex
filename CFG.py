import math
import os
import numpy as np
import torch
from scipy import signal

# === Device and Seed ===
seed0 = 32
np.random.seed(seed0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Paths ===
synth_data_path = 'synthesized_data'
simulated_data_path = 'simulated_data'
supervised_model_path = 'models'
wav_folder = 'wav_examples/wsj0'
figs_dir = 'figures'
os.makedirs(figs_dir, exist_ok=True)
write = False

# === Core Settings ===
data_mode = 'libri'  # or 'wsj0'
add_noise = 0
SNR = 20
use_local_deep = False
plot_flag = False
param_search_flag = False
param_tries = 40
loss_plot_flag = True

# === Evaluation & Testing ===
P_methods = ['vertices', 'prob', 'both']
P_method = P_methods[1]  # Default to 'prob'
num_test_runs = 30

# === Signal Duration & Sampling ===
seconds = 20
SNRs = [20]
Q = 3 if data_mode != 'wsj0' else 2
if data_mode == 'wsj0':
    seconds = 4

# === Preprocessing ===
fs = 16000
fs_h = 48000
NFFT = 2 ** 11
olap = 0.75
pad_flag = False
pad_size = 0
seconds_pad = 0
pad_tfs = 0
if pad_flag:
    seconds_pad = 1
    pad_size = int(seconds_pad * fs)
    pad_tfs = math.ceil(pad_size / ((1 - olap) * NFFT))
    seconds += seconds_pad

sample_factor = 1
old_fs = fs
resample_flag = False
if resample_flag:
    target_fs = 20000
    sample_factor = target_fs / fs
    fs = target_fs

lens_h = seconds * fs_h
lens = int(seconds * fs)
sile = int(0.5 * fs)
sile_h = int(0.5 * fs_h)

len_win = 2 ** 11
N_frames = math.ceil(lens / ((1 - olap) * NFFT) + 1)

window = signal.get_window("hann", NFFT)
stft_scaling = np.sum(window)



# === Global Hyperparameters ===
epochs = 200
speakers_epoch_TH = 2
clip_grad_max = 10
momentum = 0.1
global_seed = 5239
lr = 5 * 1e-05
SAD_factor = 1
L2_factor = 1000
n_heads = 4
n_repeat_last_lstm = 4
betas = (0.5, 0.99)
patch = 5
dim = 200
weight_decay = 4e-5
low_energy_mask_time = True
noise_col = False
noise_col_weight = 0
dropout = None
K_dropout = None
run_multiple_initializations = False
fixed_mask_input_ratio = None

random_input = False
if random_input:
    epochs=1000

# === Local Hyperparameters ===
RTF_factor = 100
global_factor = 1000

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




# === Frequency and Beamforming ===
f1 = 1000
f2 = 2000
sv = 343
F = np.arange(int(np.ceil(f1 * NFFT / fs)), int(np.floor(f2 * NFFT / fs)) + 1)
lenF = len(F)
F0 = np.arange(NFFT // 2 + 1)
lenF0 = len(F0)
k = np.arange(0, NFFT // 2 + 1)
H_freqbands = lenF * (4 - 1) * 2  # default M=4
beamformer_type = 'lcmv'
a_beamforming = 0.01
att = 0.3
Iter = 1


# === Microphone Settings ===
if data_mode == 'libri':
    M = 4
    room_length = 6
    room_width = 4
    mic_spacing = 0.3
    middle_x = room_length / 2
    middle_y = room_width / 2
    mics = [[middle_x - (M // 2 - i) * mic_spacing, middle_y, 1] for i in range(M)]
else:
    M = 4


# === Misc (not using this data anymore) ===
speakers = 'ABCDEFHIJKLMNOPQST'
chairs = np.arange(1, 7)
revs = ['RT60_Low']  # ,'RT60_High'
# RT60_Low_rev = 0.15
low_rev = 0.3
high_rev = 0.6
Noise_type = 'air_conditioner'  # 'Babble_noise';%'air_conditioner'; %'Babble_noise'
wav_folder = 'wav_examples'
highpass = signal.firwin(199, 150, fs=fs, pass_zero="highpass")




