import os
import pickle
import random
import time

import joblib
import numpy as np
import scipy.signal as ss
import soundfile as sf
from scipy.signal import resample, stft
from tqdm import tqdm

import CFG
import rir_generator as rir
from functions import feature_extraction, calculate_W_U_realSimplex, calculate_SPA_simplex, calc_auxip

np.random.seed(CFG.seed0)
random.seed(CFG.seed0)


def concatenate_flac_with_gaps(input_directory, sample_rate=CFG.fs, max_duration=30, delay_max=5, delay_min=3):
    """
    Concatenate .flac audio files from a directory into a single signal with optional silent gaps.

    Args:
        input_directory (str): Path to speaker folder containing .flac files.
        sample_rate (int): Target sample rate (default: CFG.fs).
        max_duration (int): Max total duration (in seconds) of output signal.
        delay_max (int): Maximum silent gap between utterances (in seconds).
        delay_min (int): Minimum silent gap between utterances (in seconds).

    Returns:
        np.ndarray: Normalized concatenated waveform with inserted silence.
    """
    concatenated_signal = np.array([])
    max_samples = max_duration * sample_rate

    # Collect all .flac files in the first subdirectory
    flac_files = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if file.endswith('.flac')]
    flac_files.sort()  # Ensure a consistent order

    for i, flac_file in enumerate(flac_files):
        # Read the .flac file
        signal, sr = sf.read(flac_file)

        # Resample if needed
        if sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {flac_file} has {sr} Hz, expected {sample_rate} Hz.")


        delay_length = random.randint(delay_min * sample_rate, delay_max * sample_rate)
        delay_signal = np.zeros(delay_length)

        if len(concatenated_signal) == 0:
            # Add delay at the beginning
            concatenated_signal = np.concatenate((delay_signal, signal))
        else:
            # Add delay between segments
            concatenated_signal = np.concatenate((concatenated_signal, delay_signal, signal))

        if len(concatenated_signal) > max_samples:
            concatenated_signal = concatenated_signal[:max_samples]
            break


    concatenated_signal = concatenated_signal / np.max(np.abs(concatenated_signal))
    return concatenated_signal


def apply_silence_2_speakers(xqf, overlap_demand=0.5, num_silences=3, sample_rate=16000):
    """
        Introduce controlled silence into a 2-speaker mixture to match a target overlap ratio.

        Args:
            xqf (np.ndarray): [T, C, 2] time-domain mixture for 2 speakers.
            overlap_demand (float): Desired overlap ratio (0 to 1).
            num_silences (int): Number of silent segments to insert.
            sample_rate (int): Not used, included for compatibility.

        Returns:
            np.ndarray: Modified signal with inserted silent segments.
        """
    y = xqf.copy()
    lens = xqf.shape[0]

    silence_ratio = 1 - overlap_demand
    total_silence_samples_per_speaker = int(silence_ratio * lens)

    # Generate `num_silences` random silence segments that sum to total silence
    silence_lens = np.random.randint(
        1, total_silence_samples_per_speaker // num_silences + 1, size=num_silences
    )

    # Adjust to ensure the total silence per speaker is exactly correct
    silence_lens = silence_lens / silence_lens.sum() * total_silence_samples_per_speaker
    silence_lens = silence_lens.astype(int)

    # Ensure silences are evenly spread by picking random start positions
    start_positions = sorted(random.sample(range(lens - max(silence_lens)), num_silences))

    for i in range(num_silences):
        silent_speaker = random.choice([0, 1])  # Randomly pick which speaker goes silent
        start = start_positions[i]
        end = min(start + silence_lens[i], lens)  # Ensure we stay in bounds

        # Apply silence to the selected speaker
        y[start:end, :, silent_speaker] = 0

    return y

def calc_speaker_ratio(xqf, J=CFG.Q, TH=1e-5):
    """
    Calculate the proportion of time frames where all speakers are active.

    Args:
        xqf (np.ndarray): [T, C, Q] time-domain signal for Q sources.
        J (int): Number of speakers.
        TH (float): Energy threshold to define activity.

    Returns:
        float: Ratio of overlapping frames among active frames.
    """
    energy = np.sum(np.abs(xqf), axis=1)
    active_speakers = energy > TH  # Binary mask (lens, Q)

    one_or_more_speakers = np.sum(active_speakers, axis=1) >= 1
    active_frame_count = np.sum(one_or_more_speakers)

    overlapping_count = np.sum(active_speakers, axis=1) == J
    overlap_ratio = np.sum(overlapping_count) / active_frame_count if active_frame_count > 0 else 0

    return overlap_ratio


def save_train_val_wav_signals(input_directory='dev-clean-test', output_base_directory='dev-wav-new-5', train_size=0.8,
                               sample_rate=16000, max_duration=30, delay_max=5, delay_min=3, num_gaps=5):
    """
        Generate training and validation WAV files by concatenating .flac utterances with silence gaps.

        Args:
            input_directory (str): Path to raw LibriSpeech speaker folders.
            output_base_directory (str): Output root for 'train' and 'val' folders.
            train_size (float): Ratio of speakers to assign to training.
            sample_rate (int): Target sample rate.
            max_duration (int): Max length (seconds) of each WAV file.
            delay_max (int): Max silence between utterances (seconds).
            delay_min (int): Min silence between utterances (seconds).
            num_gaps (int): Unused. Included for compatibility/future control.

        Outputs:
            Saves .wav files to train/ and val/ folders under output_base_directory.
        """
    random.seed(CFG.seed0)
    train_directory = os.path.join(output_base_directory, 'train')
    val_directory = os.path.join(output_base_directory, 'val')
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(val_directory, exist_ok=True)

    speakers = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]
    random.shuffle(speakers)
    split_index = int(train_size * len(speakers))
    train_speakers, val_speakers = speakers[:split_index], speakers[split_index:]

    for speaker_dir in tqdm(train_speakers):
        speaker_path = os.path.join(input_directory, speaker_dir)
        first_subdir = next(os.walk(speaker_path))[1][0]
        flac_path = os.path.join(speaker_path, first_subdir)
        concatenated_signal = concatenate_flac_with_gaps(flac_path, sample_rate, max_duration, delay_max, delay_min)
        output_wav_path = os.path.join(train_directory, f"{speaker_dir}.wav")
        sf.write(output_wav_path, concatenated_signal, sample_rate, format='WAV', subtype='PCM_16')
        print(f"Saved concatenated audio for speaker {speaker_dir} to {output_wav_path}")

    for speaker_dir in tqdm(val_speakers):
        speaker_path = os.path.join(input_directory, speaker_dir)
        first_subdir = next(os.walk(speaker_path))[1][0]
        flac_path = os.path.join(speaker_path, first_subdir)
        concatenated_signal = concatenate_flac_with_gaps(flac_path, sample_rate, max_duration, delay_max, delay_min)
        output_wav_path = os.path.join(val_directory, f"{speaker_dir}.wav")
        sf.write(output_wav_path, concatenated_signal, sample_rate, format='WAV', subtype='PCM_16')
        print(f"Saved concatenated audio for speaker {speaker_dir} to {output_wav_path}")
    print('Finished creating new WAV datafiles')

def generate_random_angle(used_angles, min_angle_difference=30):
    """
    Generate a random angle [0, 360) that differs by at least `min_angle_difference`
    from all angles in `used_angles`.
    """
    while True:
        angle = np.random.uniform(0, 360)
        if all(abs((angle - a + 180) % 360 - 180) >= min_angle_difference for a in used_angles):
            return angle

def generate_RIRs(room_length, room_width, mic_spacing, num_mics, min_angle_difference, radius,
                  num_of_RIRs, rev, angles=None):
    """
        Generate Room Impulse Responses (RIRs) for `num_of_RIRs` sources placed around a microphone array.

        Args:
            room_length (float): Length of room (in meters).
            room_width (float): Width of room (in meters).
            mic_spacing (float): Spacing between microphones (in meters).
            num_mics (int): Number of microphones.
            min_angle_difference (float): Minimum angular separation between sources.
            radius (float): Distance of sources from array center.
            num_of_RIRs (int): Number of source RIRs to generate.
            rev (float): Reverberation time (RT60 in seconds).
            angles (list or None): Optional fixed source angles in degrees.

        Returns:
            tuple: (List of RIRs, list of used angles in degrees)
        """
    middle_x = room_length / 2
    middle_y = room_width / 2
    mics = [[middle_x - (num_mics // 2 - i) * mic_spacing, middle_y, 1] for i in range(num_mics)]
    RIRs = []
    used_angles = []
    for i in range(num_of_RIRs):
        if angles is None:
            angle = generate_random_angle(used_angles, min_angle_difference)
        else: angle = angles[i]

        theta = np.deg2rad(angle)
        used_angles.append(angle)

        source_x = middle_x + radius * np.cos(theta)
        source_y = middle_y + radius * np.sin(theta)
        source_position = [source_x, source_y, 1]
        h = rir.generate(
            c=340,  # Sound velocity (m/s)
            fs=CFG.fs,  # Sample frequency (samples/s) from CFG
            r=mics,  # Receiver positions
            s=source_position,  # Current speaker's source position
            L=[room_length, room_width, 2.4],  # Room dimensions
            reverberation_time=rev,  # Reverberation time
            nsample=CFG.lenF0  # Number of output samples
        )
        RIRs.append(h)

    return RIRs, used_angles



def add_noise_to_signal(signal, noise, SNR):
    """
        Add scaled noise to a signal to achieve a specified SNR.

        Args:
            signal (np.ndarray): Clean signal [T] or [T, C].
            noise (np.ndarray): Noise signal (same shape or [T]).
            SNR (float): Desired signal-to-noise ratio in dB.

        Returns:
            np.ndarray: Noisy signal with specified SNR.
        """
    min_length = min(signal.shape[0], noise.shape[0])
    signal = signal[:min_length]
    noise = noise[:min_length]

    signal_power = np.var(signal**2, axis=0)
    noise_power = np.var(noise**2, axis=0)

    noise_scaling = np.sqrt(signal_power / (noise_power * 10**(SNR / 10)))

    if not len(noise.shape) == len(signal.shape):
        noise = noise[:,np.newaxis]

    return signal + noise_scaling * noise

def combine_speaker_signals(speakers_signals, RIRs, num_mics, J=2, overlap_demand=None,
                            add_noise=CFG.add_noise, SNR=CFG.SNR):
    """
       Simulate multichannel mixture by convolving source signals with RIRs,
       adding noise (optional), and computing STFTs.

       Args:
           speakers_signals (list): List of raw waveforms, one per speaker.
           RIRs (list): List of RIRs, one per speaker.
           num_mics (int): Number of microphones.
           J (int): Number of speakers.
           overlap_demand (float): Target overlap ratio (used for 2-speaker mixtures).
           add_noise (bool): Whether to add white noise.
           SNR (float): Signal-to-noise ratio (if noise is added).

       Returns:
           tuple: Xt, Tmask, f, t, xqf, Xq, overlap_ratio, low_energy_mask, low_energy_mask_time, x
       """
    lens = CFG.lens
    # Initialize the filtered signal array
    xqf = np.zeros((lens, num_mics, J))


    # Convolve each speaker's signal with the RIR
    for q, signal in enumerate(speakers_signals):
        h = RIRs[q]
        convolved_signal = np.zeros((CFG.lens, num_mics))  # Placeholder for convolved signal

        # Convolve with the RIR for each microphone
        for m in range(num_mics):
            # Apply convolution per microphone, ensuring 'same' mode to keep length
            convolved_signal[:, m] = ss.convolve(signal.squeeze(), h[:, m], mode='same')
            # Apply high-pass filter for each microphone
            xqf[:, m, q] = ss.filtfilt(CFG.highpass, 1, convolved_signal[:, m])


    if overlap_demand is not None and J==2:
        xqf = apply_silence_2_speakers(xqf, overlap_demand, num_silences=4)
    overlap_ratio = calc_speaker_ratio(xqf, J, TH=1e-5)
    # Sum the filtered signals across speakers for the seen mixture
    x = np.sum(xqf, axis=2)

    # Perform STFT on the filtered signals
    Xq = np.empty((CFG.NFFT // 2 + 1, CFG.N_frames, CFG.M, J), dtype=complex)
    noise = np.random.normal(size=(x.shape))
    N = np.empty((CFG.NFFT // 2 + 1, CFG.N_frames, CFG.M), dtype=complex)

    for m in range(num_mics):
        N[:, :, m] = stft(noise[:, m], nperseg=CFG.NFFT, noverlap=CFG.olap * CFG.NFFT, fs=CFG.fs)[2]
        for q in range(J):
            Xq[:, :, m, q] = stft(xqf[:, m, q], nperseg=CFG.NFFT, noverlap=CFG.olap * CFG.NFFT, fs=CFG.fs)[2]

    absql = np.abs(Xq[CFG.F0, :, 0, :])
    absNl = np.abs(N[CFG.F0, :, 0])

    if add_noise:
        Gn = np.sqrt(np.mean(np.var(xqf[:, 0, :], axis=0)) / np.var(noise[:, 0]) * 10 ** (-SNR / 10))
        x = x + Gn * noise


    # Compute STFT for the combined signal
    Xt = np.empty((CFG.NFFT // 2 + 1, CFG.N_frames, CFG.M), dtype=complex)
    for m in range(num_mics):
        f, t, Xt[:, :, m] = stft(x[:, m], nperseg=CFG.NFFT, noverlap=CFG.olap * CFG.NFFT, fs=CFG.fs)



    # Xq *= CFG.stft_scaling
    # Xt *= CFG.stft_scaling
    # Time-frequency mask to detect strongest source
    if J==3:
        TH = -150
    elif J==2:
        TH = -140
    Tmask = np.argmax(absql, axis=-1)
    if add_noise:
        Tmask = np.argmax(np.concatenate((absql, Gn * absNl[:, :, None]), axis=2), axis=-1)

    low_energy_mask = (20 * np.log10(np.abs(Xt[:, :, 0] + 1e-8)) <= TH) | (Xt[:, :, 0] == 0)
    Tmask[low_energy_mask] = J
    Tmask[:, np.sum(Tmask == J, axis=0) / (CFG.NFFT // 2 + 1) > 0.85] = J


    low_energy_mask_time = None
    if CFG.low_energy_mask_time:
        low_energy_mask_time = (20 * np.mean(np.log10(np.abs(Xt[:, :, 0] + 1e-8)), axis=0) <= TH) | (np.mean(Xt[:, :, 0], axis=0) == 0)
        print('Using P time TH mask...')

    return Xt, Tmask, f, t, xqf, Xq, overlap_ratio, low_energy_mask, low_energy_mask_time, x

def get_speaker_signals(dataset_path, previous_combinations=None, J=CFG.Q, speakers_list=None):
    """
    Selects CFG.Q unique speakers from the dataset folder, avoiding duplicate combinations.
    """
    speaker_signals = []
    if previous_combinations is None:
        previous_combinations = set()

    # Set random seed for reproducibility
    random.seed(CFG.seed0+1)

    audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
    speakers = [f.split('.')[0] for f in audio_files]  # Extract speaker IDs from filenames

    if len(speakers) < J:
            raise ValueError(f"Not enough speakers in the dataset to select {J} speakers.")

    # Try random combinations until a unique one is found
    unique_combination_found = False
    while not unique_combination_found:
        # Randomly select J speakers
        selected_speakers = random.sample(speakers, J)
        sorted_combination = tuple(sorted(selected_speakers))  # Sort to avoid order-based duplicates

        if sorted_combination not in previous_combinations:
            previous_combinations.add(sorted_combination)
            unique_combination_found = True

    if not speakers_list==None:
        selected_speakers = speakers_list

    for speaker in selected_speakers:
        audio_file = f"{speaker}.wav"
        audio_path = os.path.join(dataset_path, audio_file)

        if not os.path.exists(audio_path):
            print(f"Warning: {audio_path} not found.")
            continue

        # Load the audio signal
        # sample_rate, signal = wavfile.read(audio_path)
        signal, sample_rate = sf.read(audio_path, always_2d=True)
        if len(signal) > CFG.lens:
            # signal = signal[:CFG.lens]
            signal = signal[len(signal) // 2 - CFG.lens // 2:len(signal) // 2 + CFG.lens // 2]

        # Ensure the signal is 2D for processing
        if signal.ndim == 1:
            signal = signal[:, None]  # Convert 1D to 2D if needed

        # Ensure the signal is at the correct sampling rate (CFG.fs)
        if sample_rate != CFG.fs:
            print(f"Warning: {audio_path} has a different sample rate {sample_rate}. Expected: {CFG.fs}.")

        # Append the signal to the list
        speaker_signals.append(signal)

    return speaker_signals, previous_combinations, selected_speakers

def create_mixes_data_file(num_samples, input_directory='dev-wav-full', train_val_path='train', J=CFG.Q):
    """
        Simulate mixtures for training or validation and save W and P matrices to a joblib file.

        Args:
            num_samples (int): Number of mixture examples to generate.
            input_directory (str): Base folder with speaker WAVs.
            train_val_path (str): Subfolder name ('train' or 'val').
            J (int): Number of speakers per mixture.

        Output:
            Saves a joblib file with 'Ws' and 'Ps' lists of size `num_samples`.
        """
    dataset_path = os.path.join(input_directory, train_val_path)
    L = CFG.N_frames
    K = CFG.H_freqbands
    noise_dim = 0
    data_dict = {'Ws': [], 'Ps': []}

    previous_combinations = set()  # Keep track of used speaker combinations
    print("Generating RIRs..")
    num_of_RIRs = 20
    RIRs, _ = generate_RIRs(room_length=6, room_width=6, mic_spacing=0.3, num_mics=6, min_angle_difference=30, radius=2,
                  num_of_RIRs=num_of_RIRs)
    start = time.time()
    print(f"Creating simulated dataset, J = {J}, L = {L}, noise = {0}, for {num_samples} samples...")
    for _ in tqdm(range(num_samples)):
        speakers_signals, previous_combinations, paths = get_speaker_signals(dataset_path, previous_combinations)
        Xt, Tmask, f, t, xqf = combine_speaker_signals(speakers_signals, RIRs, num_mics=6)
        Hl, Fall, lenF, F = feature_extraction(Xt)
        Hln, W, E0, P, _ = calculate_W_U_realSimplex(Hl, Fall, Tmask, lenF, F, file=paths)

        data_dict['Ws'].append(W)
        data_dict['Ps'].append(P)

    filename = f'{CFG.simulated_data_path}/DataDict_J{J}_L{L}_{num_samples}_{CFG.data_mode}_{train_val_path}.joblib'

    joblib.dump(data_dict, filename, compress=('zlib', 3))

    time_elapsed = time.time() - start
    print(f"Created data file with {num_samples} samples, in {time_elapsed} seconds")
# create_data_file(2400, train_val_path='train')
# create_data_file(600, train_val_path='val')




def read_wsj_sample(base_dir_path, used_indices=set(), type='npz'):
    # Define subdirectories
    mixes_dir = os.path.join(base_dir_path, "mixes")
    ys_dir = os.path.join(base_dir_path, "ys")
    paras_dir = os.path.join(base_dir_path, "paras")

    # List all mix files
    mix_files = sorted([f for f in os.listdir(mixes_dir) if f.endswith('.npz')])

    if len(used_indices) >= len(mix_files):
        raise ValueError("All samples have been used. No unique samples left.")

    # Find a unique index
    while True:
        idx = random.randint(0, len(mix_files) - 1)
        if idx not in used_indices:
            used_indices.add(idx)
            break
    # Load mix
    mix_path = os.path.join(mixes_dir, f"mix_{idx}.{type}")
    if type == 'wav':
        mix, _ = sf.read(mix_path)
    else:
        mix = np.load(mix_path)["mix"]

    # Load ys
    ys_path = os.path.join(ys_dir, f"ys_{idx}.{type}")
    if type == 'wav':
        ys = sf.read(ys_path)
    else:
        ys = np.load(ys_path)["ys"]

    # Load paras
    paras_path = os.path.join(paras_dir, f"paras_{idx}.pkl")
    with open(paras_path, "rb") as f:
        paras = pickle.load(f)

    return mix, ys, paras

def extract_wsj0_features(mixed_signal, ys, num_mics, J=CFG.Q, pad=CFG.pad_flag):
    pad_size = CFG.pad_size
    ys_copy = ys.copy()
    if pad and pad_size > 0:
        ys_copy = np.pad(ys_copy, ((pad_size, pad_size), (0, 0), (0, 0)), mode='constant')
        mixed_signal = np.pad(mixed_signal, ((pad_size, pad_size), (0, 0)), mode='constant')
        # ys_copy = np.concatenate((ys_copy[:pad_size], ys_copy, ys_copy[-pad_size:]), axis=0) ### Padding with duplicates of the signals
        # mixed_signal = np.concatenate((mixed_signal[:pad_size], mixed_signal, mixed_signal[-pad_size:]), axis=0)

    if CFG.resample_flag:
        ys_copy = resample(ys, int(ys.shape[0] * CFG.sample_factor), axis=0)
        mixed_signal = resample(mixed_signal, int(mixed_signal.shape[0] * CFG.sample_factor), axis=0)

    # Perform STFT on the filtered signals
    Xq = np.empty((CFG.NFFT // 2 + 1, CFG.N_frames, CFG.M, J), dtype=complex)


    for m in range(num_mics):
        for q in range(J):
            _, _, Xq[:, :, m, q] = stft(ys_copy[:, m, q], nperseg=CFG.NFFT, noverlap=CFG.olap * CFG.NFFT, fs=CFG.fs)


    absql = np.abs(Xq[CFG.F0, :, 0, :])

    # Compute STFT for the combined signal
    Xt = np.empty((CFG.NFFT // 2 + 1, CFG.N_frames, CFG.M), dtype=complex)
    for m in range(num_mics):
        f, t, Xt[:, :, m] = stft(mixed_signal[:, m], nperseg=CFG.NFFT, noverlap=CFG.olap * CFG.NFFT, fs=CFG.fs)
    # Xq *= CFG.stft_scaling
    # Xq += 1e-8
    # Xt *= CFG.stft_scaling
    # Xt += 1e-8
    overlap_ratio = calc_speaker_ratio(ys, J, TH=1e-5)

    # Time-frequency mask to detect strongest source
    Tmask = np.argmax(absql, axis=-1)
    energy = np.abs(Xq[CFG.F0, :, :, :])  # F0 × T × M × J
    avg_energy = energy.mean(axis=2)  # average over M
    Tmask = np.argmax(avg_energy, axis=-1)  # F0 × T

    TH = -150
    low_energy_mask = (20 * np.log10(np.abs(Xt[:, :, 0] + 1e-8)) <= TH) | (Xt[:, :, 0] == 0)
    Tmask[low_energy_mask] = J
    low_energy_mask_time = (20 * np.mean(np.log10(np.abs(Xt[:, :, 0] + 1e-8)), axis=0) <= TH) | (
            np.mean(Xt[:, :, 0], axis=0) == 0)

    # Tmask[:, np.sum(Tmask == CFG.Q, axis=0) / (CFG.NFFT // 2 + 1) > 0.85] = CFG.Q
    return Xt, Tmask, f, t, ys, Xq, overlap_ratio, low_energy_mask, low_energy_mask_time, mixed_signal


if __name__ == "__main__":
    np.random.seed(CFG.seed0)
    random.seed(CFG.seed0)
    save_train_val_wav_signals(input_directory='dev-clean-test', output_base_directory='dev-wav-5-3', train_size=0.8,
                               sample_rate=16000,
                               max_duration=20, delay_max=5, delay_min=3, num_gaps=3)





