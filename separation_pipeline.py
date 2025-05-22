import os
import random
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import CFG
from CFG import use_local_deep

from data_simulator import (
    combine_speaker_signals, extract_wsj0_features,
    generate_RIRs, get_speaker_signals, read_wsj_sample
)
from functions import (
    calc_needed_audio_scores, calculate_SPA_simplex, calculate_W_U_realSimplex,
    feature_extraction, find_top_active_indices, local_mapping,
    plot_heat_mat, plot_results, plot_simplex
)

from unsupervised import global_method


random.seed(CFG.seed0)
np.random.seed(CFG.seed0)

def run_pipeline(
    previous_combinations=None,
    J=CFG.Q,
    run_number=None,
    P_method=CFG.P_method,
    speakers=None,
    combined_data=None,
    overlap_demand=None,
    rev=CFG.low_rev,
    add_noise=CFG.add_noise,
    signals_file='dev-wav-0/train',
    data_mode=CFG.data_mode,
):
    """
    Runs one complete separation pipeline instance:
    - Loads or simulates data
    - Extracts features and correlation matrix W
    - Runs classic SPA and deep global models
    - Computes local masks and evaluates separation

    Args:
        previous_combinations (set): Track used speaker combinations (Libri).
        J (int): Number of sources/speakers.
        run_number (int): Index of run (for logging/debug).
        P_method (str): Global model variant to run ("prob", "vertices", or "both").
        speakers (list): Optional speaker IDs.
        combined_data (tuple): Precomputed data tuple (can skip simulation).
        overlap_demand (float): Overlap setting (typically 0.3–1).
        rev (float): Reverberation level.
        add_noise (bool): Whether to add white noise (based on SNR).
        signals_file (str): Path to audio data for Libri.
        data_mode (str): Either 'libri' or 'wsj0'.

    Returns:
        tuple:
            data_dict (dict): Intermediate variables (W, U, pr2, P, pe, etc.).
            scores (dict): Evaluation metrics (SI-SDR, PESQ, etc.).
            overlap_ratio (float): Measured speaker overlap for the run.
    """

    # Fix overlap setting for 0.5 if needed (compatibility to calculation)
    if overlap_demand == 0.5:
        overlap_demand = 0.3

    # --- Load or simulate data ---
    if data_mode == 'libri':
        if combined_data is None:
            RIRs, angles = generate_RIRs(
                room_length=6, room_width=6, mic_spacing=0.3,
                num_mics=6, min_angle_difference=30,
                radius=2, num_of_RIRs=J, rev=rev
            )
            signals, previous_combinations, speakers = get_speaker_signals(
                signals_file, previous_combinations, J, speakers_list=speakers
            )
            combined_data = combine_speaker_signals(
                signals, RIRs, num_mics=CFG.M, J=J,
                overlap_demand=overlap_demand, add_noise=add_noise
            )
        Xt, Tmask, f, t, xqf, Xq, overlap_ratio, low_energy_mask, low_energy_mask_time, x = combined_data

    elif data_mode == 'wsj0':
        mix, y, par = read_wsj_sample('wsj0_mix/dataset/sp_wsj/frontend4_15speakers_2mix/train', previous_combinations)
        y = y.transpose(2, 1, 0)
        mix = mix.T
        combined_data = extract_wsj0_features(mix, y, num_mics=CFG.M, J=2, pad=True)
        Xt, Tmask, f, t, xqf, Xq, overlap_ratio, low_energy_mask, low_energy_mask_time, _ = combined_data

    # --- Feature extraction ---
    Hl, Hlm, Hlf, Fall, lenF, F = feature_extraction(Xt)
    Hq = np.stack([feature_extraction(Xq[:, :, :, q])[1] for q in range(J)], axis=-1)
    Hln, W, E0, pr2, first_non0 = calculate_W_U_realSimplex(Hl, Fall, Tmask, lenF, F, J=J)

    if run_number is not None:
        print(f'Run_number: {run_number}')

    if data_mode == 'wsj0':
        spk1 = par['spk1'].split('/')[-2]
        spk2 = par['spk2'].split('/')[-2]
        print(f'Mix index: {par["index"]}, Speakers: {[spk1, spk2]}')
        print(f'Array geometry: {par["rir"]["arr_geometry"]}')
    else:
        print(f'Speakers: {speakers}')
        print(f'Sources angles relative to room center: {[int(angle) for angle in angles]}')
        print(f'{J} overlap ratio: {overlap_ratio}')

    # --- Classic SPA estimation ---
    pe, id0, ext0 = calculate_SPA_simplex(np.real(E0), pr2, J)
    Q_mat = E0[ext0, :J]
    U_torch = torch.from_numpy(E0[:, :J + CFG.add_noise].real).float()
    SPA_Q_torch = torch.from_numpy(Q_mat.real).float()

    # --- Prepare input matrix ---
    L = CFG.N_frames
    W_torch = torch.from_numpy(W).float().unsqueeze(0).to(CFG.device)
    input_mat = W_torch.clone()

    if CFG.random_input:
        print('Using random input...')
        uniform_noise = torch.distributions.Uniform(-0.1 ** 0.5, 0.1 ** 0.5).sample((1, L, L))
        gaussian_noise = torch.normal(mean=0, std=0.03 ** 0.5, size=(1, L, L))
        input_mat = uniform_noise + gaussian_noise

    # --- Global model inference ---
    P2 = None
    if P_method == 'both':
        print('Running vertices model')
        deep_dict_global, P, A = global_method(
            input_mat, W_torch, first_non0, pr2, low_energy_mask_time, J=J,
            P_method='vertices', Hlf=Hlf, low_energy_mask=low_energy_mask,
            dropout=CFG.dropout, K_dropout=CFG.K_dropout,
            run_multiple_initializations=CFG.run_multiple_initializations,
            mask_input_ratio=CFG.mask_input_ratio,
            fixed_mask_input_ratio=CFG.fixed_mask_input_ratio,
            noise_col=CFG.noise_col, add_noise=CFG.add_noise,
        )
        # plot_results(P, pr2, pe, id0, J=J)
        top_indices = np.argsort(A, axis=0)[-3:][::-1]
        top_vals = np.sort(A, axis=0)[-3:][::-1]
        # np.set_printoptions(precision=3, suppress=True)
        # print(f'A top values:\n{top_vals}')
        #

        print('Running probabilistic model')
        start_time = time.time()
        deep_dict_global2, P2, _ = global_method(
            input_mat, W_torch, first_non0, pr2, low_energy_mask_time, J=J,
            P_method='prob', Hlf=Hlf, low_energy_mask=low_energy_mask,
            dropout=CFG.dropout, K_dropout=CFG.K_dropout,
            run_multiple_initializations=CFG.run_multiple_initializations,
            mask_input_ratio=CFG.mask_input_ratio,
            fixed_mask_input_ratio=CFG.fixed_mask_input_ratio,
            noise_col=CFG.noise_col, add_noise=CFG.add_noise,
        )
        print(f"Probabilistic model ran in {time.time() - start_time:.2f} seconds")
        plot_results(P2, pr2, pe, id0, J=J, t=t, plot_flag=True, noise_P=deep_dict_global2['P_noise'])

        # plot3d_simplex(pr2, top_indices, title='pr2 with Amodel top vertices Simplex', azim=30, elev=30)
    else:
        print(f'Running {P_method} model')
        start_time = time.time()
        A = None
        deep_dict_global, P, _ = global_method(
            input_mat, W_torch, first_non0, pr2, low_energy_mask_time, J=J,
            P_method=P_method, Hlf=Hlf, low_energy_mask=low_energy_mask,
            dropout=CFG.dropout, K_dropout=CFG.K_dropout,
            run_multiple_initializations=CFG.run_multiple_initializations,
            mask_input_ratio=CFG.mask_input_ratio,
            fixed_mask_input_ratio=CFG.fixed_mask_input_ratio,
            noise_col=CFG.noise_col, add_noise=CFG.add_noise,
        )
        print(f"{P_method} model ran in {time.time() - start_time:.2f} seconds")
        plot_results(P, pr2, pe, id0, J=J, t=t, plot_flag=True, noise_P=deep_dict_global['P_noise'])

    # --- Local mask standard estimation ---

    fh2, fh22, fh2_pe, fh = find_top_active_indices(pe, P, P2, pr2, P_method=P_method, add_noise=CFG.add_noise)

    Emask, Emask2, Emask_pe = local_mapping(
        pe, P, P2, pr2, Hlf, Xt, low_energy_mask, J, f, t,
        P_method, deep_dict_global['model_name'], Tmask, add_noise=add_noise, plot_Emask=False
    )




    # --- Build list of mask strategies to evaluate ---
    dict_list = []
    show_best_global = False

    if P_method == 'both':
        dict_list.extend([
            {'P': P, 'local_mask': Emask, 'P_method': 'vertices', 'local_method': 'NN', 'fh2': fh2},
            {'P': P2, 'local_mask': Emask2, 'P_method': 'prob', 'local_method': 'NN', 'fh2': fh22},
        ])
        show_best_global = True
    else:
        dict_list.append({
            'P': P, 'local_mask': Emask, 'P_method': P_method, 'local_method': 'NN', 'fh2': fh2
        })
    dict_list.append({
        'P': pe, 'local_mask': Emask_pe, 'P_method': 'SPA', 'local_method': 'NN', 'fh2': fh2_pe
    })

    scores = calc_needed_audio_scores(
        dict_list, pr2, fh, Tmask, Hl, Xt, Hq, Xq, xqf, J=J,
        show_best_local=False, show_best_global=show_best_global, print_scores=True
    )
    data_dict = {
        'W': W, 'U': E0, 'pr2': pr2, 'ext0': ext0, 'id0': id0,
        'pe': pe, 'P': P, 'A': A, 'deep_dict_global': deep_dict_global,
        'speakers': speakers, 'combined_data': combined_data
    }
    return (data_dict, scores, overlap_ratio)


def comparing_wins(wins_over_comparison_dict, scores, comparing_suff1, comparing_suff2):
    """
       Compare two models' performance on each metric, and record which model wins for each.

       Args:
           wins_over_comparison_dict (dict): Running count of wins per metric (to be updated).
           scores (dict): Dictionary of scalar metrics for all methods in the current run.
           comparing_suff1 (str): Suffix of the first method (e.g., "prob_NN").
           comparing_suff2 (str): Suffix of the second method (e.g., "SPA_NN").

       Updates:
           wins_over_comparison_dict is updated in-place, incrementing a metric key
           if model 1 wins on that metric against model 2.
   """
    if scores[f"L2_P_{comparing_suff1}"] <= scores[f"L2_P_{comparing_suff2}"]:  # Smaller is better for L2
        wins_over_comparison_dict["L2_P"] += 1
    if scores[f"MD_{comparing_suff1}"] <= scores[f"MD_{comparing_suff2}"]:  # Smaller is better for MD
        wins_over_comparison_dict["MD"] += 1
    if scores[f"FA_{comparing_suff1}"] <= scores[f"FA_{comparing_suff2}"]:  # Smaller is better for FA
        wins_over_comparison_dict["FA"] += 1
    if scores[f"Err_{comparing_suff1}"] <= scores[f"Err_{comparing_suff2}"]:  # Larger is better for SDR
        wins_over_comparison_dict["Err"] += 1
    if scores[f"SDR_{comparing_suff1}"] >= scores[f"SDR_{comparing_suff2}"]:  # Larger is better for SDR
        wins_over_comparison_dict["SDR"] += 1
    if scores[f"si-sdr_{comparing_suff1}"] >= scores[f"si-sdr_{comparing_suff2}"]:  # Larger is better for si_sdr
        wins_over_comparison_dict["si-sdr"] += 1
    if scores[f"pesq_{comparing_suff1}"] >= scores[f"pesq_{comparing_suff2}"]:  # Larger is better for si_sdr
        wins_over_comparison_dict["pesq"] += 1
    if scores[f"stoi_{comparing_suff1}"] >= scores[f"stoi_{comparing_suff2}"]:  # Larger is better for si_sdr
        wins_over_comparison_dict["stoi"] += 1

def compute_avg_std(metrics, suffix, avg_results, std_results):
    """
        Formats the average and standard deviation values into strings for each metric.

        Args:
            metrics (list of str): Base metric names (e.g., "SDR", "L2_P", ...).
            suffix (str): Method identifier to match column keys (e.g., "prob_NN").
            avg_results (pd.Series): Series with mean values from all runs.
            std_results (pd.Series): Series with std deviation values from all runs.

        Returns:
            list of str: Formatted "AVG ± STD" strings for each metric.
    """
    metric_names = [f"{metric}_{suffix}" for metric in metrics]
    avg_std_list = [
        f"{avg_results[m]:.3f} ± {std_results[m]:.3f}" for m in metric_names
    ]
    return avg_std_list

def compute_comparison_table(results, num_test_runs, comparison_pairs, method_suffixes):
    """
        Computes a comparison table showing:
        - Average and std per metric per method
        - Win ratios comparing method pairs on each metric

        Args:
            results (dict): Dict of {metric_name: [values over runs]}.
            num_test_runs (int): Number of independent experiment runs.
            comparison_pairs (list of tuple): Pairs of method suffixes to compare.
            method_suffixes (list of str): All method suffixes to include in results.

        Returns:
            pd.DataFrame: Comparison table with AVG±STD and win ratios.
    """
    metrics = ["L2_P", "Err", "MD", "FA", "SDR", "si-sdr", "pesq", "stoi"]
    df = pd.DataFrame(results)


    avg_results = df.mean()
    std_results = df.std()


    method_scores = {}
    for suffix in method_suffixes:
        method_scores[suffix] = compute_avg_std(metrics, suffix, avg_results, std_results)

    win_counts = {pair: {m: 0 for m in metrics} for pair in comparison_pairs}

    for i in range(num_test_runs):
        scores = df.iloc[i].to_dict()
        for pair in comparison_pairs:
            if all(f"{metric}_{pair[0]}" in scores and f"{metric}_{pair[1]}" in scores for metric in metrics):
                comparing_wins(win_counts[pair], scores, *pair)

    win_ratios = {pair: [f"{win_counts[pair][m] / num_test_runs:.3f}" for m in metrics] for pair in comparison_pairs}

    comparison_table_data = {"Metric": metrics}

    for suffix, scores in method_scores.items():
        comparison_table_data[f"{suffix} Model (AVG ± STD)"] = scores

    for pair, ratios in win_ratios.items():
        comparison_table_data[f"Win Ratio {pair[0]} vs {pair[1]}"] = ratios

    comparison_table = pd.DataFrame(comparison_table_data)

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.max_columns', None)

    print(comparison_table)
    return comparison_table

def run_scenario(J, num_test_runs=30, overlap_demand=None, rev=CFG.low_rev, signals_path=None,
                 data_mode=CFG.data_mode, wsj_mode=None):
    """
       Runs multiple independent experiments and aggregates evaluation results.

       Each experiment runs `run_pipeline()` once with randomized data,
       and the results are averaged and compared using win ratios and standard metrics.

       Args:
           J (int): Number of speakers to separate.
           num_test_runs (int): How many independent runs to perform.
           overlap_demand (float): Desired speaker overlap (0.3, 0.5, or 1).
           rev (float): Reverberation level.
           signals_path (str): Where to load Libri data from (optional).
           data_mode (str): 'libri' or 'wsj0'.
           wsj_mode (optional): Reserved for future settings for WSJ0.

       Returns:
           pd.DataFrame: Comparison table of results with AVG±STD and win ratios.
   """
    reset_all_seeds()
    CFG.Q = J
    previous_combinations = set()

    results = {}

    data_vertices = []
    overlap_ratio_sum = 0

    if data_mode == 'libri':
        if overlap_demand==0.5:
            if J==2:
                signals_path = 'dev-wav-0/train'
                overlap = 0.3
            else:
                signals_path = 'dev-wav-full/train'
                overlap = None

        elif overlap_demand==1:
            overlap = None
            signals_path = 'dev-wav-0/train'
    elif data_mode == 'wsj0':
        overlap = overlap_demand
    saving_path = f'{data_mode}_{J}speakers_rev{str(rev).replace(".", "")}_olap{str(overlap_demand).replace(".", "")}_{num_exps}exp_new.xlsx'
    print('Running scenario...')
    print('data mode:', data_mode)
    print('J:', J)
    print('overlap_demand:', overlap_demand)
    print('signals_path:', signals_path)
    print('num_experiments:', num_test_runs)
    print('global epochs:', CFG.epochs)
    print('local epochs:', CFG.epochs_local)
    print('rev:', rev)
    if CFG.add_noise:
        print(f'white noise added, SNR: {CFG.SNR}dB')
    print('saving_path:', saving_path)

    # --- Main experiment loop ---
    for i in tqdm(range(num_test_runs)):
        data_dict, scores, overlap_ratio = run_pipeline(
            previous_combinations,
            run_number=i + 1,
            J=J,
            P_method=CFG.P_method,
            overlap_demand=overlap,
            signals_file=signals_path,
            rev=rev,
            data_mode=data_mode,
        )
        overlap_ratio_sum += overlap_ratio
        data_vertices.append(data_dict)
        # Append the scalar values to the results dictionary
        for k in scores.keys():
            if i==0:
                results[k] = []
            results[k].append(scores[k])

    # --- Summary statistics ---
    print('num_speakers:', J)
    print('num_experiments:', num_test_runs)
    print('overlap_measured:', overlap_ratio_sum / num_test_runs)
    print('rev:', rev)

    # --- Set up metric comparisons ---
    if CFG.P_method == 'both':
        comparison_pairs = [('prob_NN', 'vertices_NN'), ('prob_NN', 'SPA_NN')]
        method_suffixes = ["ideal", "prob_NN", "vertices_NN", "best_global_NN", "SPA_NN"]
    else:
        comparison_pairs = [('prob_NN', 'SPA_NN')]
        method_suffixes = ["ideal", f"{CFG.P_method}_NN", "SPA_NN"]

    comparison_table = compute_comparison_table(
        results, num_test_runs, comparison_pairs, method_suffixes=method_suffixes
    )

    # --- Save to Excel file ---
    if signals_path is not None:
        results_dir = "experiments_results"
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, saving_path)
        comparison_table.to_excel(file_path, index=False)
        print(f"File saved at: {file_path}")

    return comparison_table


def reset_all_seeds(seed=CFG.seed0):
    """Reset all relevant random seeds for full reproducibility."""
    random.seed(seed)  # Python's built-in RNG
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensuring deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For hash-based randomness
    os.environ["PYTHONHASHSEED"] = str(seed)



if __name__ == "__main__":
    speaker_nums = [3,2]
    revs = [CFG.low_rev, CFG.high_rev]
    overlaps = [1, 0.5]
    num_exps = 30

    # for J in speaker_nums:
    #     for rev in revs:
    #         for overlap in overlaps:
    #
    #             table = run_scenario(J=J, rev=rev, num_test_runs=num_exps, overlap_demand=overlap)
    table = run_scenario(J=3, rev=0.3, num_test_runs=num_exps, overlap_demand=1, data_mode='libri')

