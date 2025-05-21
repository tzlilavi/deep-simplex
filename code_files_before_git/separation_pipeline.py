import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import CFG
import time

from CFG import use_local_deep
from functions import (
    process_signals, feature_extraction, calculate_W_U_realSimplex, calculate_SPA_simplex, save_plots,
    initialize_arrays, generate_test_positions, plot_heat_mat, plot_P_speakers, plot_results,
    plot_simplex, local_mapping, find_top_active_indices,
    dist_scores, audio_scores, calc_calibration_loss, calibration, save_np_arrays, calc_needed_audio_scores, plot3d_simplex
)
from utils import throwlow, MaskErr
from data_simulator import combine_speaker_signals, get_speaker_signals, generate_RIRs, read_wsj_sample, extract_wsj0_features
CFG.set_mode('unsupervised')
from unsupervised import run_global_model, run_model_unknown_J, run_local_model, global_method, deep_local_masking
from custom_losses import Unsupervised_Loss, SupervisedLoss, find_best_permutation_supervised, LocalLoss
from Transformer_model import AutoEncoder
from MiSiCNet import MiSiCNet2, MiSiCNet_QinvU
from LSTMs import BiLSTM_Att
import random
import pickle
from itertools import permutations
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


random.seed(CFG.seed0)
np.random.seed(CFG.seed0)

def run_pipeline(previous_combinations=None, J=CFG.Q, run_number=None, expert=CFG.expert, P_method=CFG.P_method, speakers=None, combined_data=None, overlap_demand=None, rev=CFG.low_rev,
                 add_noise=CFG.add_noise, signals_file='dev-wav-0/train', data_mode=CFG.data_mode):

    if overlap_demand == 0.5:
        overlap_demand = 0.3

        Xt, Tmask, f, t, xqf = process_signals(0, rev, 0, 0, 0, SNR)
    if CFG.data_mode == 'libri':
        if combined_data==None:
            # signals, _ = get_speaker_signals('dev-clean')
            RIRs, angles = generate_RIRs(room_length=6, room_width=6, mic_spacing=0.3, num_mics=6, min_angle_difference=30,
                          radius=2,
                          num_of_RIRs=J, rev=rev)
            signals, previous_combinations, speakers = get_speaker_signals(signals_file, previous_combinations, J, speakers_list=speakers)
            combined_data = combine_speaker_signals(signals, RIRs, num_mics=CFG.M, J=J, overlap_demand=overlap_demand, add_noise=add_noise)

        Xt, Tmask, f, t, xqf, Xq, overlap_ratio, low_energy_mask, low_energy_mask_time, x = combined_data
    elif CFG.data_mode == 'wsj0':
         ### wsj0_mix/dataset/sp_wsj/max4_21speakers_2mix/train, wsj0_mix/dataset/sp_wsj/frontend4_15speakers_2mix/train
        mix, y, par = read_wsj_sample('wsj0_mix/dataset/sp_wsj/frontend4_15speakers_2mix/train', previous_combinations)
        y = y.transpose(2,1,0)
        mix = mix.T
        combined_data = extract_wsj0_features(mix, y, num_mics=CFG.M, J=2, pad=True)
        Xt, Tmask, f, t, xqf, Xq, overlap_ratio, low_energy_mask, low_energy_mask_time, _ = combined_data


    Hl, Hlm, Hlf, Fall, lenF, F = feature_extraction(Xt)
    Hq = np.stack([feature_extraction(Xq[:, :, :, q])[1] for q in range(J)], axis=-1)


    Hln, W, E0, pr2, first_non0 = calculate_W_U_realSimplex(Hl, Fall, Tmask, lenF, F, J=J)
    if not run_number == None:
        print(f'Run_number: {run_number}')


    if CFG.data_mode=='wsj0':
        spk1 = par['spk1'].split('/')[-2]
        spk2 = par['spk2'].split('/')[-2]
        mix_index = par['index']
        print(f'Mix index: {mix_index}, Speakers: {[spk1, spk2]}')
        geo = par['rir']['arr_geometry']
        print(f'Array geometry: {geo}')
        # plt.plot(np.linspace(0, CFG.seconds, CFG.lens), xqf[:, 0, :])
        # plt.show()
        # plt.plot(t, low_energy_mask_time)
        # plt.show()
    else:
        print(f'Speakers: {speakers}')
        print(f'Sources angles relative to room center: {[int(angle) for angle in angles]}')
        print(f'{J} overlap ratio: {overlap_ratio}')
        # plt.plot(np.linspace(0, CFG.seconds, CFG.lens), xqf[:, 0, :])
        # plt.show()
        # plt.plot(t, low_energy_mask_time)
        # plt.show()

    pe, id0, ext0 = calculate_SPA_simplex(np.real(E0), pr2, J)

    Q_mat = E0[ext0, :J]
    #

    U_torch = torch.from_numpy(E0[:, :J + CFG.add_noise].real).float()
    SPA_Q_torch = torch.from_numpy(Q_mat.real).float()
    L = CFG.N_frames


    # model = MiSiCNet2(CFG.N_frames, out_dim=J, P_method=P_method).to(CFG.device)


    W_torch = torch.from_numpy(W).float()
    W_torch = W_torch.unsqueeze(0).to(CFG.device)
    input_mat = W_torch.clone().to(CFG.device)
    if CFG.random_input:
        print('Using random input...')
        uniform_noise = torch.distributions.Uniform(-0.1 ** 0.5, 0.1 ** 0.5).sample((1, L, L))
        gaussian_noise = torch.normal(mean=0, std=0.03 ** 0.5, size=(1, L, L))
        input_mat = uniform_noise + gaussian_noise

    P2 = None
    if P_method=='both':
        print('Running vertices model')
        deep_dict_global, P, A = global_method(input_mat, W_torch, first_non0, pr2, low_energy_mask_time, J=J,
                                               P_method='vertices', Hlf=Hlf, low_energy_mask=low_energy_mask,
                                               dropout=CFG.dropout, K_dropout=CFG.K_dropout, run_multiple_initializations=CFG.run_multiple_initializations,
                                               mask_input_ratio=CFG.mask_input_ratio, fixed_mask_input_ratio=CFG.fixed_mask_input_ratio,
                                               noise_col=CFG.noise_col, add_noise=CFG.add_noise,)

        # plot_results(P, pr2, pe, id0, J=J)

        chosen_P, pe_will_lose = P_expert(P, ext0, pe, min_speaker_TH=0.75, max_two_speakers_TH=0.9)

        top_indices = np.argsort(A, axis=0)[-3:][::-1]
        top_vals = np.sort(A, axis=0)[-3:][::-1]
        # np.set_printoptions(precision=3, suppress=True)
        # print(f'A top values:\n{top_vals}')
        #


        print('Running probabilistic model')
        start_time = time.time()
        deep_dict_global2, P2, _ = global_method(input_mat, W_torch, first_non0, pr2, low_energy_mask_time, J=J,
                                                 P_method='prob', Hlf=Hlf, low_energy_mask=low_energy_mask,
                                                 dropout=CFG.dropout, K_dropout=CFG.K_dropout, run_multiple_initializations=CFG.run_multiple_initializations,
                                                 mask_input_ratio=CFG.mask_input_ratio,
                                                 noise_col=CFG.noise_col, add_noise=CFG.add_noise,
                                                 fixed_mask_input_ratio=CFG.fixed_mask_input_ratio,)
        end_time = time.time()
        print(f"Probabilistic model ran in {end_time - start_time:.2f} seconds")
        # P2 = zero_P_below_TH(P2, TH=0.2)
        plot_results(P2, pr2, pe, id0, J=J, t=t, plot_flag=True, noise_P=deep_dict_global2['P_noise'])

        decision_vector = None
        # plot3d_simplex(pr2, top_indices, title='pr2 with Amodel top vertices Simplex', azim=30,
        #                elev=30)


    else:
        print(f'Running {P_method} model')
        start_time = time.time()
        A = None
        deep_dict_global, P, _ = global_method(input_mat, W_torch, first_non0, pr2, low_energy_mask_time, J=J,
                                                 P_method=P_method, Hlf=Hlf, low_energy_mask=low_energy_mask,
                                                 dropout=CFG.dropout, K_dropout=CFG.K_dropout,
                                                 run_multiple_initializations=CFG.run_multiple_initializations,
                                                 mask_input_ratio=CFG.mask_input_ratio,
                                                 noise_col=CFG.noise_col, add_noise=CFG.add_noise,
                                                 fixed_mask_input_ratio=CFG.fixed_mask_input_ratio, )
        end_time = time.time()
        print(f"{P_method} model ran in {end_time - start_time:.2f} seconds")

        plot_results(P, pr2, pe, id0, J=J, t=t, plot_flag=True,
                     noise_P=deep_dict_global['P_noise'])
        print(f'Speakers: {speakers}')
        print(f'Sources angles relative to room center: {[int(angle) for angle in angles]}')
        decision_vector = None


    data_dict = {'W': W, 'U': E0, 'pr2': pr2, 'ext0': ext0, 'id0': id0, 'pe': pe, 'P': P, 'A': A,
                 'deep_dict_global': deep_dict_global, 'speakers': speakers, 'combined_data': combined_data}

    model_tested = deep_dict_global['model_name']


    fh2, fh22, fh2_pe, fh = find_top_active_indices(pe, P, P2, pr2, P_method=P_method, add_noise=CFG.add_noise)

    Emask, Emask2, Emask_pe = local_mapping(pe, P, P2, pr2, Hlf, Xt, low_energy_mask, J, f, t, P_method, model_tested,
                                                                   Tmask, add_noise=add_noise, plot_Emask=False, )





    if P_method == 'both':
        deep_dict_local, deep_mask_soft, deep_mask_hard = deep_local_masking(Xt, P, Hlf, Tmask, Emask=Emask, P_method='vertices', plot_mask=True)
        deep_dict_local2, deep_mask_soft2, deep_mask_hard2 = deep_local_masking(Xt, P2, Hlf, Tmask, Emask=Emask2, P_method='prob', plot_mask=True)
    else:
        deep_dict_local, deep_mask_soft, deep_mask_hard = deep_local_masking(Xt, P, Hlf, Tmask, Emask=Emask,
                                                                             P_method=P_method, plot_mask=True, low_energy_mask=low_energy_mask)

    if P_method == 'both':
        dict_list = [{'P': P, 'local_mask': Emask, 'P_method': 'vertices', 'local_method': 'NN', 'fh2': fh2},
                     {'P': P2, 'local_mask': Emask2, 'P_method':'prob', 'local_method': 'NN', 'fh2': fh22}]
        show_best_global = True
    else:

        dict_list = [{'P': P, 'local_mask': Emask, 'P_method': P_method, 'local_method': 'NN', 'fh2': fh2}]
        show_best_global = False
        if use_local_deep:
            dict_list.append({'P': P, 'local_mask': deep_mask_hard, 'P_method': P_method, 'local_method': 'SpatialNet', 'fh2': fh2})
    dict_list.append({'P': pe, 'local_mask': Emask_pe, 'P_method': 'SPA', 'local_method': 'NN', 'fh2': fh2_pe})

    scores = calc_needed_audio_scores(dict_list, pr2, fh, Tmask, Hl, Xt, Hq, Xq, xqf, J=J, show_best_local=False, show_best_global=show_best_global, print_scores=True)



    # if False:
        # arrays_list = [P, P2, pe, pr2, Hlf, Tmask, Emask, Emask2, Emask_pe, fh2, fh22, fh2_pe, W, Xq, Xt, first_non0, low_energy_mask_time, low_energy_mask, xqf]#, par]
        # arrays_names_list = ['P', 'P2', 'pe', 'pr2', 'Hlf', 'Tmask', 'Emask', 'Emask2', 'Emask_pe', 'fh2', 'fh22', 'fh2_pe', 'W',
        #                      'Xq', 'Xt', 'first_non0', 'low_energy_mask_time', 'low_energy_mask', 'xqf']
        # arrays_dict = {name: arr for name, arr in zip(arrays_names_list, arrays_list)}
        #
        #
        # os.makedirs('array_data_global_tuning_full', exist_ok=True)
        # joblib.dump(arrays_dict, f"array_data_global_tuning_full/arrays_dict_mix{run_number}.pkl")

    model_decision = None
    model_decision_scores = None
    if P_method == 'both':
        model_decision = scores['si-sdr_vertices_NN'] > scores['si-sdr_prob_NN']  ## 1 if prob model won, else: 0
        model_decision_scores = [scores['si-sdr_vertices_NN'], scores['si-sdr_prob_NN']]
    decision_was_right = 0

    C_loss = None
    if CFG.calibrate:
        C_P_loss = calc_calibration_loss(P,pr2)
        C_P2_loss = None
        if P_method=='both':
            C_P2_loss = calc_calibration_loss(P2,pr2)
        C_pe_loss = calc_calibration_loss(pe,pr2)
        C_loss = {'C_P_loss':C_P_loss, 'C_P2_loss':C_P2_loss, 'C_pe_loss':C_pe_loss}

    return (data_dict, scores, decision_vector, model_decision, model_decision_scores, decision_was_right, C_loss, overlap_ratio)



def zero_P_below_TH(P, TH=0.2):
    P_copy = P.copy()

    P[P_copy < TH] = 0
    P[P_copy > 1 - TH] = 1
    return P



def model_speaker_count(J, W_input, W_torch, loss_function, E0, pr2, plot_flag=True):
    deep_dicts = {}
    print(f'Real speakers num is: {J}')
    loss_weights = np.array([2, 4, 8])
    for K in np.arange(3, 6):
        pe, _, ext = calculate_SPA_simplex(np.real(E0), pr2, K)
        SPA_Q_torch = torch.from_numpy(E0[ext, :K].T).float()
        U_torch = torch.from_numpy(E0[:, :K]).float()
        model = BiLSTM_Att(dim_output=K).to(CFG.device)
        # model = MiSiCNet2(CFG.N_frames, out_dim=K).to(CFG.device)
        deep_dicts[K] = run_global_model(model, W_input, W_torch, loss_function, CFG.epochs,
                                    param_search=CFG.param_search_flag)
        min_distance = float('inf')
        deep_dicts[K]['output_mat'] = deep_dicts[K]['output_mat'].squeeze(0).cpu().numpy()
        for perm in permutations(range(K)):
            # Permute the columns of P2
            pe_permuted = pe[:, perm]
            # Compute the Frobenius norm (or any distance metric)
            distance = np.linalg.norm(deep_dicts[K]['output_mat'] - pe_permuted, ord='fro')

            if distance < min_distance:
                min_distance = distance
                best_pe = pe_permuted
        deep_dicts[K]['SPA'] = best_pe
        deep_dicts[K]['SPA_deep_spearman'] = [
            spearmanr(deep_dicts[K]['output_mat'][:, i], deep_dicts[K]['SPA'][:, i])[0]
            for i in range(K)]
        deep_dicts[K]['SPA_deep_pearson'] = [pearsonr(deep_dicts[K]['output_mat'][:, i], deep_dicts[K]['SPA'][:, i])[0]
                                             for i in range(K)]

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
        fig.suptitle(f'{J} real speakers, {K} model speakers')
        ax1.plot(deep_dicts[K]['output_mat'][100:200, :])
        ax1.set_title('output')
        ax2.plot(deep_dicts[K]['SPA'][100:200, :], label='SPA')
        ax2.set_title('SPA')
        ax3.plot(pr2[100:200, :], label='real')
        ax3.set_title('real')
        fig.text(0.2, 0.02,
                 f"Weighted loss = {deep_dicts[K]['loss'] * loss_weights[K - 3]:.2f}, Corr = {np.mean(sorted(deep_dicts[K]['SPA_deep_pearson'], reverse=True)[:2]):.4f}",
                 fontsize=12)
        plt.tight_layout(rect=[0, 0.02, 1, 1])
        plt.show()
    # spearman_coeffs = np.array(
    #     [spearmanr(deep_dicts[5]['SPA'][:, j], deep_dicts[5]['output_mat'][:, j])[0] for j in range(5)])
    spa_deep_correlations = np.array([np.mean(sorted(deep_dicts[3]['SPA_deep_pearson'], reverse=True)[:2]),
                                      np.mean(sorted(deep_dicts[4]['SPA_deep_pearson'], reverse=True)[:2]),
                                      np.mean(sorted(deep_dicts[5]['SPA_deep_pearson'], reverse=True)[:2])])

    deep_losses = np.array([deep_dicts[3]['loss'], deep_dicts[4]['loss'], deep_dicts[5]['loss']])
    weighted_deep_losses = deep_losses * np.array([2, 4, 6])
    speakers_decision_arr = weighted_deep_losses * 1 / spa_deep_correlations
    chosen_speakers_num = np.argmin(speakers_decision_arr) + 3
    chose_real = chosen_speakers_num == J
    print(f'Speaker count was {chose_real}')
    return chose_real

def P_expert(P, ext0, pe, min_speaker_TH=0.8, max_two_speakers_TH=0.91):
    Q = P[ext0]

    max_speakers = np.max(Q, axis=1)
    if (max_speakers < min_speaker_TH).any(): #or np.sum(max_speakers>max_two_speakers_TH)<2:
        # print(f'Max speakers: {max_speakers} \n SPA will lose')
        # print(f'Q: {Q} \n SPA will lose')
        return P, True
    # print(f'Max speakers: {max_speakers} \n SPA will win')
    # print(f'Q: {Q} \n SPA will win')
    return pe, False

def extract_decision_features(W, P, P2, pe, top_vals, top_indices, ext0, deep_dict, deep_dict2, loss_function, J, print_results=False):
    Pvertices_Pprob_loss, _, _, Pvertices_Pprob_SAD, Pvertices_Pprob_RE = find_best_permutation_supervised(loss_function, deep_dict['output_mat'].to(CFG.device), deep_dict2['output_mat'].to(CFG.device))
    Pvertices_pe_loss, _, _, Pvertices_pe_SAD, Pvertices_pe_RE = find_best_permutation_supervised(loss_function, deep_dict['output_mat'].to(CFG.device), torch.from_numpy(pe).unsqueeze(0).float().to(CFG.device))
    Pprob_pe_loss, _, _, Pprob_pe_SAD, Pprob_pe_RE = find_best_permutation_supervised(loss_function, deep_dict2['output_mat'].to(CFG.device), torch.from_numpy(pe).unsqueeze(0).float().to(CFG.device))

    Pvertices_Pprob_pearson = np.mean([pearsonr(P[:, i], P2[:, i])[0] for i in range(J)])
    Pvertices_pe_pearson = np.mean([pearsonr(P[:, i], pe[:, i])[0] for i in range(J)])
    Pprob_pe_pearson = np.mean([pearsonr(P2[:, i], pe[:, i])[0] for i in range(J)])
    if print_results:
        print(f'Pvertices, Pprob:\nLoss:{Pvertices_Pprob_loss:.2f}, SAD: {Pvertices_Pprob_SAD:.2f}, RE:{Pvertices_Pprob_RE:.2f}, Pearson:{Pvertices_Pprob_pearson:.2f}')
        print(f'Pvertices, SPA:\nLoss:{Pvertices_pe_loss:.2f}, SAD: {Pvertices_pe_SAD:.2f}, RE:{Pvertices_pe_RE:.2f}, Pearson:{Pvertices_pe_pearson:.2f}')
        print(f'Pprob, SPA:\nLoss:{Pprob_pe_loss:.2f}, SAD: {Pprob_pe_SAD:.2f}, RE:{Pprob_pe_RE:.2f}, Pearson:{Pprob_pe_pearson:.2f}')
        print()
    def matrix_features(matrix):
        return np.mean(np.sort(P2[top_indices[0]], axis=0)[::-1], axis=1)[:-1]

    Q_PprobPverts_feats = matrix_features(P2[top_indices[0]])
    Q_Pvertspe_feats = matrix_features(P[ext0])
    Q_Pprobpe_feats = matrix_features(P2[ext0])

    Wmaxmean = W.mean(axis=0).max()
    l_J = []
    K = W.copy()
    K[range(CFG.N_frames), range(CFG.N_frames)] = 0
    W_max_element = K.max()
    for j in range(J):
        argmax_vector = K.argmax(axis=0)
        counts = np.bincount(argmax_vector)
        l_j = counts.argmax()
        l_J.append(l_j)
        K[l_j, :] = 0
        K[:, l_j] = 0
    W_leading_timeframes_mean = W[l_J].mean(axis=1)

    feature_vector = np.concatenate([[np.mean(top_vals[0]), np.mean(top_vals[1])],  # Top values mean
        [Pvertices_Pprob_loss.item(), Pvertices_Pprob_SAD.item(), Pvertices_Pprob_RE.item(), Pvertices_pe_loss.item(), Pvertices_pe_SAD.item(), Pvertices_pe_RE.item(),
         Pprob_pe_loss.item(), Pprob_pe_SAD.item(), Pprob_pe_RE.item(), Pvertices_Pprob_pearson, Pvertices_pe_pearson, Pprob_pe_pearson
         ,Wmaxmean, W_max_element],  # Metrics
        W_leading_timeframes_mean, Q_PprobPverts_feats, Q_Pvertspe_feats, Q_Pprobpe_feats])
    # feature_vector =  np.concatenate([[Wmaxmean, W_max_element], W_leading_timeframes_mean])

    return feature_vector

def decision_data_save(model_decision_results, path="decision_data.pkl"):
    if CFG.P_method == 'both' and True:
        with open(path, "wb") as f:
            pickle.dump(model_decision_results, f)

def train_decision_model(model=XGBClassifier(eval_metric='logloss'), path='decision_data.pkl', model_path = 'decision_model.pkl', scaler_path ='scaler.pkl',test_size=0.1):
    if isinstance(path, str):
        with open(path, 'rb') as f:
            data = pickle.load(f)

    # Extract features (decision_vector) and labels (model_decision)
    X = np.array([result['decision_vector'] for result in data])  # Features
    y = np.array([result['model_decision'] for result in data])  # Labels (True/False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)  # Fit on training data and transform
    X_test = scaler.transform(X_test)  # Transform test data using the same scaler
    # Train the model
    model.fit(X_train, y_train)
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test set: {accuracy:.4f}")
    # Save the trained model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Model saved as 'decision_model.pkl and 'scaler.pkl'")

    return model, accuracy

def infer_decision(decision_vector, model=None, model_path='decision_model.pkl', scaler_path='scaler.pkl'):
    if model is None:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Ensure the decision_vector is in the correct format (2D array for prediction)
    decision_vector = np.array(decision_vector).reshape(1, -1)
    decision_vector = scaler.transform(decision_vector)

    # Predict the class
    prediction = model.predict(decision_vector)[0]  # Returns the class (True/False)

    # Predict the probability
    probability = model.predict_proba(decision_vector)[0][int(prediction)]  # Probability for the predicted class

    return prediction, probability

def comparing_wins(wins_over_comparison_dict, scores, comparing_suff1, comparing_suff2):

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

    metric_names = [f"{metric}_{suffix}" for metric in metrics]
    avg_std_list = [
        f"{avg_results[m]:.3f} ± {std_results[m]:.3f}" for m in metric_names
    ]
    return avg_std_list

def compute_comparison_table(results, num_test_runs, comparison_pairs, method_suffixes):
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
    # sdr_ii_avg = df["SDRii"].mean()
    # sdr_ii_std = df["SDRii"].std()

    # print(f"SDRii: {sdr_ii_avg:.3f} ± {sdr_ii_std:.3f}")
    return comparison_table


def run_scenario(J, num_test_runs=30, overlap_demand=None, rev=CFG.low_rev, signals_path=None,
                 data_mode=CFG.data_mode, wsj_mode=None):
    reset_all_seeds()
    CFG.Q = J
    previous_combinations = set()

    results = {}


    right_decisions = 0

    data_vertices = []
    C_losses = []
    model_decision_results = []
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

    for i in tqdm(range(num_test_runs)):
        data_dict, scores, decision_vector, model_decision, model_decision_scores, decision_was_right, C_loss, overlap_ratio  = \
            run_pipeline(previous_combinations, run_number=i + 1, expert=CFG.expert, J=J,
                         P_method=CFG.P_method, overlap_demand=overlap, signals_file=signals_path, rev=rev, data_mode=data_mode)
        overlap_ratio_sum += overlap_ratio
        data_vertices.append(data_dict)
        C_losses.append(C_loss)
        right_decisions += decision_was_right
        # Append the scalar values to the results dictionary
        for k in scores.keys():
            if i==0:
                results[k] = []
            results[k].append(scores[k])

        if CFG.P_method == 'both' and False:
            model_decision_results.append({
                "decision_vector": decision_vector,
                "model_decision": model_decision,
                "si-sdr_probabilistic": model_decision_scores[0],
                "si-sdr_vertices": model_decision_scores[1],
            })

    if False:
        decision_data_save(model_decision_results, path="decision_data_new.pkl")
        model, accuracy = train_decision_model(test_size=0.05, path='decision_data_new.pkl', model_path = 'decision_model_new.pkl', scaler_path ='scaler_new.pkl')

    if CFG.calibrate:
        calibration(loss_list=C_losses)

    print('num_speakers:', J)
    print('num_experiments:', num_test_runs)
    print('overlap_measured:', overlap_ratio_sum / num_test_runs)
    print('rev:', rev)

    if CFG.P_method == 'both':
        comparison_pairs = [('prob_NN', 'vertices_NN'), ('prob_NN', 'SPA_NN')]
        method_suffixes = ["ideal", "prob_NN", "vertices_NN", "best_global_NN", "SPA_NN"]#, "Aux"]
    else:
        comparison_pairs = [('prob_NN', 'SPA_NN'), ('prob_SpatialNet','prob_NN')]
        method_suffixes = ["ideal", f"{CFG.P_method}_NN", "SPA_NN"]#, "Aux"]
        if CFG.use_local_deep:
            method_suffixes.append(f"{CFG.P_method}_SpatialNet")
    comparison_table = compute_comparison_table(results, num_test_runs, comparison_pairs,
                                                method_suffixes=method_suffixes)


    if signals_path is not None:
        results_dir = "experiments_results"

        # Create the results folder if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Define the file path
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



    # table = run_scenario(J=2, rev=CFG.low_rev, signals_path='dev-wav-0/train', num_experiments=100, overlap_demand=0.5)
    #
    # table = run_scenario(J=2, rev=CFG.low_rev, signals_path='dev-wav-0/train', num_experiments=100, overlap_demand=0.5)