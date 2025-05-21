import os
import numpy as np
import pandas as pd
import random
import CFG
from separation_pipeline import run_pipeline



def random_search_transformer(n_tries):
    best_sir1_diff = float('inf')
    best_sir2_diff = float('inf')
    best_params1 = None
    best_params2 = None
    results = []

    for i in range(n_tries):
        print(f'Try num: {i+1}')
        global CFG
        CFG.lr = random.choice(CFG.lrs)

        CFG.lr = random.choice(CFG.lrs)
        CFG.center_factor = random.choice(CFG.center_factors)
        CFG.reg_factor = random.choice(CFG.reg_factors)
        CFG.SAD_factor = random.choice(CFG.SAD_factors)
        CFG.L2_factor = random.choice(CFG.L2_factors)


        SIRi, SIR, deep_L2 = run_pipeline()
        SIR_diff = SIRi-SIR

        if SIR_diff < best_sir_diff1:
            best_sir_diff1 = SIR_diff
            best_params2 = best_params1

            best_sir1 = SIR_diff
            best_params1 = (CFG.lr, CFG.center_factor, CFG.reg_factor, CFG.SAD_factor, CFG.L2_factor)
            print(f"New best SIR: {best_sir_diff1} with params: {best_params1}, deep_L2: {deep_L2}")

        elif SIR_diff < best_sir2_diff:
            best_sir_diff2 = SIR_diff
            best_params2 = (CFG.lr, CFG.center_factor, CFG.reg_factor, CFG.SAD_factor, CFG.L2_factor)
            print(f"New second best SIR: {best_sir_diff2} with params: {best_params2}, deep_L2: {deep_L2}")

        results.append({
            'try': i + 1,
            'lr': CFG.lr,
            'center_factor': CFG.center_factor,
            'reg_factor': CFG.reg_factor,
            'SAD_factor': CFG.SAD_factor,
            'L2_factor': CFG.L2_factor,
            'SIRi-SIR': SIR_diff,
            'deep_L2': deep_L2
        })
def random_search_transformer_2(n_tries):
    best_sir_diff1 = float('inf')
    best_sir_diff2 = float('inf')
    best_params1 = None
    best_params2 = None
    results = []

    for i in range(n_tries):
        print(f'Try num: {i + 1}')
        global CFG
        CFG.lr = random.choice(CFG.lrs)
        CFG.center_factor = random.choice(CFG.center_factors)
        CFG.reg_factor = random.choice(CFG.reg_factors)
        CFG.SAD_factor = random.choice(CFG.SAD_factors)
        CFG.L2_factor = random.choice(CFG.L2_factors)
        CFG.model1_factor = random.choice(CFG.model1_factors)
        CFG.Qcenter_factor = random.choice(CFG.Qcenter_factors)
        CFG.SAD2_factor = random.choice(CFG.SAD2_factors)
        CFG.L22_factor = random.choice(CFG.L22_factors)
        CFG.model_2_epoch_TH = random.choice(CFG.model_2_epoch_THs)

        SIRi, SIR, deep_L2 = run_pipeline()
        SIR_diff = SIRi - SIR

        if SIR_diff < best_sir_diff1:
            best_sir_diff2 = best_sir_diff1
            best_params2 = best_params1

            best_sir_diff1 = SIR_diff
            best_params1 = (CFG.lr, CFG.center_factor, CFG.reg_factor, CFG.SAD_factor, CFG.L2_factor,
                            CFG.model1_factor, CFG.Qcenter_factor, CFG.SAD2_factor, CFG.L22_factor,
                            CFG.model_2_epoch_TH)
            print(f"New best SIR_diff: {best_sir_diff1} with params: {best_params1}, deep_L2: {deep_L2}")

        elif SIR_diff < best_sir_diff2:
            best_sir_diff2 = SIR_diff
            best_params2 = (CFG.lr, CFG.center_factor, CFG.reg_factor, CFG.SAD_factor, CFG.L2_factor,
                            CFG.model1_factor, CFG.Qcenter_factor, CFG.SAD2_factor, CFG.L22_factor,
                            CFG.model_2_epoch_TH)
            print(f"New second best SIR_diff: {best_sir_diff2} with params: {best_params2}, deep_L2: {deep_L2}")

        results.append({
            'try': i + 1,
            'lr': CFG.lr,
            'center_factor': CFG.center_factor,
            'reg_factor': CFG.reg_factor,
            'SAD_factor': CFG.SAD_factor,
            'L2_factor': CFG.L2_factor,
            'model1_factor': CFG.model1_factor,
            'Qcenter_factor': CFG.Qcenter_factor,
            'SAD2_factor': CFG.SAD2_factor,
            'L22_factor': CFG.L22_factor,
            'model_2_epoch_TH': CFG.model_2_epoch_TH,
            'SIRi-SIR': SIR_diff,
            'deep_L2': deep_L2
        })
    df = pd.DataFrame(results)
    df.to_excel('param_search_2models.xlsx', index=False)

# random_search_transformer(n_tries=CFG.param_tries)
random_search_transformer_2(n_tries=CFG.param_tries)