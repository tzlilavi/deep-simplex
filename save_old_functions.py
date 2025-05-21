



def audio_scores(pe, pr2, P, P2, Tmask, Emask, Emask2, Emask_pe, Hl, Xt, Hq, Xq, xqf, fh2, fh22, fh2_pe, P_method, calc_SPA_scores=True):

    Q, olap, lens, att, fs = CFG.Q, CFG.olap, CFG.lens, CFG.att, CFG.old_fs

    Pmask = (CFG.Q + 2) * np.ones((CFG.NFFT // 2 + 1, CFG.N_frames))
    fq = []

    Pmask2 = (CFG.Q + 2) * np.ones((CFG.NFFT // 2 + 1, CFG.N_frames))
    fq2 = []

    Pmask_pe = (CFG.Q + 2) * np.ones((CFG.NFFT // 2 + 1, CFG.N_frames))
    fq_pe = []
    Np = 10
    sisdr2 = None
    SIR2 = None
    SDR2 = None

    for q in range(CFG.Q + CFG.add_noise):
        srow = np.argsort(P[:, q])[::-1]
        fq.append(srow[:Np])
        Pmask[:, fq[q]] = q

        srow_pe = np.argsort(pe[:, q])[::-1]
        fq_pe.append(srow_pe[:Np])
        Pmask_pe[:, fq_pe[q]] = q

        if P_method == 'both':
            srow2 = np.argsort(P2[:, q])[::-1]
            fq2.append(srow2[:Np])
            Pmask2[:, fq2[q]] = q

    if CFG.add_noise == 1:
        # MD and FA for model
        MD, FA = MaskErr(Tmask, Emask, CFG.Q)
        print(f'MD and FA model: {MD:.2f} {FA:.2f}\n')

        if calc_SPA_scores:
            # MD and FA for SPA
            MD_pe, FA_pe = MaskErr(Tmask, Emask_pe, CFG.Q)
            print(f'MD and FA SPA: {MD_pe:.2f} {FA_pe:.2f}\n')

        # SDRp and SIRp for model
        SDRp, SIRp, _ = beamformer(Xt, Pmask, xqf, Q, fq, olap, lens, 0.01, 0.01, 0, CFG.fs)
        print(f'SDRp and SIRp model: {SDRp:.2f} {SIRp:.2f}\n')



        if calc_SPA_scores:
            # SDRp and SIRp for SPA
            SDRp_pe, SIRp_pe, _ = beamformer(Xt, Pmask_pe, xqf, Q, fq_pe, olap, lens, 0.01, 0.01, 0,
                                                                     CFG.fs)
            print(f'SDRp and SIRp SPA: {SDRp_pe:.2f} {SIRp_pe:.2f}\n')



        # SDRi and SIRi (shared line before model and SPA)
        SDRi, SIRi, yi = beamformer(Xt, Tmask, xqf, Q, fh2, olap, lens, 0.0001, 0.0001, 1,
                                                            CFG.fs, CFG.att)


        print(f'SDRi and SIRi: {SDRi:.2f} {SIRi:.2f}')


        # SDR and SIR for model
        SDR, SIR, ym = beamformer(Xt, Emask, xqf, Q, fh2, olap, lens, 0.01, 0.01, 1, CFG.fs,
                                                          CFG.att)
        print(f'SDR and SIR model: {SDR:.2f} {SIR:.2f}')

        sisdr = si_sdr(yi, ym)
        print(f'si-sdr model: {sisdr:.2f}')


        if calc_SPA_scores:
            # SDR and SIR for SPA
            SDR_pe, SIR_pe, ym_pe = beamformer(Xt, Emask_pe, xqf, Q, fh2_pe, olap, lens, 0.01, 0.01,
                                                                       1, CFG.fs, CFG.att)
            print(f'SDR and SIR SPA: {SDR_pe:.2f} {SIR_pe:.2f}')

            sisdr_pe = si_sdr(yi, ym_pe)
            print(f'si-sdr SPA: {sisdr_pe:.2f}')


    else:
        # MD and FA for model
        MD, FA = MaskErr(Tmask, Emask, CFG.Q)
        if P_method == 'both':
            MD2, FA2 = MaskErr(Tmask, Emask2, CFG.Q)
            print(f'MD and FA vertices model: {MD:.2f} {FA:.2f}')
            print(f'MD and FA prob model: {MD2:.2f} {FA2:.2f}')
        else:
            print(f'MD and FA {P_method} model: {MD:.2f} {FA:.2f}')

        if calc_SPA_scores:
            # MD and FA for SPA
            MD_pe, FA_pe = MaskErr(Tmask, Emask_pe, CFG.Q)
            print(f'MD and FA SPA: {MD_pe:.2f} {FA_pe:.2f}\n')

        C_P, C_P2, C_pe = (None, None, None)
        if CFG.beamformer_type[-1] =='C':
            calib = np.load('calib.npz')
            alpha = 0.1
            lhat_P = calib['lhats_P'][1]
            lhat_pe = calib['lhats_pe'][1]
            C_P = P > lhat_P
            if P_method == 'both':
                lhat_P2 = calib['lhats_P2'][1]
                C_P2 = smoother(P2 > lhat_P2)
            C_pe = smoother(pe > lhat_pe)




        # SDRi and SIRi (shared line before model and SPA)
        SDRi, sisdri, yi = beamformer_nonoise(Xt, Tmask, xqf, Q, fh2, olap, lens, 0.0001, 1,
                                                                    CFG.fs, CFG.att)


        SDRii, sisdrii, yii = beamformer_nonoise(Xt, None, xqf, Q, fh2, olap, lens, 0.0001, 0,
                                                 CFG.fs, CFG.att, Xq, Hq, Hl)

        # SDRi, sisdri, yi = MVDR_over_speakers(Xt.transpose(2,0,1), Xq.transpose(2,0,1,3), xqf, fs, olap, Q=CFG.Q, ref_mic=0)
        # SDRii, sisdrii, yii = MVDR_over_speakers(Xt.transpose(2,0,1), Xq.transpose(2,0,1,3), xqf, fs, olap, Q=CFG.Q, ref_mic=0, give_target=True)

        print(f'SDRi and SI-SDRi: {SDRi:.2f} {sisdri:.2f}')
        print(f'SDRii and SI-SDRii: {SDRii:.2f} {sisdrii:.2f}')



        # SDR and SIR for model
        SDR, sisdr, ym = beamformer_nonoise(Xt, Emask, xqf, Q, fh2, olap, lens, 0.01, 1, CFG.fs,
                                          CFG.att, C=C_P)

        # sisdr = si_sdr(xqf[:, 0, :], ym)
        ui = np.nan_to_num(np.asarray(xqf[:, 0, :].real, dtype=np.float32))
        um = np.nan_to_num(np.asarray(ym.real, dtype=np.float32))
        sti = np.mean([stoi(ui[:, j], um[:, j], fs) for j in range(Q)])
        psq = calc_psq(ui, um)



        if P_method == 'both':
            SDR2, sisdr2, ym2 = beamformer_nonoise(Xt, Emask2, xqf, Q, fh22, olap, lens, 0.01, 1, CFG.fs,
                                              CFG.att, C=C_P2)
            print(f'SDR and SI-SDR vertices model: {SDR:.2f} {sisdr:.2f}')
            print(f'SDR and SI-SDR prob model: {SDR2:.2f} {sisdr2:.2f}')

            um2 = np.nan_to_num(np.asarray(ym2.real, dtype=np.float32))
            sti2 = np.mean([stoi(ui[:, j], um2[:, j], fs) for j in range(Q)])
            psq2 = calc_psq(ui, um2)

            # print(f'stoi vertices model: {sti:.3f}')
            # print(f'stoi prob model: {sti2:.3f}')
            #
            # print(f'pesq vertices model: {psq:.3f}')
            # print(f'pesq prob model: {psq2:.3f}')

        else:

            print(f'SDR and SI-SDR {P_method} model: {SDR:.2f} {sisdr:.2f}')
            # print(f'stoi {P_method} model: {sti:.3f}')
            # print(f'pesq {P_method} model: {psq:.3f}')

        if calc_SPA_scores:
            # SDR and SIR for SPA
            SDR_pe, sisdr_pe, ym_pe = beamformer_nonoise(Xt, Emask_pe, xqf, Q, fh2_pe, olap, lens,
                                                                               0.01, 1, CFG.fs, CFG.att, C=C_pe)
            print(f'SDR and SI-SDR SPA: {SDR_pe:.2f} {sisdr_pe:.2f}')
            # #
            # sisdr_pe = si_sdr(yi, ym_pe)

            um_pe = np.nan_to_num(np.asarray(ym_pe.real, dtype=np.float32))
            sti_pe = np.mean([stoi(ui[:, j], um_pe[:, j], fs) for j in range(Q)])
            psq_pe = calc_psq(ui, um_pe)

            # print(f'stoi SPA: {sti_pe:.3f}')
            # print(f'pesq SPA: {psq_pe:.3f}')


    if CFG.write:

        os.makedirs(CFG.wav_folder, exist_ok=True)
        sf.write(os.path.join(CFG.wav_folder, f'ideal_speaker_{q}_clean.wav'), xqf[:, 0, q].real, CFG.old_fs)

        sf.write(os.path.join(CFG.wav_folder, f'ideal_speaker_{q}_beamformed.wav'), yi[:, q].real, CFG.old_fs)

        sf.write(os.path.join(CFG.wav_folder, f'est_speaker_{q}_SPA.wav'), ym_pe[:, q].real, CFG.old_fs)

        if P_method == 'both':
            sf.write(os.path.join(CFG.wav_folder, f'est_speaker_{q}_vertices.wav'), ym[:, q].real, CFG.old_fs)

            sf.write(os.path.join(CFG.wav_folder, f'est_speaker_{q}_probs.wav'), ym2[:, q].real, CFG.old_fs)

            np.savez(os.path.join(CFG.wav_folder, f'P_mats_{P_method}.npz'), p_est=pe, p_true=pr2, P=P, P2=P2)
        else:
            sf.write(os.path.join(CFG.wav_folder, f'est_speaker_{q}_{P_method}.wav'), ym[:, q].real, CFG.old_fs)

            np.savez(os.path.join(CFG.wav_folder, f'P_mats_{P_method}.npz'), p_est=pe, p_true=pr2, P=P)



    if not calc_SPA_scores:
        MD_pe, FA_pe, SDR_pe, SIR_pe, SDRp_pe, SIRp_pe, sisdr_pe= np.zeros(5)

    scores = {"MD": MD, "FA": FA, "SDR": SDR, "si-sdr": sisdr, 'pesq':psq, 'stoi': sti,
              "MD_pe": MD_pe, "FA_pe": FA_pe, "SDR_pe": SDR_pe, "si-sdr_pe": sisdr_pe, 'pesq_pe':psq_pe, 'stoi_pe': sti_pe,
              "SDRi": SDRi}
    if P_method =='both':
        scores = {"MD": MD, "FA": FA, "SDR": SDR, "si-sdr": sisdr, 'pesq':psq, 'stoi': sti,
                  "MD2": MD2, "FA2": FA2, "SDR2": SDR2, "si-sdr2": sisdr2, 'pesq2':psq2, 'stoi2': sti2,
                  "MD_best": min(MD, MD2), "FA_best": min(FA, FA2), "SDR_best": max(SDR, SDR2),
                  "si-sdr_best": max(sisdr, sisdr2), 'pesq_best':max(psq, psq2), 'stoi_best': max(sti, sti2),
                  "MD_pe": MD_pe, "FA_pe": FA_pe, "SDR_pe": SDR_pe,
                  "si-sdr_pe": sisdr_pe, 'pesq_pe':psq_pe, 'stoi_pe': sti_pe,
                  "SDRi": SDRi, "SI-SDRi": sisdri, "SDRii": SDRii, "SI-SDRii": sisdrii}
    return scores


class TwoModelsLoss(nn.Module):
    def __init__(self, J=CFG.Q, center_factor=CFG.center_factor, factor=0.35, weight_decay=1e-8):
        super(TwoModelsLoss, self).__init__()
        self.name = '2loss_SAD_L2_SAD2_center2'
        self.model1_loss = Unsupervised_Loss()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.SAD_loss = SAD()
        self.J = J
        self.center_factor = CFG.center_factor
        self.factor = factor
        self.weight_decay = weight_decay
        self.model1_losses = []
        self.model2_center_losses = []
        self.models_diff_losses = []
        self.losses = []


class QinvU_estimator(nn.Module):
    def __init__(self, weight_init, dropout=0):
        super(QinvU_estimator, self).__init__()
        self.fc = nn.Linear(weight_init.shape[0],weight_init.shape[1])
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            self.fc.weight.copy_(weight_init)

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        x = Functional.softmax(x, dim=-1)
        return x



    def model2_center_loss(self, u, model2):
        m = torch.mean(u, dim=0)  # 1xJ

        Q = torch.linalg.inv(model2.fc.weight)
        m_1 = torch.ger(m, torch.ones(Q.shape[0]))  # Jx1 * 1xJ = JxJ

        return self.mse_loss(Q, m_1)

    def forward(self, P_output, W_target, QinvU_P=None, U=None, model2=None, W_output=None, E_output=None):
        L = W_target.size(1)  # L
        model1_loss, SAD_loss, loss_RE, diagonal_loss, PPt_output = self.model1_loss(P_output, W_target, E_output=E_output)
        model2_center_loss = self.model2_center_loss(U, model2)
        QinvU_P = torch.unsqueeze(QinvU_P, dim=0)
        P_output = P_output[:,:,:self.J]

        L2_diff = self.mse_loss(P_output, QinvU_P)
        SAD_diff = self.SAD_loss(P_output, QinvU_P)

        model2_loss =  CFG.SAD_factor * SAD_diff + CFG.L2_factor * L2_diff + CFG.center_factor * model2_center_loss

        loss = model1_loss * CFG.two_models_factor + model2_loss * (1-CFG.two_models_factor)

        self.losses.append(loss.item())
        return loss, SAD_loss, loss_RE, diagonal_loss, PPt_output



class BiLSTM_Att_copy(nn.Module):
    def __init__(
            self, Q_weights=None,
            dim_input=CFG.N_frames,
            dim_output=CFG.Q + CFG.add_noise,
            mult_heads=False,
            activation="GELU",
            hidden_size=(256, 128),
            n_repeat_last_lstm=1,
            dropout=None,
            eps=1e-05, P_method='prob'
    ):
        super(BiLSTM_Att_copy, self).__init__()
        self.name = 'BiLSTM_Att'
        self.input_size = dim_input
        self.output_size = dim_output
        self.hidden_size = hidden_size
        self.mult_heads = mult_heads
        self.activation = activation
        self.dropout = CFG.dropout
        self.P_method = P_method


        self.Att = nn.MultiheadAttention(embed_dim=dim_input, num_heads=int(dim_input / 2))

        self.blstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size[0], batch_first=True,
                              bidirectional=True)  # type:ignore
        self.blstm2 = nn.LSTM(input_size=self.hidden_size[0] * 2, hidden_size=self.hidden_size[1], batch_first=True,
                              bidirectional=True, num_layers=n_repeat_last_lstm)  # type:ignore



        self.Conv2 = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(128, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.LeakyReLU(0.1, inplace=True))  ## [b, 128,L]

        self.Conv3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(64, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.LeakyReLU(0.1, inplace=True))  ## [b, 64,L]

        self.Conv4 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(32, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.LeakyReLU(0.1, inplace=True))  ## [b, 32,L]
        self.ConvSkip2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
                                       nn.BatchNorm1d(32, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                       nn.LeakyReLU(0.1, inplace=True))

        if dropout is not None:
            self.dropout1 = nn.Dropout(p=self.dropout)
            self.dropout2 = nn.Dropout(p=self.dropout)
            self.dropout3 = nn.Dropout(p=self.dropout)
            self.dropout4 = nn.Dropout(p=self.dropout)



        self.linear = nn.Linear(32, self.output_size)

        if CFG.unknown_J:
            self.linear2 = nn.Linear(32, 2)
            self.linear3 = nn.Linear(32, 3)
            self.linear4 = nn.Linear(32, 4)
            self.linear5 = nn.Linear(32, 5)

        if self.activation is not None and len(self.activation) > 0:  # type:ignore
            self.activation_func = getattr(nn, self.activation)()  # type:ignore
        else:
            self.activation_func = None
        # self.apply(self.init_weights)
        # for layer in self.children():
        #     self.init_weights(layer)
        self.init_weights()

        self.decoder = nn.Linear(in_features=3, out_features=CFG.N_frames, bias=False)

        if isinstance(Q_weights, dict):

            self.U_calculators = nn.ModuleDict({
                f"U_calculator_{J}": nn.Linear(in_features=J, out_features=J, bias=False)
                for J in Q_weights
            })
            with torch.no_grad():
                for J, Q_weight in Q_weights.items():
                    self.U_calculators[f"U_calculator_{J}"].weight.copy_(Q_weight.T)
        elif Q_weights is None:
            self.U_calculator = nn.Linear(in_features=dim_output, out_features=dim_output, bias=False)
        else:
            self.U_calculator = nn.Linear(in_features=dim_output, out_features=dim_output, bias=False)
            with torch.no_grad():
                self.U_calculator.weight.copy_(Q_weights.T)

        self.J_predictor = nn.Linear(in_features=3, out_features=CFG.N_frames, bias=False)

    def init_weights(self):
        torch.manual_seed(1)
        for layer in self.children():
            # Skip the attention layer
            if isinstance(layer, MultiHeadAttention):
                continue
            elif isinstance(layer, nn.LSTM):
                for name, param in layer.named_parameters():
                    if isinstance(param, torch.Tensor):  # Only initialize tensors
                        if "weight" in name:
                            nn.init.xavier_uniform_(param)  # Xavier initialization for weights
                        elif "bias" in name:
                            nn.init.constant_(param, 0)  # Zero initialization for biases

            elif isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    if isinstance(sub_layer, nn.Conv1d):
                        nn.init.xavier_uniform_(sub_layer.weight)  # Xavier initialization for Conv1d weights
                        if sub_layer.bias is not None:
                            nn.init.constant_(sub_layer.bias, 0)  # Zero initialization for Conv1d biases
                    elif isinstance(sub_layer, nn.BatchNorm1d):
                        nn.init.constant_(sub_layer.weight, 1)  # BatchNorm weights to 1
                        nn.init.constant_(sub_layer.bias, 0)  # BatchNorm biases to 0

            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for Linear weights
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # Zero initialization for Linear biases

    def forward(self, x, epoch=None):  ## [b, L , L]
        W = x.clone()
        B, L, _ = x.shape
        x, _ = self.Att(x, x, x)
        x = x.reshape(B * L, L)
        x, _ = self.blstm1(x)
        if self.dropout:
            x = self.dropout1(x)

        x, _ = self.blstm2(x)
        if self.dropout:
            x = self.dropout2(x)


        y = torch.unsqueeze(x.T, 0)
        y = self.Conv2(y)
        y_skip = self.ConvSkip2(y)
        y = self.Conv3(y)
        y = self.Conv4(y) + y_skip

        if self.mult_heads:
            P = {}
            U = {}
            for i in range(3, 6):
                linear_layer = getattr(self, f"linear{i}")
                y_linear = self.activation_func(linear_layer(torch.transpose(y, 1, 2)))
                P[i] = F.softmax(y_linear, dim=-1)
                U_calculator = self.U_calculators[f'U_calculator_{i}']
                U[i] = U_calculator(P[i])  # Compute and store U matrices


        else:
            y = self.activation_func(self.linear(torch.transpose(y, 1, 2)))
            y = y.reshape(B, L, self.output_size)
            if self.P_method=='prob':
                P = F.softmax(y, dim=-1)  ## [b, L , J+1]
                U = self.U_calculator(P)
                E = P.clone()
            elif self.P_method=='vertices':
                E = F.softmax(y, dim=-1)  ## [b, L , J+1]
                A = F.softmax(y.squeeze(0), dim=0)
                As = A.detach().cpu().numpy()
                top_vals = np.sort(As, axis=0)[-3:][::-1]
                P = torch.matmul(W.squeeze(0), A)
                P = P * (P > 0)
                P[P.sum(1) > 1, :] = P[P.sum(1) > 1, :] / P[P.sum(1) > 1, :].sum(1, keepdims=True)

                P = P.unsqueeze(0)
                U = A.clone()



        W = torch.transpose(torch.bmm(P, torch.transpose(P, 1, 2)), 1,2)
        W[:, range(CFG.N_frames), range(CFG.N_frames)] = 1




        return P, W, E, U


def run_2_models(model1, model2, W_input, W, U_input, num_epochs, lr, clip_grad_max=CFG.clip_grad_max):

    J = U_input.size(1)
    optimizer1 = optim.Adam(model1.parameters(), lr=lr, betas=(0.5,0.999))
    optimizer2 = optim.Adam(model2.parameters(), lr=lr, betas=(0.5,0.999))

    loss_func = TwoModelsLoss()

    scheduler1 = StepLR(optimizer1, step_size=10, gamma=0.9)
    scheduler2 = StepLR(optimizer2, step_size=10, gamma=0.9)

    for epoch in range(num_epochs):
        model1.train()
        model2.train()

        P_output, W_output, E_output = model1(W_input)
        QinvU_P = model2(U_input)

        loss, P_output, best_sad_loss, best_loss_RE, best_diagonal_loss, best_PPt = find_best_permutation_unsupervised(loss_func, P_output, W, QinvU_P=QinvU_P, U=U_input, model2=model2, W_output=W_output, E_output=E_output)
        optimizer1.zero_grad()

        if epoch < CFG.two_models_epoch_TH:
            optimizer2.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=clip_grad_max)
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=clip_grad_max)


        optimizer1.step()
        scheduler1.step()
        # optimizer2.step()
        # scheduler2.step()
        if epoch < CFG.two_models_epoch_TH:
            optimizer2.step()
            scheduler2.step()

        if not CFG.param_search_flag:
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, SAD_loss: {best_sad_loss.item()},"
                      f" loss_RE: {best_loss_RE.item()}, diagonal_loss = {best_diagonal_loss}")

    d = {}
    d['model_name'] = model1.name
    d['lr'] = CFG.lr
    d['epochs'] = num_epochs
    d['loss_name'] = loss_func.name

    d['loss'] = round(loss.item(), 4)
    # d['early_loss'] = round(early_loss.item(), 4)
    d['output_mat'] = P_output.detach()
    d['P_torch'] = P_output.detach()


    return d


def run_pipeline(previous_combinations=None, J=CFG.Q, run_number=None, expert=CFG.expert, P_method=CFG.P_method,
                 speakers=None, calc_SPA_scores=True, combined_data=None):
    # MD, FA, MD_pe, FA_pe, SDR, SIR, SDR_pe, SIR_pe , SDRi, SIRi, SDRiva, SIRiva, SDRp, SIRp, SDRp_pe, SIRp_pe, SNRin, den, sumin, spk, chr = initialize_arrays()
    # for rr, rev in enumerate(CFG.revs):
    #     for ss, SNR in enumerate(CFG.SNRs):
    spk = np.empty((CFG.Iter, CFG.Q), dtype=int)
    chr = np.empty((CFG.Iter, CFG.Q), dtype=int)
    generate_test_positions(spk, chr)
    if CFG.data_mode == 'real':
        rev = CFG.revs[0]
        SNR = CFG.SNR[0]

        Xt, Tmask, f, t, xqf = process_signals(0, rev, spk, chr, 0, SNR)
    elif CFG.data_mode == 'libri':
        if combined_data == None:
            # signals, _ = get_speaker_signals('dev-clean')
            RIRs, angles = generate_RIRs(room_length=6, room_width=6, mic_spacing=0.3, num_mics=6,
                                         min_angle_difference=30,
                                         radius=2,
                                         num_of_RIRs=J)
            signals, previous_combinations, speakers = get_speaker_signals('dev-wav-2/train', previous_combinations, J,
                                                                           speakers_list=speakers)
            combined_data = combine_speaker_signals_no_noise(signals, RIRs, num_mics=CFG.M, J=J)

        Xt, Tmask, f, t, xqf, Xq = combined_data
    elif CFG.data_mode == 'wsj0':
        mix, y, par = read_wsj_sample('wsj0_mix/dataset/sp_wsj/frontend4_15speakers_2mix/train', previous_combinations)
        y = y.transpose(2, 1, 0)
        mix = mix.T
        combined_data = extract_wsj0_features(mix, y, num_mics=CFG.M, J=2, pad=True)
        Xt, Tmask, f, t, Xq, xqf = combined_data

    Hl, Hlm, Hlf, Fall, lenF, F = feature_extraction(Xt)
    Hq = np.stack([feature_extraction(Xq[:, :, :, q])[1] for q in range(CFG.Q)], axis=-1)

    Hln, W, E0, pr2, first_non0 = calculate_W_U_realSimplex(Hl, Fall, Tmask, lenF, F, J=J)
    if not run_number == None:
        print(f'Run_number: {run_number}')

    # plot_heat_mat(W)
    if CFG.data_mode == 'wsj0':
        spk1 = par['spk1'].split('/')[-2]
        spk2 = par['spk2'].split('/')[-2]
        mix_index = par['index']
        print(f'Mix index: {mix_index}, Speakers: {[spk1, spk2]}')
        geo = par['rir']['arr_geometry']
        print(f'Array geometry: {geo}')
    else:
        print(f'Speakers: {speakers}')
        print(f'Sources angles relative to room center: {[int(angle) for angle in angles]}')
    if CFG.pad_flag:
        pe = np.zeros((CFG.N_frames, J))
        pe[CFG.pad_tfs:-CFG.pad_tfs], id0, ext0 = calculate_SPA_simplex(np.real(E0), pr2, J)
    else:
        pe, id0, ext0 = calculate_SPA_simplex(np.real(E0), pr2, J)
    Q_mat = E0[ext0, :J]
    pe[:, :CFG.Q] = pe[:, id0]

    U_torch = torch.from_numpy(E0[:, :J + CFG.add_noise].real).float()
    SPA_Q_torch = torch.from_numpy(Q_mat.real).float()
    J = Q_mat.shape[0]
    L = CFG.N_frames

    # model = MiSiCNet2(CFG.N_frames, out_dim=J, P_method=P_method).to(CFG.device)

    if P_method == 'both':
        print('Running vertices model')
        deep_dict_global, P, A = global_method(W, first_non0, pr2, J, P_method='vertices')

        # P = zero_P_below_TH(P, TH=0.2)
        plot_results(P, pr2, pe, id0)

        chosen_P, pe_will_lose = P_expert(P, ext0, pe, min_speaker_TH=0.75, max_two_speakers_TH=0.9)

        top_indices = np.argsort(A, axis=0)[-3:][::-1]
        top_vals = np.sort(A, axis=0)[-3:][::-1]
        # np.set_printoptions(precision=3, suppress=True)
        # print(f'A top values:\n{top_vals}')
        #

        print('Running probabilistic model')
        deep_dict_global2, P2, _ = global_method(W, first_non0, pr2, J, P_method='prob')

        # P2 = zero_P_below_TH(P2, TH=0.2)
        plot_results(P2, pr2, pe, id0)

        decision_vector = None
        # plot3d_simplex(pr2, top_indices, title='pr2 with Amodel top vertices Simplex', azim=30,
        #                elev=30)
        # plot3d_simplex(pr2, ext0, title='pr2 with pe top vertices Simplex',
        #                azim=30,
        #                elev=30)

        # decision_vector = extract_decision_features(W, P, P2, pe,
        #                                             top_vals, top_indices, ext0, deep_dict_global, deep_dict2, loss_function, J, print_results=False)

        # decision_model, _ = train_decision_model()

        # decision_vector = decision_vector[[11,12,15,18]]
        # chosen_model, confidence = infer_decision(decision_vector, model=None, model_path='decision_model_try.pkl')
        # chosen_model, confidence = infer_decision(decision_vector[15:16], model=None,
        #                                           model_path='decision_model_try.pkl', scaler_path='scaler_try.pkl')
        # print(chosen_model)
        # with open("decision_data_new.pkl", "rb") as f:
        #     data = pickle.load(f)
        #
        # for i in range(len(data)):
        #     data[i]['decision_vector'] = data[i]['decision_vector'][[11, 12, 15, 18]]
        # decision_data_save(data, path="decision_data_try.pkl")
        # decision_model_new, _ = train_decision_model(path='decision_data_try.pkl',
        #                                              model_path='decision_model_try.pkl',
        #                                              scaler_path='scaler_try.pkl', test_size=1)
        # xgb.plot_importance(decision_model_new, importance_type='gain')
        # plt.show()


    else:
        deep_dict_global, P, A = global_method(W, first_non0, pr2, J, P_method=P_method)

        plot_results(P, pr2, pe, id0)
        print(f'Speakers: {speakers}')
        print(f'Sources angles relative to room center: {[int(angle) for angle in angles]}')

        if P_method == 'vertices':
            top_indices = np.argsort(A, axis=0)[-3:][::-1]
            top_vals = np.sort(A, axis=0)[-3:][::-1]
            # np.set_printoptions(precision=3, suppress=True)
            # print(f'A top values:\n{top_vals}')
            # plot3d_simplex(pr2, top_indices, title='pr2 with Amodel top vertices Simplex', azim=30,
            #                elev=30)
            # plot3d_simplex(pr2, ext0, title='pr2 with pe top vertices Simplex',
            #                azim=30,
            #                elev=30)

        chosen_P, pe_will_lose = P_expert(P, ext0, pe, min_speaker_TH=0.75, max_two_speakers_TH=0.9)
        if expert:
            P = chosen_P.copy()

    if not P_method == 'both':
        P2 = None
    deep_L2, deep_mse, deep_L22, deep_mse2, SPA_L2, SPA_mse = dist_scores(pe, P, P2, pr2, J, P_method)

    data_dict = {'W': W, 'U': E0, 'pr2': pr2, 'ext0': ext0, 'id0': id0, 'pe': pe, 'P': P, 'A': A,
                 'deep_dict_global': deep_dict_global, 'speakers': speakers, 'combined_data': combined_data}

    model_tested = deep_dict_global['model_name']

    Emask, fh2, Emask2, fh22, Emask_pe, fh2_pe = local_mapping(pe, P, P2, Hlf, Xt, J, f, t, P_method, model_tested,
                                                               plot_Emask=False)

    deep_dict_local, deep_mask_soft, deep_mask_hard = deep_local_masking(Xt, P, Hlf, Emask, Tmask, P_method='vertices',
                                                                         plot_mask=True)
    deep_dict_local2, deep_mask_soft2, deep_mask_hard2 = deep_local_masking(Xt, P2, Hlf, Emask2, Tmask, P_method='prob',
                                                                            plot_mask=True)

    # scores = audio_scores(pe, pr2, P, P2, Tmask, deep_mask_hard, deep_mask_hard2, Emask_pe, Hl, Xt, Hq, Xq, xqf, fh2, fh22, fh2_pe, P_method, calc_SPA_scores=calc_SPA_scores)
    # scores = audio_scores(pe, pr2, P, P2, Tmask, Emask, Emask2, Emask_pe, Hl, Xt, Hq, Xq, xqf, fh2, fh22, fh2_pe, P_method, calc_SPA_scores=calc_SPA_scores)

    if False:
        arrays_list = [P, P2, pe, pr2, Hlf, Tmask, Emask, Emask2, Emask_pe, fh2, fh22, fh2_pe, W, Xq, Xt, mix, xqf, par]
        arrays_names_list = ['P', 'P2', 'pe', 'pr2', 'Hlf', 'Tmask', 'Emask', 'Emask2', 'Emask_pe', 'fh2', 'fh22',
                             'fh2_pe', 'W',
                             'Xq', 'Xt', 'mix', 'xqf', 'par']
        arrays_dict = {name: arr for name, arr in zip(arrays_names_list, arrays_list)}
        arrays_dict['SDR_vertices'] = scores['SDR']
        arrays_dict['si-sdr_vertices'] = scores['si-sdr']

        arrays_dict['SDR_prob'] = scores['SDR2']
        arrays_dict['si-sdr_prob'] = scores['si-sdr2']

        arrays_dict['SDR_best'] = scores['SDR_best']
        arrays_dict['si-sdr_best'] = scores['si-sdr_best']

        os.makedirs('array_data', exist_ok=True)
        joblib.dump(arrays_dict, f"array_data/arrays_dict_mix{mix_index}.pkl")

    scores['L2'] = deep_L2
    scores['L2_pe'] = SPA_L2
    if P_method == 'both':
        scores['L22'] = deep_L22
        scores['L2_best'] = min(deep_L2, deep_L22)
    deep_win_MD = 0
    deep_win_FA = 0

    if expert and not pe_will_lose:
        if scores['SDR'] > scores['SDR_pe'] or scores['SIR'] > scores['SIR_pe']:
            deep_win_MD = 1
            print("Deep win miss detection")
        return data_dict, scores, pe_will_lose, deep_win_MD, deep_win_FA
    elif expert and pe_will_lose:
        if scores['SDR'] < scores['SDR_pe'] or scores['SIR'] < scores['SIR_pe']:
            print("Deep win false alarm")
            deep_win_FA = 1
        return data_dict, scores, pe_will_lose, deep_win_MD, deep_win_FA

    model_decision = None
    model_decision_scores = None
    if P_method == 'both':
        model_decision = scores['si-sdr2'] > scores['si-sdr']  ## 1 if prob model won, else: 0
        model_decision_scores = [scores['si-sdr2'], scores['si-sdr']]
    decision_was_right = -100

    C_loss = None
    if CFG.calibrate:
        C_P_loss = calc_calibration_loss(P, pr2)
        C_P2_loss = None
        if P_method == 'both':
            C_P2_loss = calc_calibration_loss(P2, pr2)
        C_pe_loss = calc_calibration_loss(pe, pr2)
        C_loss = {'C_P_loss': C_P_loss, 'C_P2_loss': C_P2_loss, 'C_pe_loss': C_pe_loss}
    # if True:
    #     decision_was_right = chosen_model == model_decision
    #     print(f'Decision model chose {CFG.P_method[chosen_model]} with confdence: {confidence:.3f}, decision was {decision_was_right}')
    return (data_dict, scores,
            pe_will_lose, deep_win_MD, deep_win_FA, decision_vector, model_decision, model_decision_scores,
            decision_was_right, C_loss)

if __name__ == "__main__":
    previous_combinations = set()

    results = {}

    num_test_runs = CFG.num_test_runs
    SPA_will_lose_counts = 0
    deep_win_missed_counter = 0
    deep_win_false_counter = 0
    deep_model_wins_over_SPA = {metric: 0 for metric in ["L2", "MD", "FA", "SDR", "si-sdr", "pesq", "stoi"]}#"SDRp", "SIRp"]}
    right_decisions = 0
    if CFG.P_method == 'both':
        vertices_wins_over_prob  = deepcopy(deep_model_wins_over_SPA)
        best_deep_wins_over_SPA = deepcopy(deep_model_wins_over_SPA)
    data_vertices = []
    C_losses = []
    model_decision_results = []
    for i in range(num_test_runs):
        data_dict, scores, decision_vector, model_decision, model_decision_scores, decision_was_right, C_loss  = \
            run_pipeline(previous_combinations, run_number=i + 1, expert=CFG.expert, J=CFG.Q,
                         P_method=CFG.P_method, calc_SPA_scores=True)

        data_vertices.append(data_dict)
        C_losses.append(C_loss)
        SPA_will_lose_counts += SPA_will_loss
        deep_win_missed_counter += deep_win_missed
        deep_win_false_counter += deep_win_false
        right_decisions += decision_was_right
        # Append the scalar values to the results dictionary
        for k in scores.keys():
            if i==0:
                results[k] = []
            results[k].append(scores[k])
        # Determine wins for this run
        # Count deep wins for each metric
        if CFG.P_method == 'both':
            comparing_suff = '2'
            comparing_wins(vertices_wins_over_prob, scores, '', '2')
            comparing_wins(deep_model_wins_over_SPA, scores, '', '_pe')
            comparing_wins(best_deep_wins_over_SPA, scores, '_best', '_pe')
        else:
            comparing_wins(vertices_wins_over_prob, scores, '', '2')

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

    # Convert results to a DataFrame for easier averaging
    df = pd.DataFrame(results)

    # Calculate the average for each metric
    avg_results = df.mean()
    std_results = df.std()

    metrics = ["L2", "MD", "FA", "SDR", "si-sdr", "pesq", "stoi"]#"SDRp", "SIRp"]
    if CFG.P_method == 'both':
        vertices_win_ratios_over_prob = [f"{vertices_wins_over_prob[m] / num_test_runs:.3f}" for m in metrics]
        best_win_ratios_over_SPA = [f"{best_deep_wins_over_SPA[m] / num_test_runs:.3f}" for m in metrics]

    win_ratios_over_SPA = [f"{deep_model_wins_over_SPA[m] / num_test_runs:.3f}" for m in metrics]


    # deep_avg_std = [f"{avg_results[m]:.3f} ± {std_results[m]:.3f}" for m in metrics]
    # spa_metrics = ["L2_pe", "MD_pe", "FA_pe", "SDR_pe", "SIR_pe", "si-sdr_pe", "pesq_pe", "stoi_pe"]#"SDRp_pe", "SIRp_pe"]
    # spa_avg_std = [f"{avg_results[m]:.3f} ± {std_results[m]:.3f}" for m in spa_metrics]

    deep_avg_std = compute_avg_std(metrics, "", avg_results, std_results)
    spa_avg_std = compute_avg_std(metrics, "_pe", avg_results, std_results)

    if CFG.P_method=='both':
        # prob_metrics = ["L22", "MD2", "FA2", "SDR2", "SIR2", "si-sdr2", "pesq2", "stoi2"]  # "SDRp_pe", "SIRp_pe"]
        # prob_avg_std = [f"{avg_results[m]:.3f} ± {std_results[m]:.3f}" for m in prob_metrics]
        # best_metrics = ["L2_best", "MD_best", "FA_best", "SDR_best", "SIR_best", "si-sdr_best", "pesq_best", "stoi_best"]
        # best_avg_std = [f"{avg_results[m]:.3f} ± {std_results[m]:.3f}" for m in best_metrics]
        prob_avg_std = compute_avg_std(metrics, "2", avg_results, std_results)
        best_avg_std = compute_avg_std(metrics, "_best", avg_results, std_results)



        # Create a comparison table
        comparison_table = pd.DataFrame({
            "Metric": metrics,
            "Probabilistic Model (AVG ± STD)": prob_avg_std,
            "Vertices Model (AVG ± STD)": deep_avg_std,
            "Baseline Best Model (AVG ± STD)": best_avg_std,
            "SPA (AVG ± STD)": spa_avg_std,
            "Vertices Win Ratio over probabilistic": vertices_win_ratios_over_prob,
            "Baseline Win Ratio over SPA": best_win_ratios_over_SPA,
            "Vertices Win Ratio over SPA": win_ratios_over_SPA})

    else:
        comparison_table = pd.DataFrame({
            "Metric": metrics,
            "Probabilistic Model (AVG ± STD)": prob_avg_std,
            "Vertices Model (AVG ± STD)": deep_avg_std,
            "Baseline Best Model (AVG ± STD)": best_avg_std,
            "SPA (AVG ± STD)": spa_avg_std,
            "Vertices Win Ratio over probabilistic": vertices_win_ratios_over_prob,
            "Baseline Win Ratio over SPA": best_win_ratios_over_SPA,
            "Vertices Win Ratio over SPA": win_ratios_over_SPA})

    # Display the table
    pd.set_option('display.width', 1000)  # Increase the allowed width
    pd.set_option('display.max_colwidth', 1000)  # Allow unlimited column width
    pd.set_option('display.max_columns', None)  # Display all columns

    print(comparison_table)
    sdr_i_avg = df["SDRi"].mean()
    sdr_i_std = df["SDRi"].std()
    sisdr_i_avg = df["SI-SDRi"].mean()
    sisdr_i_std = df["SI-SDRi"].std()

    sdr_ii_avg = df["SDRii"].mean()
    sdr_ii_std = df["SDRii"].std()
    sisdr_ii_avg = df["SI-SDRii"].mean()
    sisdr_ii_std = df["SI-SDRii"].std()

    # Format and display the ground truth results
    print(f"SDRi: {sdr_i_avg:.3f} ± {sdr_i_std:.3f}")
    print(f"SI-SDRi: {sisdr_i_avg:.3f} ± {sisdr_i_std:.3f}")

    print(f"SDRii: {sdr_ii_avg:.3f} ± {sdr_ii_std:.3f}")
    print(f"SI-SDRii: {sisdr_ii_avg:.3f} ± {sisdr_ii_std:.3f}")

    # if True:
    #     print(f'Decision model choosing acuracy = {right_decisions/num_test_runs:.3f}')

    if CFG.expert:
        print('Expert run')
        print(f"The expert chose the model {SPA_will_lose_counts} times")
        print(f"The expert missed model wins over SPA {deep_win_missed_counter} times")
        print(f"The expert false alarmed model wins over SPA {deep_win_false_counter} times")


def dist_scores(P, pr2, J, P_method):
    deep_L22 = None
    deep_mse2 = None

    deep_L2 = np.sum((pr2[:, :J] - P[:, :J]) ** 2)
    deep_mse = mean_squared_error(pr2[:, :J], P[:, :J])

    if P_method=='both':
        deep_L22 = np.sum((pr2[:, :J] - P2[:, :J]) ** 2)
        deep_mse2 = mean_squared_error(pr2[:, :J], P2[:, :J])
        print(f'L2(vertices model, real): {deep_L2:.4f}')
        print(f'L2(probabilistic model, real): {deep_L22:.4f}')
    else:
        print(f'L2({P_method} model, real): {deep_L2:.4f}')
    SPA_L2 = np.sum((pr2[:, :J] - pe[:, :J]) ** 2)
    SPA_mse = mean_squared_error(pr2[:, :J], pe[:, :J])


    print(f'L2(SPA, real): {SPA_L2:.4f}')
    return deep_L2, deep_mse, deep_L22, deep_mse2, SPA_L2, SPA_mse






    previous_combinations = set()

    results = {}

    num_test_runs = CFG.num_test_runs

    right_decisions = 0

    data_vertices = []
    C_losses = []
    model_decision_results = []
    overlap_ratio_sum = 0
    for i in range(num_test_runs):
        data_dict, scores, decision_vector, model_decision, model_decision_scores, decision_was_right, C_loss, overlap_ratio  = \
            run_pipeline(previous_combinations, run_number=i + 1, expert=CFG.expert, J=CFG.Q,
                         P_method=CFG.P_method, calc_SPA_scores=True)
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

    print(f'overlap_ratio mean: {overlap_ratio_sum/num_test_runs}')
    comparison_table = compute_comparison_table(results, num_test_runs, [('prob_NN', 'vertices_NN'), ('prob_NN', 'SPA_NN')],
                                                method_suffixes=["ideal","prob_NN", "vertices_NN", "best_global_NN", "SPA_NN"])

    if __name__ == "__main__":
        np.random.seed(CFG.seed0)
        random.seed(CFG.seed0)
        # save_train_val_wav_signals(input_directory='dev-clean-test', output_base_directory='dev-wav-8-4', train_size=0.8,
        #                            sample_rate=16000,
        #                            max_duration=20, delay_max=5, delay_min=3, num_gaps=3)
        previous_combinations = None
        J = 2
        overlap_ratios = 0
        total_sisdr_ilrma = 0
        total_sdr_ilrma = 0
        total_pesq_ilrma = 0
        total_stoi_ilrma = 0

        total_sisdr_aux = 0
        total_sdr_aux = 0
        total_pesq_aux = 0
        total_stoi_aux = 0
        num_test_runs = 100
        for i in tqdm(range(num_test_runs)):
            signals, previous_combinations, speakers = get_speaker_signals('dev-wav-full/train', previous_combinations,
                                                                           J)
            RIRs, angles = generate_RIRs(room_length=6, room_width=6, mic_spacing=0.3, num_mics=6,
                                         min_angle_difference=30,
                                         radius=2,
                                         num_of_RIRs=J, rev=CFG.low_rev)
            combined_data = combine_speaker_signals_no_noise(signals, RIRs, num_mics=CFG.M, J=J, overlap_demand=0.6)
            Xt, Tmask, f, t, xqf, Xq, overlap_ratio, low_energy_mask, low_energy_mask_time = combined_data
            print(f'Speakers: {speakers}')
            print(f'Sources angles relative to room center: {[int(angle) for angle in angles]}')
            print(f'{J} overlap ratio: {overlap_ratio}')

            # SDR_ilrma, sisdr_ilrma, stoi_ilrma, pesq_ilrma = calc_ilrma(Xt, xqf[:,0,:])
            # SDR_aux, sisdr_aux, stoi_aux, pesq_aux = calc_auxip(Xt, xqf[:,0,:])
            # print(f'Ilrma SI-SDR: {sisdr_ilrma}')
            # print(f'AuxIVA-IP SI-SDR: {sisdr_aux}')
            #
            # total_sisdr_ilrma += sisdr_ilrma
            # total_sdr_ilrma += SDR_ilrma
            # total_stoi_ilrma += stoi_ilrma
            # total_pesq_ilrma += pesq_ilrma

            # total_sisdr_aux += sisdr_aux
            # total_sdr_aux += SDR_aux
            # total_stoi_aux += stoi_aux
            # total_pesq_aux += pesq_aux

            overlap_ratios += overlap_ratio

        print(f'Overlap ratio: {overlap_ratios / num_test_runs:.4f}')
        print(
            f'Ilrma mean SDR and SI-SDR: {total_sdr_ilrma / num_test_runs:.2f} {total_sisdr_ilrma / num_test_runs:.2f}')
        print(
            f'Ilrma mean pesq and stoi: {total_pesq_ilrma / num_test_runs:.2f} {total_stoi_ilrma / num_test_runs:.2f}')
        print(
            f'AuxIVA_IP mean SDR and SI-SDR: {total_sdr_aux / num_test_runs:.2f} {total_sisdr_aux / num_test_runs:.2f}')
        print(
            f'AuxIVA_IP mean pesq and stoi: {total_pesq_aux / num_test_runs:.2f} {total_stoi_aux / num_test_runs:.2f}')
        a = 5

def run_pipeline(previous_combinations=None, J=CFG.Q, run_number=None, expert=CFG.expert, P_method=CFG.P_method, speakers=None, combined_data=None, overlap_demand=None, rev=CFG.low_rev,
                 signals_file='dev-wav-0/train'):

    if overlap_demand ==0.5:
        overlap_demand=0.3
    if CFG.data_mode == 'real':
        rev = CFG.revs[0]
        SNR = CFG.SNRs[0]

        Xt, Tmask, f, t, xqf = process_signals(0, rev, 0, 0, 0, SNR)
    elif CFG.data_mode == 'libri':
        if combined_data==None:
            # signals, _ = get_speaker_signals('dev-clean')
            RIRs, angles = generate_RIRs(room_length=6, room_width=6, mic_spacing=0.3, num_mics=6, min_angle_difference=30,
                          radius=2,
                          num_of_RIRs=J, rev=rev)
            signals, previous_combinations, speakers = get_speaker_signals(signals_file, previous_combinations, J, speakers_list=speakers)
            combined_data = combine_speaker_signals_no_noise(signals, RIRs, num_mics=CFG.M, J=J, overlap_demand=overlap_demand)

        Xt, Tmask, f, t, xqf, Xq, overlap_ratio, low_energy_mask, low_energy_mask_time, x = combined_data
    elif CFG.data_mode == 'wsj0':
        mix, y, par = read_wsj_sample('wsj0_mix/dataset/sp_wsj/frontend4_15speakers_2mix/train', previous_combinations)
        y = y.transpose(2,1,0)
        mix = mix.T
        combined_data = extract_wsj0_features(mix, y, num_mics=CFG.M, J=2, pad=True)
        Xt, Tmask, f, t, Xq, xqf = combined_data


    Hl, Hlm, Hlf, Fall, lenF, F = feature_extraction(Xt)
    Hq = np.stack([feature_extraction(Xq[:, :, :, q])[1] for q in range(J)], axis=-1)


    Hln, W, E0, pr2, first_non0 = calculate_W_U_realSimplex(Hl, Fall, Tmask, lenF, F, J=J)
    if not run_number == None:
        print(f'Run_number: {run_number}')

    # plot_heat_mat(W)
    if CFG.data_mode=='wsj0':
        spk1 = par['spk1'].split('/')[-2]
        spk2 = par['spk2'].split('/')[-2]
        mix_index = par['index']
        print(f'Mix index: {mix_index}, Speakers: {[spk1, spk2]}')
        geo = par['rir']['arr_geometry']
        print(f'Array geometry: {geo}')
    else:
        print(f'Speakers: {speakers}')
        print(f'Sources angles relative to room center: {[int(angle) for angle in angles]}')
        print(f'{J} overlap ratio: {overlap_ratio}')
    if CFG.pad_flag:
        pe = np.zeros((CFG.N_frames,J))
        pe[CFG.pad_tfs:-CFG.pad_tfs], id0, ext0 = calculate_SPA_simplex(np.real(E0), pr2, J)
    else:
        pe, id0, ext0 = calculate_SPA_simplex(np.real(E0), pr2, J)
    Q_mat = E0[ext0, :J]
    #

    U_torch = torch.from_numpy(E0[:, :J + CFG.add_noise].real).float()
    SPA_Q_torch = torch.from_numpy(Q_mat.real).float()
    L = CFG.N_frames


    # model = MiSiCNet2(CFG.N_frames, out_dim=J, P_method=P_method).to(CFG.device)


    if P_method=='both':
        print('Running vertices model')
        deep_dict_global, P, A = global_method(W, first_non0, pr2, low_energy_mask_time, J=J, P_method='vertices', Hlf=Hlf, low_energy_mask=low_energy_mask, epochs=1)

        # plot_results(P, pr2, pe, id0, J=J)

        chosen_P, pe_will_lose = P_expert(P, ext0, pe, min_speaker_TH=0.75, max_two_speakers_TH=0.9)

        top_indices = np.argsort(A, axis=0)[-3:][::-1]
        top_vals = np.sort(A, axis=0)[-3:][::-1]
        # np.set_printoptions(precision=3, suppress=True)
        # print(f'A top values:\n{top_vals}')
        #


        print('Running probabilistic model')
        deep_dict_global2, P2, _ = global_method(W, first_non0, pr2, low_energy_mask_time, J=J, P_method='prob', Hlf=Hlf, low_energy_mask=low_energy_mask)

        # P2 = zero_P_below_TH(P2, TH=0.2)
        # plot_results(P2, pr2, pe, id0, J=J, t=t)

        decision_vector = None
        # plot3d_simplex(pr2, top_indices, title='pr2 with Amodel top vertices Simplex', azim=30,
        #                elev=30)
        # plot3d_simplex(pr2, ext0, title='pr2 with pe top vertices Simplex',
        #                azim=30,
        #                elev=30)
        # plot3d_simplex(pr2, ext0, title='Real P Simplex with SPA vertices',
        #                azim=0,
        #                elev=30, vector_type='p')
        # plot3d_simplex(E0[:, :J], ext0, title='U Simplex with SPA vertices',
        #                azim=0,
        #                elev=30)




    else:
        deep_dict_global, P, A = global_method(W, first_non0, pr2, J, P_method=P_method)


        plot_results(P, pr2, pe, id0)
        print(f'Speakers: {speakers}')
        print(f'Sources angles relative to room center: {[int(angle) for angle in angles]}')

        if P_method=='vertices':
            top_indices = np.argsort(A, axis=0)[-3:][::-1]
            top_vals = np.sort(A, axis=0)[-3:][::-1]
            # np.set_printoptions(precision=3, suppress=True)
            # print(f'A top values:\n{top_vals}')
            # plot3d_simplex(pr2, top_indices, title='pr2 with Amodel top vertices Simplex', azim=30,
            #                elev=30)
            # plot3d_simplex(pr2, ext0, title='pr2 with pe top vertices Simplex',
            #                azim=30,
            #                elev=30)

        chosen_P, pe_will_lose = P_expert(P, ext0, pe, min_speaker_TH=0.75, max_two_speakers_TH=0.9)
        if expert:
            P = chosen_P.copy()

    if not P_method=='both':
        P2=None
