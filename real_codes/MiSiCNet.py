import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import CFG
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MiSiCNet2(nn.Module):
    def __init__(self, L, out_dim, P_method=CFG.P_method, dropout=CFG.dropout, eps = 1e-05, momentum=CFG.momentum): ## Input shape [b, L,L]
        super(MiSiCNet2, self).__init__()
        self.name = 'MiSiCNet2'
        self.P_method = P_method
        bits = 2 ** np.arange(10)
        init_channels = int(bits[np.argmin(abs(L - bits))])
        self.Conv1 = nn.Sequential(nn.Conv1d(in_channels=L, out_channels=init_channels//2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(init_channels//2, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Dropout(dropout)) ## [b, 256,L]

        self.Conv2 = nn.Sequential(nn.Conv1d(in_channels=init_channels//2, out_channels=init_channels//4, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(init_channels//4, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                   # nn.LeakyReLU(0.1, inplace=True),
                                   nn.Dropout(dropout)) ## [b, 128,L]

        self.ConvSkip1 = nn.Sequential(nn.Conv1d(in_channels=L, out_channels=init_channels//4, kernel_size=3, padding=1),
                                       nn.BatchNorm1d(init_channels//4, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                       # nn.LeakyReLU(0.1, inplace=True),
                                       nn.Dropout(dropout))
        self.Conv3 = nn.Sequential(nn.Conv1d(in_channels=init_channels//4, out_channels=init_channels//8, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(init_channels//8, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Dropout(dropout))## [b, 64,L]
        self.Conv4 = nn.Sequential(nn.Conv1d(in_channels=init_channels//8, out_channels=init_channels//16, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(init_channels//16, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                   # nn.LeakyReLU(0.1, inplace=True),
                                   nn.Dropout(dropout)) ## [b, 32,L]
        self.ConvSkip2 = nn.Sequential(nn.Conv1d(in_channels=init_channels//4, out_channels=init_channels//16, kernel_size=3, padding=1),
                                       nn.BatchNorm1d(init_channels//16, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                       # nn.LeakyReLU(0.1, inplace=True),
                                       nn.Dropout(dropout))
        self.fc1 = nn.Sequential(nn.Linear(in_features=init_channels//16, out_features=out_dim + CFG.add_noise),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.LayerNorm(out_dim + CFG.add_noise)) ## [L,J + 1]
        self.decoder = nn.Linear(in_features=out_dim + CFG.add_noise, out_features=L, bias=False)
        self.leakyReLU = nn.LeakyReLU(0.1, inplace=True)

        if Q_weights is None:
            self.U_calculator = nn.Linear(in_features=out_dim, out_features=out_dim, bias=False)
        else:
            self.U_calculator = nn.Linear(in_features=out_dim, out_features=out_dim, bias=False)
            with torch.no_grad():
                self.U_calculator.weight.copy_(Q_weights.T)
        # self.Q = nn.Parameter(Q_weights.clone())
        # self.Q.requires_grad = False


        self.out_dim = out_dim
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight, mean=1.0, std=0.02)
            init.constant_(m.bias, 0)

    def forward(self, x, epoch=None): ## [b, L , L]
        # x = x.unsqueeze(0)
        y=x.clone()
        skip_connect1 = self.ConvSkip1(x)
        x = self.Conv1(x) ## [b, 256 , L]
        x = self.Conv2(x) ## [b, 128 , L]
        x = self.leakyReLU(x + skip_connect1)
        skip_connect2 = self.ConvSkip2(x)
        x = self.Conv3(x) ## [b, 64 , L]
        x = self.Conv4(x) ## [b, 32 , L]
        x= self.leakyReLU(x + skip_connect2)
        x = torch.transpose(x, 1, 2) ## [b, 32 , L]
        x = self.fc1(x) ## [b, L , J+noisedim]
        no_pad_x = x[:, CFG.pad_tfs:-CFG.pad_tfs, :]

        if self.P_method=='prob':
            P = F.softmax(x, dim=-1)  ## [b, L , J+1]
            A = torch.zeros((CFG.N_frames, self.output_size)).to(CFG.device)
            A[CFG.pad_tfs:-CFG.pad_tfs, :] = F.softmax(no_pad_x.squeeze(0), dim=0)

        elif self.P_method=='vertices':
            A = torch.zeros((CFG.N_frames, self.output_size)).to(CFG.device)
            A[CFG.pad_tfs:-CFG.pad_tfs, :] = F.softmax(no_pad_x.squeeze(0), dim=0)
            P = torch.matmul(y.squeeze(0), A)
            P = P * (P > 0)
            P[P.sum(1) > 1, :] = P[P.sum(1) > 1, :] / P[P.sum(1) > 1, :].sum(1, keepdims=True)
            P = P.unsqueeze(0)


        # P = F.sigmoid(P)
        # P = P / P.sum(dim=2, keepdim=True)
        W = self.decoder(P)
        W[:, range(CFG.N_frames), range(CFG.N_frames)] = 1

        E = self.decoder.weight.unsqueeze(0)

        return P, W, E, A


class MiSiCNet_QinvU(nn.Module):
    def __init__(self, U_matrix, L, out_dim, dropout=CFG.dropout, eps = 1e-05, momentum=CFG.momentum): ## Input shape [b, L,L]
        super(MiSiCNet_QinvU, self).__init__()
        self.name = 'MiSiCNet_QinvU'
        self.U_matrix = U_matrix
        ## input size - ## [b, 1, L,L]
        self.Conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0),
                                   nn.MaxPool2d(kernel_size=2),
                                   nn.BatchNorm2d(16, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Dropout(dropout)) ## [b, 16, (L-2)/2,(L-2)/2]

        self.Conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0),
                                   nn.MaxPool2d(kernel_size=2),
                                   nn.BatchNorm2d(32, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Dropout(dropout)) ## [b, 32, (((L-2)/2)-4)/2,(((L-2)/2)-4)/2]

        self.ConvSkip1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=13, padding=0),
                                       nn.MaxPool2d(kernel_size=2, stride=4),
                                       nn.BatchNorm2d(32, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Dropout(0))  ## [b, 32, (L-2)/2,(L-2)/2]

        self.Conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
                                   nn.MaxPool2d(kernel_size=2),
                                   nn.BatchNorm2d(64, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Dropout(dropout)) ## [b, 64, (((((L-2)/2)-4)/2)-2)/2,(((((L-2)/2)-4)/2)-2)/2]
        self.Conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=0),
                                   nn.MaxPool2d(kernel_size=2),
                                   nn.BatchNorm2d(128, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Dropout(dropout)) ## (((((((L-2)/2)-4)/2)-2)/2)-4)/2,(((((((L-2)/2)-4)/2)-2)/2)-4)/2]
        self.ConvSkip2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=13, padding=0),
                                       nn.MaxPool2d(kernel_size=2, stride=4),
                                       nn.BatchNorm2d(128, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Dropout(dropout))
        self.fc1 = nn.Sequential(nn.Linear(in_features=int((((((((L-2)/2)-4)/2)-2)/2)-4)/2)**2 *128, out_features=(out_dim + CFG.add_noise) ** 2),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.LayerNorm((out_dim + CFG.add_noise) ** 2)) ## [L,J + 1]
        self.decoder = nn.Linear(in_features=out_dim + CFG.add_noise, out_features=L, bias=False)
        self.out_dim = out_dim
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight, mean=1.0, std=0.02)
            init.constant_(m.bias, 0)

    def forward(self, x): ## [b, L , L]
        #
        y=x.clone()
        x = x.unsqueeze(0)
        skip_connect1 = self.ConvSkip1(x)
        x = self.Conv1(x) ##
        x = self.Conv2(x) ##
        x = x + skip_connect1
        skip_connect2 = self.ConvSkip2(x)
        x = self.Conv3(x) ##
        x = self.Conv4(x) ##
        x= x + skip_connect2
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x) ##
        Q_inverse = x.reshape(1, self.out_dim + CFG.add_noise, CFG.Q + CFG.add_noise)

        P = torch.matmul(self.U_matrix, Q_inverse)

        # P = F.softmax(P, dim=-1)
        #
        # P = P[:, :, :self.out_dim + CFG.noise] ## [b, L , J+1]

        W = self.decoder(P)
        W[:, range(CFG.N_frames), range(CFG.N_frames)] = 1

        E = F.softmax(self.decoder.weight.unsqueeze(0), dim=-1)

        return P, W, E

