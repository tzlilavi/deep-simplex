import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from utils import throwlow
import CFG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG.set_mode('unsupervised')

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, attn_dropout=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dropout_p = attn_dropout
        self.in_dim = in_dim


        multi_head_dim = (in_dim + n_heads - 1) // n_heads * n_heads  ### Add dimensions for division by n_heads
        self.multi_head_dim = multi_head_dim
        self.pad_dim = multi_head_dim - in_dim

        self.Q = nn.Linear(multi_head_dim, multi_head_dim, bias=False)
        self.K = nn.Linear(multi_head_dim, multi_head_dim, bias=False)
        self.V = nn.Linear(multi_head_dim, multi_head_dim, bias=False)
        self.Wo = nn.Linear(multi_head_dim, out_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # nn.init.normal_(self.Q.weight, std=np.sqrt(multi_head_dim // n_heads))
        # nn.init.normal_(self.K.weight, std=np.sqrt(multi_head_dim // n_heads))
        nn.init.zeros_(self.Wo.bias)

    def forward(self, queries, keys, values, mask=None):

        if self.pad_dim > 0:
            queries = F.pad(queries, (0, self.pad_dim))
            keys = F.pad(keys, (0, self.pad_dim))
            values = F.pad(values, (0, self.pad_dim))

        B, T, _ = queries.shape
        _, Ts, _ = keys.shape
        if mask is not None: # unsqueeze the mask to account for the head dim
            mask = mask.unsqueeze(dim=1)

        q = self.Q(queries).view(B, T, self.n_heads, -1).transpose(1, 2) # X @ Q
        k = self.K(keys).view(B, Ts, self.n_heads, -1).transpose(1, 2)   # X @ K
        v = self.V(values).view(B, Ts, self.n_heads, -1).transpose(1, 2) # X @ V

        attn = torch.matmul(q, k.transpose(2, 3)) # XQ @ (XK)^T

        dk = k.shape[-1]
        attn /=  np.sqrt(dk)
        if mask is not None:
            attn.masked_fill_(~mask, -1e9)
        attn = torch.softmax(attn, dim=-1)  # shape (B, nh, T, Ts)
        attn = self.attn_dropout(attn)

        z = torch.matmul(attn, v)           # shape (B, nh, T, hid)
        z = z.transpose(1, 2).reshape(B, T, -1)
        out = self.Wo(z)                    # shape (B, T, out_dims)

        return out, attn

class BiLSTM_Att(nn.Module):
    def __init__(
            self,
            dim_input=CFG.N_frames + CFG.pad_tfs,
            dim_output=CFG.Q,
            mult_heads=False,
            activation="GELU",
            hidden_size=(512, 256),
            n_repeat_last_lstm=1,
            dropout=CFG.dropout,
            eps=1e-05, P_method='prob', seed=1,
            n_heads=8, low_energy_mask=None,
            mask_input_ratio=0.2
    ):
        super(BiLSTM_Att, self).__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        bits = 2 ** np.arange(10)
        init_channels = int(bits[np.argmin(abs(dim_input - bits))])
        hidden_size = [init_channels, init_channels // 2]

        self.name = 'BiLSTM_Att'
        self.input_size = dim_input
        self.output_size = dim_output
        self.hidden_size = hidden_size
        self.mult_heads = mult_heads
        self.activation = activation
        self.dropout = dropout
        self.P_method = P_method
        self.n_heads = n_heads
        self.low_energy_mask = low_energy_mask
        self.mask_input_ratio = mask_input_ratio

        self.att_dim_input = dim_input - (dim_input % self.n_heads)
        self.Att = nn.MultiheadAttention(embed_dim=self.att_dim_input, num_heads=self.n_heads)



        self.blstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size[0], batch_first=True,
                              bidirectional=True, num_layers=1)  # type:ignore

        self.blstm2 = nn.LSTM(input_size=self.hidden_size[0] * 2, hidden_size=self.hidden_size[1], batch_first=True,
                              bidirectional=True, num_layers=n_repeat_last_lstm)  # type:ignore

        self.ConvSkip1 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels, out_channels=init_channels // 4, kernel_size=1),
            # nn.BatchNorm1d(init_channels//4, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
            # nn.GroupNorm(num_groups=32, num_channels=init_channels//4),
            # TransposeLayerNorm(init_channels//4)
        )

        self.Conv1 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels, out_channels=init_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(init_channels//2, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
            # nn.GroupNorm(num_groups=32, num_channels=init_channels//2),
            # TransposeLayerNorm(init_channels//2)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels // 2, out_channels=init_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(init_channels//4, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
            # nn.GroupNorm(num_groups=32, num_channels=init_channels//4),
            # TransposeLayerNorm(init_channels//4)
        )

        self.ConvSkip2 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels // 4, out_channels=init_channels // 16, kernel_size=1),
            nn.BatchNorm1d(init_channels//16, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
            # nn.GroupNorm(num_groups=32, num_channels=init_channels//16),
            # TransposeLayerNorm(init_channels//16)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels // 4, out_channels=init_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(init_channels//8, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
            # nn.GroupNorm(num_groups=32, num_channels=init_channels//8),
            # TransposeLayerNorm(init_channels//8)
        )

        self.Conv4 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels // 8, out_channels=init_channels // 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(init_channels//16, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
            # nn.GroupNorm(num_groups=32, num_channels=init_channels//16),
            # TransposeLayerNorm(init_channels//16)
        )

        if dropout is not None:
            self.dropout1 = nn.Dropout(p=self.dropout)
            self.dropout2 = nn.Dropout(p=self.dropout)
            self.dropout3 = nn.Dropout(p=self.dropout)
            self.dropout4 = nn.Dropout(p=self.dropout)



        self.linear = nn.Linear(init_channels//16, self.output_size)

        if self.activation is not None and len(self.activation) > 0:  # type:ignore
            self.activation_func = getattr(nn, self.activation)()  # type:ignore
        else:
            self.activation_func = None
        # self.apply(self.init_weights)
        # for layer in self.children():
        #     self.init_weights(layer)
        self.init_weights()

        self.decoder = nn.Linear(in_features=dim_output, out_features=CFG.N_frames, bias=False)



    def init_weights(self):

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
        x = W[:, :, :self.att_dim_input]
        res_feats = W[:, :, self.att_dim_input:]
        x, _ = self.Att(x, x, x)
        x = torch.concatenate([x, res_feats], dim=-1)
        x = x.reshape(B * L, L)


        x, _ = self.blstm1(x)
        if self.dropout:
            x = self.dropout1(x)


        x, _ = self.blstm2(x)
        if self.dropout:
            x = self.dropout2(x)

        y = torch.unsqueeze(x.T, 0)
        skip1 = self.ConvSkip1(y)
        y = self.Conv1(y)
        y = F.leaky_relu(y, negative_slope=0.1, inplace=True)
        y = self.Conv2(y)
        y = F.leaky_relu(y + skip1, negative_slope=0.1, inplace=True)
        if self.dropout:
            y = self.dropout3(y)
        skip2 = self.ConvSkip2(y)
        y = self.Conv3(y)
        y = F.leaky_relu(y, negative_slope=0.1, inplace=True)
        y = self.Conv4(y)
        y = F.leaky_relu(y + skip2, negative_slope=0.1, inplace=True)
        if self.dropout:
            y = self.dropout4(y)

        if self.mult_heads:
            P = {}

            for i in range(3, 6):
                linear_layer = getattr(self, f"linear{i}")
                y_linear = self.activation_func(linear_layer(torch.transpose(y, 1, 2)))
                P[i] = F.softmax(y_linear, dim=-1)

            W = self.decoder(P[3])
        else:
            y = self.activation_func(self.linear(torch.transpose(y, 1, 2)))
            y = y.reshape(B, L, self.output_size)


            A = F.softmax(y.squeeze(0), dim=0)

            As = A.detach().cpu().numpy()
            top_vals = np.sort(As, axis=0)[-3:][::-1]
            if self.P_method=='prob':
                # logits = y[:, :, :-1]
                # P_speakers = F.softmax(logits, dim=-1)
                # P_noise = torch.sigmoid(y[:, :, -1:])  # [B, T, 1]
                # P = torch.cat([P_speakers * (1 - P_noise), P_noise], dim=-1)
                P = F.softmax(y, dim=-1)  ## [b, L , J+1]
                E = F.softmax(self.decoder.weight.unsqueeze(0), dim=-1)
            elif self.P_method=='vertices':
                E = F.softmax(y, dim=-1)  ## [b, L , J+1]


                P = torch.matmul(W.squeeze(0), A)
                P = P * (P > 0)
                P[P.sum(1) > 1, :] = P[P.sum(1) > 1, :] / P[P.sum(1) > 1, :].sum(1, keepdims=True)

                P = P.unsqueeze(0)


        if self.low_energy_mask is not None:
            P = P.clone()
            P[:, self.low_energy_mask, :] = 0


        W = torch.transpose(torch.bmm(P, torch.transpose(P, 1, 2)), 1,2)
        W[:, range(CFG.N_frames), range(CFG.N_frames)] = 1


        return P, W, E, A

class TransposeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(TransposeLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        # x: (B, C, L)
        x = x.permute(0, 2, 1)     # (B, L, C)
        x = self.ln(x)
        x = x.permute(0, 2, 1)     # (B, C, L)
        return x