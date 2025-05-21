import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from utils import throwlow
import CFG
from einops import rearrange, repeat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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
            nn.BatchNorm1d(init_channels//4, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
        )

        self.Conv1 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels, out_channels=init_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(init_channels//2, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels // 2, out_channels=init_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(init_channels//4, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
        )

        self.ConvSkip2 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels // 4, out_channels=init_channels // 16, kernel_size=1),
            nn.BatchNorm1d(init_channels//16, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels // 4, out_channels=init_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(init_channels//8, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
        )

        self.Conv4 = nn.Sequential(
            nn.Conv1d(in_channels=init_channels // 8, out_channels=init_channels // 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(init_channels//16, eps=eps, momentum=0.1, affine=True, track_running_stats=True),
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


        self.U_calculator = nn.Linear(in_features=out_dim, out_features=out_dim, bias=False)




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


        W = self.decoder(P)
        W[:, range(CFG.N_frames), range(CFG.N_frames)] = 1

        E = self.decoder.weight.unsqueeze(0)

        return P, W, E, A



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"Dim should be divisible by heads dim={dim}, heads={num_heads}"
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # B1C -> B1H(C/H) -> BH1(C/H)
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=3., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.15, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x[:, 1:, :])))
        x = x[:, 0:1, ...] + self.drop_path(self.attn(x))  # Better result
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttentionBlock(dim, num_heads=heads, drop=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = torch.cat((attn(x), self.norm(x[:, 1:, :])), dim=1)
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, patch_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)


        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size. '

        num_patches = (image_height // patch_height)
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('c (h p1) -> h (p1 c)', p1=patch_size),
        #     # nn.Linear(patch_dim, dim),
        # )
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) -> b h (p1 c)', p1=patch_height),
            nn.Linear(patch_dim * patch_width, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, img):
        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # cls_tokens = repeat(self.cls_token, '() d -> b d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x = torch.cat((cls_tokens, x), dim=0)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return x

class TransposeNorm(nn.Module):
    def __init__(self, size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.size = size
        self.batchnorm = nn.BatchNorm1d(size, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        self.layer_norm = nn.LayerNorm(size, eps=eps)

    def forward(self, x):
        # Transpose the input tensor so that the columns correspond to features
        x = x.transpose(1, 2)
        # x = self.batchnorm(x)
        x = self.layer_norm(x)
        # Transpose the input tensor back
        x = x.transpose(1, 2)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, J, L, size, patch, dim, eps = 1e-5, momentum = 0.9, Q_weights=None, pool='cls', find_vertices=False):
        super(AutoEncoder, self).__init__()
        self.name = 'Encoder'
        J = J + CFG.add_noise  ## Add noise dim
        self.J, self.L, self.patch, self.dim = J, L, patch, dim
        self.size = (size // patch) * patch
        self.find_vertices = find_vertices
        self.encoder = nn.Sequential(
            nn.Conv1d(self.size, 128, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm1d(128, eps=eps, momentum=momentum, affine=True, track_running_stats=True), #TransposeNorm(128, momentum=0.9)
            TransposeNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(CFG.dropout),
            nn.Conv1d(128, 64, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm1d(64, eps=eps, momentum=momentum, affine=True, track_running_stats=True),
            TransposeNorm(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, (dim * J) // patch ** 2, kernel_size=1, stride=1, padding=0),
            TransposeNorm((dim * J) // patch ** 2)
            # nn.BatchNorm1d((dim * J) // patch ** 2, eps=eps, momentum=momentum, affine=True, track_running_stats=True) #nn.BatchNorm1d((dim * J) // patch ** 2, momentum=0.5), #TransposeNorm((dim * J) // patch ** 2, momentum=0.5)
        )

        self.vtrans = ViT(image_size=self.size, patch_size=patch, dim=dim * J,
                          patch_dim=(dim * J) // patch ** 2, depth=2,
                                      heads=8, mlp_dim=12, pool=pool)

        self.upscale = nn.Linear(dim, self.L)

        self.smooth = nn.Sequential(
            nn.Conv1d(J, J, kernel_size=3, stride=1, padding=1))
        self.softmax = nn.Softmax(dim=1)

        self.decoder = nn.Linear(in_features=J, out_features=L, bias=False)
        self.apply(self.weights_init)
        self.U_calculator = nn.Linear(in_features=J, out_features=J, bias=False)

        if Q_weights is None:
            self.U_calculator = nn.Linear(in_features=J, out_features=J, bias=False)
        else:
            self.U_calculator = nn.Linear(in_features=J, out_features=J, bias=False)
            with torch.no_grad():
                self.U_calculator.weight.copy_(Q_weights.T)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv1d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, epoch=None):
        W = x.clone()
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x[:, :self.size, :self.size]
        b, _, _ = x.shape
        abu_est = self.encoder(x)
        cls_emb = self.vtrans(abu_est)
        cls_emb = cls_emb.view(b, self.J, -1)
        abu_est = self.upscale(cls_emb).view(b, self.J, self.L)
        P_output = self.smooth(abu_est)
        P_output = torch.transpose(P_output, dim0=1, dim1=2)
        if not self.find_vertices:
            P_output = F.softmax(P_output, dim=-1)  ## [b, L , J+1]
        else:
            A = F.softmax(P_output.squeeze(0), dim=0)

            P_output = torch.matmul(W.squeeze(0), A)
            P_output = P_output * (P_output > 0)
            P_output[P_output.sum(1) > 1, :] = P_output[P_output.sum(1) > 1, :] / P_output[P_output.sum(1) > 1, :].sum(1, keepdims=True)

            P_output = P_output.unsqueeze(0)

        # P_output = F.sigmoid(P_output)
        # P_output = P_output / P_output.sum(dim=2, keepdim=True)

        W_output = self.decoder(P_output)
        E = self.decoder.weight.unsqueeze(0)

        U = self.U_calculator(P_output)


        return P_output, W_output, E, U