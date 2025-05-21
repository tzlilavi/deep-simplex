import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from timm.models.vision_transformer import Mlp
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import CFG

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