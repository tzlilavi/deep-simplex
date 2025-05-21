import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.pe = self.create_positional_encoding(max_seq_len, d_model)

    def create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        x = x + self.pe[:, :x.size(-1)].detach()
        return x

class Transformer(nn.Module):
    def __init__(self, L, out_dim, n_heads, n_layers, d_ff=2048, dropout=0.3):
        super(Transformer, self).__init__()

        self.name = 'Transformer_Encoder'
        self.pe = PositionalEncoder(L, max_seq_len=L)
        self.encoder_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=L, nhead=n_heads, dim_feedforward=100,
                                        dropout=dropout) for _ in range(n_layers)])

        self.fc1 = nn.Sequential(nn.Linear(in_features=L, out_features=out_dim + 1, ),
                                 nn.GELU(),
                                 nn.LayerNorm(out_dim + 1))
        self.fc_try = nn.Sequential(nn.Linear(in_features=L, out_features=out_dim + 1, ),
                                 nn.GELU(),
                                 nn.LayerNorm(out_dim + 1), nn.Sigmoid())

        self.in_dim = L
        self.out_dim = out_dim

    def forward(self, x):
        # x = x + self.pe(x)
        x = x + torch.zeros(1, self.in_dim, self.in_dim)
        skip_connection = x

        for layer in self.encoder_layers:
            x = layer(x)
            x = skip_connection + x


        # x = sum(skip_connections) + x

        # x = self.fc1(x)
        # x = F.softmax(x, dim=-1)
        # x = x[:, :, :self.out_dim]
        # x, _ = torch.topk(x, k=self.out_dim + 1, dim=-1)
        x = self.fc1(x)
        return x