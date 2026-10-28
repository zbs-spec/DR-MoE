import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class TopKAttention(nn.Module):
    def __init__(self, dim, heads, top_k, dropout):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.top_k = top_k
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-1, -2) * ((d // self.heads) ** -0.5)
        count = min(self.top_k, n)
        indices = scores.topk(count, dim=-1).indices
        masked = torch.full_like(scores, torch.finfo(scores.dtype).min)
        masked.scatter_(-1, indices, scores.gather(-1, indices))
        weights = self.dropout(torch.softmax(masked, dim=-1))
        values = (weights @ v).transpose(1, 2).contiguous().reshape(b, n, d)
        return self.proj(values)


class GlobalExpert(nn.Module):
    def __init__(self, dim, heads, dropout, state, conv, expand):
        super().__init__()
        if Mamba is None:
            raise ImportError('mamba-ssm is required. Install the dependencies in requirements-mamba.txt.')
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=state, d_conv=conv, expand=expand)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(a))
        return self.norm2(x + self.drop(self.mamba(x)))


class LocalExpert(nn.Module):
    def __init__(self, dim, heads, top_k, dropout):
        super().__init__()
        self.attn = TopKAttention(dim, heads, top_k, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        return self.norm2(x + self.drop(self.ffn(x)))


class DRMoE(nn.Module):
    def __init__(self, input_length=180, horizon=1, dim=256, heads=8, global_patch=48, global_stride=16, local_patch=16, local_stride=8, top_k=8, dropout=0.1, d_state=16, d_conv=4, expand=2, ablation='none'):
        super().__init__()
        assert input_length >= max(global_patch, local_patch)
        assert dim % heads == 0
        assert ablation in {'none', 'global', 'local', 'router'}
        self.input_length = input_length
        self.horizon = horizon
        self.global_patch = global_patch
        self.global_stride = global_stride
        self.local_patch = local_patch
        self.local_stride = local_stride
        self.ablation = ablation
        self.global_tokens = (input_length - global_patch) // global_stride + 1
        self.local_tokens = (input_length - local_patch) // local_stride + 1
        self.global_embed = nn.Linear(global_patch, dim)
        self.local_embed = nn.Linear(local_patch, dim)
        self.global_position = nn.Parameter(torch.zeros(1, self.global_tokens, dim))
        self.local_position = nn.Parameter(torch.zeros(1, self.local_tokens, dim))
        self.global_expert = GlobalExpert(dim, heads, dropout, d_state, d_conv, expand) if ablation != 'global' else None
        self.local_expert = LocalExpert(dim, heads, top_k, dropout) if ablation != 'local' else None
        self.router = nn.Sequential(nn.Linear(input_length, dim), nn.GELU(), nn.Linear(dim, 2))
        self.head = nn.Linear((self.global_tokens + self.local_tokens) * dim, horizon)
        nn.init.trunc_normal_(self.global_position, std=0.02)
        nn.init.trunc_normal_(self.local_position, std=0.02)

    def forward(self, x, return_experts=False):
        if x.ndim == 3:
            x = x.squeeze(-1)
        if x.ndim != 2 or x.shape[-1] != self.input_length:
            raise ValueError(f'Expected input shape [batch, {self.input_length}]')
        b = x.shape[0]
        if self.global_expert is None:
            g = x.new_zeros(b, self.global_tokens, self.global_embed.out_features)
        else:
            p = x.unfold(1, self.global_patch, self.global_stride)
            g = self.global_expert(self.global_embed(p) + self.global_position)
        if self.local_expert is None:
            l = x.new_zeros(b, self.local_tokens, self.local_embed.out_features)
        else:
            p = x.unfold(1, self.local_patch, self.local_stride)
            l = self.local_expert(self.local_embed(p) + self.local_position)
        if self.ablation == 'router':
            weights = x.new_full((b, 2), 0.5)
        else:
            weights = self.router(x).softmax(-1)
        features = torch.cat([g.flatten(1) * weights[:, :1], l.flatten(1) * weights[:, 1:2]], dim=1)
        output = self.head(features)
        if return_experts:
            return output, {'global': g, 'local': l, 'weights': weights}
        return output
