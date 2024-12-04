import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from lib.config import cfg  


class PlainEncoder(nn.Module):
    def __init__(
        self, 
        num_layers, 
        dim=512, 
        num_heads=8,
        attn_drop=0.1,
        proj_drop=0.1,
        drop_path=0, 
        qkv_bias = True,
        act_layer=nn.GELU,
        qk_scale = None,
        ):
        
        super(PlainEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):  
            layer = EncoderLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=4.,
                # mlp_ratio=2.,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=drop_path,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                act_layer=act_layer,
            )
            self.layers.append(layer)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# self + ffn
class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_path=0.,
        attn_drop=0.,
        proj_drop=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm, 
        qkv_fuse=True, # True -> self attention
        ):
        super().__init__()

        self.attn = MHA(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_fuse=qkv_fuse,
            )
        
        self.mlp = FFN(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio), 
            act_layer=act_layer, 
            drop=attn_drop
            )

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask=mask))
        x = self.norm2(x + self.mlp(x))
        return x            

class MHA(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 out_dim=None,
                 qkv_bias=False, # linear bias
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False, # true -- self
                 ):
        super().__init__()
        
        if out_dim is None:
            out_dim = dim
            
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv_fuse = qkv_fuse
        
        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key=None, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N

            # [3, B, H, N, C//H]s
            qkv = rearrange(self.qkv(x), 'b n (m h c) -> m b h n c', b=B, n=N, m=3, h=self.num_heads, c=self.head_dim)
            # [B, H, N, C//H]
            q, k, v = qkv[0], qkv[1], qkv[2] 

        else:
            assert key is not None
            assert value is not None            
            B, N, C = query.shape

            S = key.size(1)
            # [B, H, N, C//H]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=self.head_dim)
            # [B, H, S, C//H]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=self.head_dim)
            # [B, H, S, C//H]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=self.head_dim)

        # [B, H, N, S]  
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1) # [b, 1, n, n] or [b, 1, 1, n]
            attn = attn.masked_fill(mask == 0, -torch.inf)  

        attn = F.softmax(attn, dim=-1) 
        attn = self.attn_drop(attn)

        assert attn.shape == (B, self.num_heads, N, S)

        # [B, H, N, C//H] -> [B, N, C]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
