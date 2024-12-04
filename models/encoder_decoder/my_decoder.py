import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg  
from einops import rearrange
import math

# 位置嵌入矩阵
def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out


class PlainDecoder(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_dim = 512, 
        depth = 3,
        num_heads = 8,
        dropout = 0.1, 
        ff_dropout = 0.1, 
        qkv_bias = True,
        act_layer = nn.ReLU(),
        qk_scale = None,
        ):
        super(PlainDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.depth = depth

        self.layers = nn.ModuleList([])
        for i in range(depth):
            sublayer = DecoderLayer( 
                embed_dim = embed_dim, 
                num_heads = num_heads, 
                dropout = dropout, 
                ff_dropout = ff_dropout,
                qkv_bias=qkv_bias,
                act_layer = act_layer,
                qk_scale = qk_scale,
            )           
            self.layers.append(sublayer)
        
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED)

        self.word_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        
        self.embed_scale = math.sqrt(self.embed_dim)
        
        self.pos_embed = nn.Embedding.from_pretrained(sinusoid_encoding_table(100, self.embed_dim, 0), freeze=True)

        self.generator = nn.Linear(self.embed_dim, self.vocab_size, bias=True)
        
        self.clear_buffer()

    def init_buffer(self, batch_size):
        self.seq_len = 0
        for layer in self.layers:
            layer.init_buffer(batch_size)

    def clear_buffer(self):
        self.seq_len = None
        for layer in self.layers:
            layer.clear_buffer()

    def apply_to_states(self, fn):
        for layer in self.layers:
            layer.apply_to_states(fn)

    def forward(self, seq, encoder_out, seq_mask=None, att_mask=None):
        
        seq_len = seq.shape[1]
        
        if self.seq_len is not None:
            seq_len = self.seq_len + seq_len
            self.seq_len = seq_len
            pos_indx = torch.arange(seq_len, seq_len + 1, device='cuda').view(1, -1)
        else:
            pos_indx = torch.arange(1, seq_len + 1, device='cuda').view(1, -1)
            
        # 词汇嵌入 + 位置嵌入
        pos_embed = self.pos_embed(pos_indx) 
        word_embed = self.word_embed(seq) #* self.embed_scale
        
        x =  word_embed + pos_embed
        x = self.dropout(x)

        for i_layer, layer in enumerate(self.layers):
            x = layer(i_layer, x, encoder_out, seq_mask, att_mask)

        x = self.dropout(x) # [b, seq_len, d]
        x = self.generator(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, ff_dropout, qkv_bias, act_layer, qk_scale):
        super(DecoderLayer, self).__init__()
        
        self.word_attn = MHA(
            embed_dim = embed_dim, 
            num_heads = num_heads,
            qkv_bias = qkv_bias,
            dropout = dropout,
            qk_scale  = qk_scale
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.cross_att = MHA(
            embed_dim = embed_dim, 
            num_heads = num_heads,
            qkv_bias = qkv_bias,
            dropout = dropout,
            qk_scale = qk_scale
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ff_layer = FFN(
            embed_dim = embed_dim, 
            ffn_embed_dim = embed_dim * 4, 
            relu_dropout = ff_dropout,
            act_layer = act_layer
        )
        self.layer_norm3 = torch.nn.LayerNorm(embed_dim)

    def apply_to_states(self, fn):
        self.word_attn.apply_to_states(fn)

    def init_buffer(self, batch_size):
        self.word_attn.init_buffer(batch_size)

    def clear_buffer(self):
        self.word_attn.clear_buffer()

    def forward(self, i_layer, x, encoder_out, seq_mask, att_mask):
    
        short_cut = x
        x = self.word_attn(
            q = x, 
            k = x,
            v = x,
            mask = seq_mask,
        )
        x = self.layer_norm1(x + short_cut)

        # cross
        short_cut = x
        x = self.cross_att(
            x, 
            encoder_out,
            encoder_out,
            mask = att_mask,
        )
        x = self.layer_norm2(x + short_cut)

        # ffn
        short_cut = x
        x = self.ff_layer(x)
        x = self.layer_norm3(x + short_cut)        
    
        return x


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias, dropout, qk_scale=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.o_linear = nn.Linear(embed_dim, embed_dim)
        
        self.drop = nn.Dropout(dropout)

        self.clear_buffer()

    def init_buffer(self, batch_size):
        self.buffer_key = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device='cuda')
        self.buffer_value = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device='cuda')
        
    def clear_buffer(self):
        self.buffer_key = None
        self.buffer_value = None
        
    def apply_to_states(self, fn):
        self.buffer_key = fn(self.buffer_key)
        self.buffer_value = fn(self.buffer_value)
    
    def forward(self, q, k, v, mask):

        B, N, C = q.size()
        q = self.q_linear(q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if self.buffer_key is not None and self.buffer_value is not None:
            self.buffer_key = torch.cat([self.buffer_key, k], dim=2)
            self.buffer_value = torch.cat([self.buffer_value, v], dim=2)
            k = self.buffer_key
            v = self.buffer_value

        # [B, H, L, L] or [B, H, L, M]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -torch.inf)
            
        attn = F.softmax(attn, dim=-1) # [b, h, n_q, n_k]
        attn = self.drop(attn)

        # [b, h, n_q, d/h]
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)     
        out = self.o_linear(out)
        out = self.drop(out)
        return out


class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout, act_layer):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.act = act_layer
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = nn.Dropout(relu_dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
