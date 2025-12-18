import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    """
    生成标准的 1D 正余弦位置编码
    """
    def __init__(self, dim, max_len=64):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        
        pe = torch.zeros(1, max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self):
        # 返回 [1, max_len, dim]
        return self.pe

class DecoupledLocalSelfAttention(nn.Module):
    """
    解耦的局部注意力：
    Q = Feature + Pos
    K = Feature (纯净)
    V = Feature (纯净)
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, pos_emb=None, mask=None):
        # x: [B, N, D] (纯特征)
        
        key_padding_mask = ~mask.bool() if mask is not None else None
        x_norm = self.norm(x)
        
        # 1. 投影位置编码
        if pos_emb is not None:
            q = x_norm + pos_emb
        else:
            q = x_norm
        
        k = x_norm
        v = x_norm 
        
        # 3. Attention
        attn_out, _ = self.attn(query=q, key=k, value=v, key_padding_mask=key_padding_mask)
        
        x = x + self.dropout(attn_out)
        x = x + self.ffn(x)
        return x

class MultiHeadAttentionPool(nn.Module):
    """特征池化"""
    def __init__(self, dim_in, num_heads, dropout=0.1):
        super().__init__()
        self.dim_out = dim_in * 2
        self.pre_proj = nn.Linear(dim_in, self.dim_out)
        self.seed_query = nn.Parameter(torch.randn(1, 1, self.dim_out)) 
        nn.init.xavier_uniform_(self.seed_query)
        self.attn = nn.MultiheadAttention(self.dim_out, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(self.dim_out)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim_out, self.dim_out * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim_out * 4, self.dim_out)
        )

    def forward(self, x, mask=None):
        BG, N, D = x.shape
        x = self.pre_proj(x) 
        query = self.seed_query.expand(BG, 1, self.dim_out)
        key_padding_mask = ~mask.bool() if mask is not None else None
        
        # Pool 纯特征
        attn_out, _ = self.attn(query=query, key=x, value=x, key_padding_mask=key_padding_mask)
        attn_out = attn_out.squeeze(1) 
        attn_out = attn_out + self.ffn(self.norm(attn_out))
        return attn_out

class DecoupledCrossAttentionBlock(nn.Module):
    """
    解耦的交叉注意力
    """
    def __init__(self, dim_q, dim_kv, num_heads, dropout=0.1):
        super().__init__()
        self.pre_proj = nn.Linear(dim_q, dim_kv)

        self.attn = nn.MultiheadAttention(dim_kv, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim_kv),
            nn.Linear(dim_kv, dim_kv * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_kv * 2, dim_q)
        )
        self.norm_out = nn.LayerNorm(dim_q)

    def forward(self, local_x, global_x, local_pos_emb, global_mask=None):
  
        q_feat = self.pre_proj(local_x) 
        k_feat = global_x
        v_feat = global_x
        
        q = q_feat + local_pos_emb

        k = k_feat
        
        key_padding_mask = ~global_mask if global_mask is not None else None

        attn_out, _ = self.attn(query=q, key=k, value=v_feat, key_padding_mask=key_padding_mask)
        
        out = self.ffn(q_feat + attn_out)
        return self.norm_out(local_x + out)

class PTBlock(nn.Module):
    def __init__(self, dim, num_heads, block_size=64, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        
        self.pos_emb_gen = SinusoidalPosEmb(dim, max_len=block_size)
        
        # 1. 局部注意力
        self.local_attn = DecoupledLocalSelfAttention(dim, num_heads, dropout)

        if dim < 512: 
            self.has_children = True
            self.pool = MultiHeadAttentionPool(dim, num_heads, dropout)
            self.next_block = PTBlock(dim * 2, num_heads * 2, block_size, dropout)
            self.cross_attn = DecoupledCrossAttentionBlock(dim, dim * 2, num_heads, dropout)

            self.cross_pos_proj = nn.Linear(dim, dim * 2)
        else:
            self.has_children = False

    def forward(self, x, mask=None):
        B, N, D = x.shape

        if N % self.block_size != 0:
            pad_len = self.block_size - (N % self.block_size)
            x_padded = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask_padded = F.pad(mask, (0, pad_len), value=False)
            else:
                mask_padded = torch.ones((B, N + pad_len), dtype=torch.bool, device=x.device)
                mask_padded[:, N:] = False
            N_padded = N + pad_len
        else:
            x_padded = x
            mask_padded = mask if mask is not None else torch.ones((B, N), dtype=torch.bool, device=x.device)
            N_padded = N

        G = N_padded // self.block_size

        x_grouped = x_padded.view(B * G, self.block_size, D)

        mask_grouped = mask_padded.view(B * G, self.block_size)

        # 位置编码
        current_sin_pos = self.pos_emb_gen()
        
        # 如果是递归到了最内层的时候，这时候需要位置编码保证对应关系。
        if self.has_children:
            pos_emb_for_self = None
        else:
            pos_emb_for_self = current_sin_pos

        x_grouped = self.local_attn(x_grouped, pos_emb=pos_emb_for_self, mask=mask_grouped)

        if not self.has_children:
            x_out = x_grouped.view(B, N_padded, D)
            return x_out[:, :N, :], mask
        
        x_pooled = self.pool(x_grouped, mask_grouped) # [BG, 2D]

        x_pooled_reshaped = x_pooled.view(B, G, -1)
        mask_pooled = mask_grouped.any(dim=1).view(B, G)

        x_global, mask_global_returned = self.next_block(x_pooled_reshaped, mask_pooled)
        
        # one-to-one    
        x_global_ready = x_global.view(B * G, 1, -1)
            
        if mask_global_returned is not None:
            mask_global_ready = mask_global_returned.view(B * G, 1)
        else:
            mask_global_ready = None

        pos_emb_for_cross = self.cross_pos_proj(current_sin_pos)

        x_out = self.cross_attn(
            local_x=x_grouped, 
            global_x=x_global_ready, 
            local_pos_emb=pos_emb_for_cross, 
            global_mask=mask_global_ready
        )

        x_out = x_out.view(B, N_padded, D)
        x_out = x_out[:, :N, :]
        
        if mask is not None:
            return x_out, mask
        else:
            return x_out, None

class ParticleTransformer(nn.Module):
    def __init__(self, dim_in=24, num_heads=4, block_size=64, dropout=0.1):
        super().__init__()
        self.dim_hidden = 64
        
        self.pre_proj = nn.Linear(dim_in, self.dim_hidden)
        
        self.ptblock = PTBlock(self.dim_hidden, num_heads, block_size, dropout)
        
        self.norm = nn.LayerNorm(self.dim_hidden)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim_hidden, self.dim_hidden * 4),
            nn.GELU(),
            nn.Linear(self.dim_hidden * 4, dim_in)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):        
        residual = x
        
        x_feature = self.pre_proj(x) 
        
        x_out, _ = self.ptblock(x_feature, mask)
        
        out = self.dropout(self.ffn(self.norm(x_out)))
        
        return residual + out