import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb3D(nn.Module):
    """生成 3D 位置的傅里叶特征 (Fourier Features)"""
    def __init__(self, dim=60, mult=10):
        super().__init__()
        self.dim = dim
        self.freq_bands = 2. ** torch.linspace(0., mult, dim // 6)

    def forward(self, x):
        # x: [B, N, 3]
        freq = self.freq_bands.to(x.device).reshape(1, 1, 1, -1)
        x_expanded = x.unsqueeze(-1)
        final_x = x_expanded * freq
        feat = torch.cat([torch.sin(final_x), torch.cos(final_x)], dim=-1)
        feat = feat.reshape(x.shape[0], x.shape[1], -1) # [B, N, 60]
        return feat

class DecoupledLocalSelfAttention(nn.Module):
    """
    解耦的局部注意力：
    Q = Feature + Pos
    K = Feature + Pos
    V = Feature (纯净)
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # 专门用于将 60 维 Fourier 位置特征投影到当前维度
        self.pos_proj = nn.Linear(60, dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, pos_emb, mask=None):
        # x: [B, N, D] (纯特征)
        # pos_emb: [B, N, 60] (位置编码)
        
        key_padding_mask = ~mask.bool() if mask is not None else None
        x_norm = self.norm(x)
        
        # 1. 投影位置编码
        p = self.pos_proj(pos_emb) # [B, N, D]
        
        # 2. 构建 Q, K (带位置) 和 V (不带位置)
        q = x_norm + p
        k = x_norm + p
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
    解耦的交叉注意力 (支持 Adaptive Mode)
    """
    def __init__(self, dim_q, dim_kv, num_heads, dropout=0.1):
        super().__init__()
        self.pre_proj = nn.Linear(dim_q, dim_kv)
        self.pos_proj = nn.Linear(60, dim_kv) # 位置投影

        self.attn = nn.MultiheadAttention(dim_kv, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim_kv),
            nn.Linear(dim_kv, dim_kv * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_kv * 2, dim_q)
        )
        self.norm_out = nn.LayerNorm(dim_q)

    def forward(self, local_x, global_x, local_pos_emb, global_pos_emb, global_mask=None):
        """
        local_x: [BG, K, D] (Query) 
        global_x: [BG, G_view, 2D] (Key/Value) 
                  G_view 可能是 G (All-to-All) 也可能是 1 (One-to-One)
        global_mask: [BG, G_view] (Bool)
        """
        
        q_feat = self.pre_proj(local_x) 
        k_feat = global_x
        v_feat = global_x # Value 纯特征
        
        # 注入位置到 Q 和 K
        q_pos = self.pos_proj(local_pos_emb)
        k_pos = self.pos_proj(global_pos_emb)
        
        q = q_feat + q_pos
        k = k_feat + k_pos
        
        key_padding_mask = ~global_mask if global_mask is not None else None

        attn_out, _ = self.attn(query=q, key=k, value=v_feat, key_padding_mask=key_padding_mask)
        
        out = self.ffn(q_feat + attn_out)
        return self.norm_out(local_x + out)

class PTBlock(nn.Module):
    def __init__(self, dim, num_heads, block_size=64, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        
        # 本层使用的位置编码生成器
        self.pos_emb_gen = SinusoidalPosEmb3D(dim=60)
        
        # 1. 局部注意力
        self.local_attn = DecoupledLocalSelfAttention(dim, num_heads, dropout)

        if dim < 512: 
            self.has_children = True
            self.pool = MultiHeadAttentionPool(dim, num_heads, dropout)
            self.next_block = PTBlock(dim * 2, num_heads * 2, block_size, dropout)
            self.cross_attn = DecoupledCrossAttentionBlock(dim, dim * 2, num_heads, dropout)
        else:
            self.has_children = False

    def forward(self, x, pos, mask=None):
        B, N, D = x.shape
        
        current_pos_emb = self.pos_emb_gen(pos) # [B, N, 60]

        if N <= self.block_size or not self.has_children:
            if mask is None:
                mask = torch.ones((B, N), dtype=torch.bool, device=x.device)
            x_out = self.local_attn(x, current_pos_emb, mask)
            return x_out, mask

        if N % self.block_size != 0:
            pad_len = self.block_size - (N % self.block_size)
            x_padded = F.pad(x, (0, 0, 0, pad_len))
            pos_padded = F.pad(pos, (0, 0, 0, pad_len)) 
            pos_emb_padded = F.pad(current_pos_emb, (0, 0, 0, pad_len))
            if mask is not None:
                mask_padded = F.pad(mask, (0, pad_len), value=False)
            else:
                mask_padded = torch.ones((B, N + pad_len), dtype=torch.bool, device=x.device)
                mask_padded[:, N:] = False
            N_padded = N + pad_len
        else:
            x_padded = x
            pos_padded = pos
            pos_emb_padded = current_pos_emb
            mask_padded = mask if mask is not None else torch.ones((B, N), dtype=torch.bool, device=x.device)
            N_padded = N

        G = N_padded // self.block_size

        x_grouped = x_padded.view(B * G, self.block_size, D)
        pos_grouped = pos_padded.view(B * G, self.block_size, 3)
        pos_emb_grouped = pos_emb_padded.view(B * G, self.block_size, 60)
        mask_grouped = mask_padded.view(B * G, self.block_size)

        x_grouped = self.local_attn(x_grouped, pos_emb_grouped, mask_grouped)

        x_pooled = self.pool(x_grouped, mask_grouped) # [BG, 2D]

        mask_float = mask_grouped.unsqueeze(-1).float()
        pos_sum = (pos_grouped * mask_float).sum(dim=1)
        valid_counts = mask_float.sum(dim=1).clamp(min=1.0)
        pos_pooled = pos_sum / valid_counts # [BG, 3]

        x_pooled_reshaped = x_pooled.view(B, G, -1)
        pos_pooled_reshaped = pos_pooled.view(B, G, 3)
        mask_pooled = mask_grouped.any(dim=1).view(B, G)

        x_global, mask_global_returned = self.next_block(x_pooled_reshaped, pos_pooled_reshaped, mask_pooled)
        
        # 决定all-to-all 还是 one-to-one
        THRESHOLD = 1024
        
        global_pos_emb_base = self.pos_emb_gen(pos_pooled_reshaped)

        if G <= THRESHOLD:
            # --- 策略 A: All-to-All (全图广播) ---
            # [B, G, 2D] -> [B, 1, G, 2D] -> [B, G, G, 2D] -> [BG, G, 2D]
            x_global_ready = x_global.unsqueeze(1).expand(B, G, G, -1).reshape(B * G, G, -1)
            
            global_pos_emb_ready = global_pos_emb_base.unsqueeze(1).expand(B, G, G, -1).reshape(B * G, G, -1)
            
            if mask_global_returned is not None:
                mask_global_ready = mask_global_returned.unsqueeze(1).expand(B, G, G).reshape(B * G, G)
            else:
                mask_global_ready = None
                
        else:
            # --- 策略 B: One-to-One (严格层级) ---
            # [B, G, 2D] -> [BG, 1, 2D]
            x_global_ready = x_global.view(B * G, 1, -1)
            
            global_pos_emb_ready = global_pos_emb_base.view(B * G, 1, -1) # [BG, 1, 60]
            
            if mask_global_returned is not None:
                mask_global_ready = mask_global_returned.view(B * G, 1)
            else:
                mask_global_ready = None

        x_out = self.cross_attn(
            local_x=x_grouped, 
            global_x=x_global_ready, 
            local_pos_emb=pos_emb_grouped, 
            global_pos_emb=global_pos_emb_ready,
            global_mask=mask_global_ready
        )

        # 8. Restore
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
        
        # 假设前3维是坐标
        pos = x[:, :, :3] 

        x_feature = self.pre_proj(x) # [B, N, 64]
        
        x_out, _ = self.ptblock(x_feature, pos, mask)
        
        out = self.dropout(self.ffn(self.norm(x_out)))
        
        return residual + out