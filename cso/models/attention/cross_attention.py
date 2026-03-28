import torch.nn as nn
from cso.models.attention.multi_headed_attention import MultiHeadedAttention
from cso.models.attention.positionwise_feed_forward import PositionwiseFeedForward


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadedAttention(
            h=h,
            d_model=d_model,
            dropout=dropout
        )
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, queries, memory):
        """
        queries: (B, N_q, D) <- object tokens
        memory: (B, N_m, D) <- container tokens
        """
        
        attn_out = self.attn(
            query=queries,
            key=memory,
            value=memory
        )
        
        x = self.norm1(queries + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x