import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Mlp
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from timm.layers.helpers import to_2tuple
from vector_quantize_pytorch import VectorQuantize

class Attention(nn.Module):
    """Multi-head self-attention with rotary embeddings over the 2D patch grid."""
    def __init__(self, dim, num_heads, seq_h, seq_w, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.seq_h = seq_h
        self.seq_w = seq_w
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        rotary = RotaryEmbedding(dim=head_dim//4, freqs_for="pixel", max_freq=seq_h*seq_w)
        freqs = rotary.get_axial_freqs(seq_h, seq_w)
        self.register_buffer("rotary_freqs", freqs, persistent=False)

    def forward(self, x):
        # x: (B, N, C), N == seq_h * seq_w
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        # reshape to (B, heads, h, w, head_dim)
        head_dim = C // self.num_heads
        q = rearrange(q, "b (H W) (heads d) -> b heads H W d",
                      H=self.seq_h, W=self.seq_w, heads=self.num_heads, d=head_dim)
        k = rearrange(k, "b (H W) (heads d) -> b heads H W d",
                      H=self.seq_h, W=self.seq_w, heads=self.num_heads, d=head_dim)
        v = rearrange(v, "b (H W) (heads d) -> b heads H W d",
                      H=self.seq_h, W=self.seq_w, heads=self.num_heads, d=head_dim)
        # rotary
        q = apply_rotary_emb(self.rotary_freqs, q)
        k = apply_rotary_emb(self.rotary_freqs, k)
        # back to (B, heads, N, head_dim)
        q = rearrange(q, "b heads H W d -> b heads (H W) d")
        k = rearrange(k, "b heads H W d -> b heads (H W) d")
        v = rearrange(v, "b heads H W d -> b heads (H W) d")
        attn = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)
        out  = rearrange(attn, "b heads n d -> b n (heads d)")
        return self.proj(out)

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, seq_h, seq_w,
                 mlp_ratio=4.0, qkv_bias=False,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads, seq_h, seq_w, qkv_bias)
        self.norm2 = norm_layer(dim)
        hidden  = int(dim * mlp_ratio)
        self.mlp  = Mlp(in_features=dim, hidden_features=hidden, act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_emb = nn.Conv2d(3, 512, kernel_size=(20,20), stride=(20,20))
        self.pos_emb = nn.Parameter(torch.zeros(1, 5*7, 512))
        # encoder transformer
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6)
        self.encoder = nn.ModuleList([
            AttentionBlock(512, 8, 5, 7, 4.0, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(6)
        ])
        self.enc_norm = norm_layer(512)

        # decoder transformer
        self.decoder = nn.ModuleList([
            AttentionBlock(512, 8, 5, 7, 4.0, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(6)
        ])
        self.dec_norm  = norm_layer(512)

        # ---------- VectorQuantize layer ----------
        self.vq = VectorQuantize(
            dim              = 512,
            codebook_size    = 84,
            decay            = 0.99,         # EMA γ
            commitment_weight= 0.25,
            use_cosine_sim   = True,          # optional – often helps
            threshold_ema_dead_code = 2       # replace dying codes
        )

        self.predictor = nn.Linear(512, 3 * 20 * 20)
        self.to_pixels = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
        self.apply(init_fn)
        w = self.patch_emb.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

    def encode(self,x):
        x = 2*(x/255.0-0.5)
        x = self.patch_emb(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = x + self.pos_emb
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)
        return x
    
    def decode(self, z):
        # z: (B, seq_len, latent_dim)
        z = z + self.pos_emb
        for blk in self.decoder:
            z = blk(z)
        z = self.dec_norm(z)
        patches = self.predictor(z)        # → (B, seq_len, patch_dim)
        dec     = self.unpatchify(patches) # → (B,3,H,W)
        return self.to_pixels(dec)
    
    def unpatchify(self, x):
        b = x.shape[0]
        x = x.view(b, 5, 7, 3, 20, 20)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, 3, 5 * 20, 7 * 20)

    def forward(self,x):
        # 1) encode 
        z_e = self.encode(x)

        # 2) quantize
        z_q, idx_map, vq_loss_base = self.vq(z_e)

        # 3) decode
        recon = self.decode(z_q)

        # 4) diversity / entropy bonus
        K = self.vq.codebook_size                         # <-- FIX ①
        flat_idx = idx_map.view(-1)
        counts   = torch.bincount(flat_idx, minlength=K).float()
        probs    = counts / counts.sum().clamp_min(1e-12)

        entropy  = -(probs * probs.clamp_min(1e-12).log()).sum()
        div_loss = 0.1 * (math.log(K) - entropy)          # weight still 0.1

        vq_loss  = vq_loss_base + div_loss                # <-- FIX ②
        return recon, vq_loss, idx_map

        



