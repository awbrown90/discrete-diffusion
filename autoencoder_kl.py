# autoencoder_kl.py

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

class PatchEmbed(nn.Module):
    """2D Image → patch tokens via a single conv stride=patch_size."""
    def __init__(self, img_height, img_width, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        ph, pw = to_2tuple(patch_size)
        self.seq_h = img_height // ph
        self.seq_w = img_width  // pw
        self.seq_len = self.seq_h * self.seq_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(ph,pw), stride=(ph,pw))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                            # → (B, D, H/ps, W/ps)
        x = rearrange(x, "b d h w -> b (h w) d")     # → (B, seq_len, D)
        return self.norm(x)

class DiagonalGaussianDistribution:
    """Helper for VAE bottleneck."""
    def __init__(self, parameters, deterministic: bool, dim: int = 1):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.deterministic = deterministic

    def sample(self):
        if self.deterministic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.std)

    def mode(self):
        return self.mean

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

class AutoencoderKL(nn.Module):
    """
    Unified VAE / VQ-VAE.
      - use_vq=False → (variational) VAE
      - use_vq=True  → discrete codebook VQ-VAE
    """
    def __init__(
        self,
        latent_dim:     int,
        input_height:   int = 256,
        input_width:    int = 256,
        patch_size:     int = 16,
        enc_dim:        int = 768,
        enc_depth:      int = 6,
        enc_heads:      int = 12,
        dec_dim:        int = 768,
        dec_depth:      int = 6,
        dec_heads:      int = 12,
        mlp_ratio:      float = 4.0,
        norm_layer=     functools.partial(nn.LayerNorm, eps=1e-6),
        use_variational:bool = True,
        use_vq:         bool = False,
        codebook_size:  int  = None,
        commitment_cost:float = 0.25,
        **kwargs,
    ):
        super().__init__()
        # patch geometry
        self.seq_h = input_height // patch_size
        self.seq_w = input_width  // patch_size
        self.seq_len   = self.seq_h * self.seq_w
        self.patch_dim = 3 * patch_size * patch_size

        # embed + pos
        self.patch_embed = PatchEmbed(input_height, input_width, patch_size, 3, enc_dim, norm_layer)
        self.pos_emb     = nn.Parameter(torch.zeros(1, self.seq_len, enc_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # encoder transformer
        self.encoder = nn.ModuleList([
            AttentionBlock(enc_dim, enc_heads, self.seq_h, self.seq_w, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(enc_depth)
        ])
        self.enc_norm = norm_layer(enc_dim)

        # decoder transformer
        self.decoder = nn.ModuleList([
            AttentionBlock(dec_dim, dec_heads, self.seq_h, self.seq_w, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(dec_depth)
        ])
        self.dec_norm  = norm_layer(dec_dim)
        self.predictor = nn.Linear(dec_dim, self.patch_dim)

        # bottleneck flags
        self.use_variational = use_variational and not use_vq
        self.use_vq          = use_vq

        if not self.use_vq:
            mult = 2  # mean + logvar
            self.quant_conv      = nn.Linear(enc_dim, mult * latent_dim)
            self.post_quant_conv = nn.Linear(latent_dim, dec_dim)
        else:
            assert codebook_size is not None, "codebook_size required for VQ-VAE"
            self.quant_conv      = nn.Linear(enc_dim, latent_dim)
            self.post_quant_conv = nn.Linear(latent_dim, dec_dim)
            self.vq = VectorQuantize(
                dim                 = latent_dim,
                codebook_size       = codebook_size,
                decay               = 0.99,
                commitment_weight   = commitment_cost,
                use_cosine_sim      = True,
                threshold_ema_dead_code = 2
            )

        # final activation
        self.to_pixels =  nn.Sigmoid()
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
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

    def encode(self, x):
        # x: (B,3,H,W)
        x = self.patch_embed(x)           # → (B, seq_len, enc_dim)
        x = x + self.pos_emb
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)

        if not self.use_vq:
            params = self.quant_conv(x)   # (B, seq_len, 2*latent_dim)
            return DiagonalGaussianDistribution(params, deterministic=not self.use_variational, dim=2)
        else:
            return self.quant_conv(x)     # pre-quant

    def quantize(self, z_e):
        return self.vq(z_e)  # (z_q, idx_map, vq_loss)

    def decode(self, z):
        # z: (B, seq_len, latent_dim)
        z = self.post_quant_conv(z)       # → (B, seq_len, dec_dim)
        z = z + self.pos_emb
        for blk in self.decoder:
            z = blk(z)
        z = self.dec_norm(z)
        patches = self.predictor(z)        # → (B, seq_len, patch_dim)
        dec     = self.unpatchify(patches) # → (B,3,H,W)
        return self.to_pixels(dec)

    def unpatchify(self, x):
        b = x.shape[0]
        x = x.view(b, self.seq_h, self.seq_w, self.patch_dim).permute(0,3,1,2)
        p = int(math.sqrt(self.patch_dim//3))
        x = x.view(b, 3, p, p, self.seq_h, self.seq_w)
        x = x.permute(0,1,4,2,5,3).reshape(b,3,self.seq_h*p,self.seq_w*p)
        return x

    def forward(self, x, sample_posterior: bool = True):
        if not self.use_vq:
            post = self.encode(x)
            z    = post.sample() if (self.use_variational and sample_posterior) else post.mode()
            return self.decode(z), post, z
        else:
            z_e, idx_map, vq_loss = self.encode(x), None, None
            z_q, idx_map, vq_loss = self.quantize(z_e)
            recon = self.decode(z_q)
            return recon, vq_loss, idx_map
