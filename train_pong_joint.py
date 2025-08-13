# train_pong_joint.py
# Joint training of VQ-ViT (+codebook) and ST world model,
# WITHOUT timm or einops to avoid GUI backend conflicts with OpenCV.

import os, math, random, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from attention import SpatialAxialAttention, TemporalAxialAttention, TimestepEmbedder

# ===================== CONFIG =====================
GRID_H = 5
GRID_W = 7
PATCH  = 20
IMG_H  = GRID_H * PATCH     # 100
IMG_W  = GRID_W * PATCH     # 140

MAX_STEPS        = 50_000
SEED             = 0
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
SHOW_GUI         = True     # set False if debugging with pdb
AUTOREGRESSIVE   = False
RANDOM_MODE      = False
SAVE_CKPT_EVERY  = 0        # 0 disables

# Training schedule
WARMUP_STEPS     = 2_700     # VQ-only first, then joint
VQ_BATCH_SIZE    = 32
IMG_BUF_CAP      = 512
TOK_BUF_CAP      = 1000
ST_BATCH         = 32
SEQ_MIN          = 2
SEQ_MAX          = 8

# Loss weights / LRs
LAMBDA_VQ_RECON  = 1.0
LAMBDA_VQ_COMMIT = 1.0
LR_ST = 1e-4
LR_VQ_WARMUP = 3e-4
LR_VQ_JOINT  = 2e-5
LOAD_CKPT = True

# ===================== SMALL UTILS (no timm/einops) =====================
def to_2tuple(x):
    if isinstance(x, tuple): return x
    return (x, x)

class MlpTiny(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.0):
        super().__init__()
        hidden = hidden_features or in_features * 4
        out    = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, out)
        self.dp  = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = self.dp(x)
        return x

# ===================== PATCH VOCAB (85) =====================
PATTERNS, IDX = [], {}
for combo in itertools.combinations(range(9), 3):
    v = torch.zeros(9, dtype=torch.long)
    v[list(combo)] = 1
    IDX[tuple(v.tolist())] = len(PATTERNS)
    PATTERNS.append(v)
zero = torch.zeros(9, dtype=torch.long)
PATTERNS.insert(0, zero)
IDX.clear()
for i, v in enumerate(PATTERNS):
    IDX[tuple(v.tolist())] = i
VOCAB_SIZE = len(PATTERNS)  # 85

def patch(i):
    return PATTERNS[i].view(3, 3)

def to_img(tokens, zoom=32):
    ext_W = GRID_W + 1
    patches = [patch(int(t)) for t in tokens]
    rows = [torch.cat(patches[r*ext_W:(r+1)*ext_W], dim=1) for r in range(GRID_H)]
    raw = torch.cat(rows, dim=0).numpy().astype(np.uint8)
    grid = raw * 255
    img  = cv2.resize(grid, (grid.shape[1]*zoom, grid.shape[0]*zoom), cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# ===================== ACTION TOKENS =====================
zero_idx = IDX[tuple(zero.tolist())]

vert = torch.zeros((3,3), dtype=torch.long); vert[:,1] = 1
vertical_idx = IDX[tuple(vert.flatten().tolist())]
uhoriz = torch.zeros((3,3), dtype=torch.long); uhoriz[0,:] = 1
uphorizontal_idx = IDX[tuple(uhoriz.flatten().tolist())]

ru_horiz = torch.zeros((3,3), dtype=torch.long)
ru_horiz[0,2] = 1; ru_horiz[0,1] = 1; ru_horiz[1,2] = 1
ru_horizontal_idx = IDX[tuple(ru_horiz.flatten().tolist())]
lu_horiz = torch.zeros((3,3), dtype=torch.long)
lu_horiz[0,0] = 1; lu_horiz[0,1] = 1; lu_horiz[1,0] = 1
lu_horizontal_idx = IDX[tuple(lu_horiz.flatten().tolist())]
rl_horiz = torch.zeros((3,3), dtype=torch.long)
rl_horiz[2,2] = 1; rl_horiz[2,1] = 1; rl_horiz[1,2] = 1
rl_horizontal_idx = IDX[tuple(rl_horiz.flatten().tolist())]
ll_horiz = torch.zeros((3,3), dtype=torch.long)
ll_horiz[2,0] = 1; ll_horiz[2,1] = 1; ll_horiz[1,0] = 1
ll_horizontal_idx = IDX[tuple(ll_horiz.flatten().tolist())]

horiz = torch.zeros((3,3), dtype=torch.long)
horiz[0,0] = 1; horiz[1,1] = 1; horiz[2,2] = 1
horizontal_idx = IDX[tuple(horiz.flatten().tolist())]
dhoriz = torch.zeros((3,3), dtype=torch.long); dhoriz[2,:] = 1
dnhorizontal_idx = IDX[tuple(dhoriz.flatten().tolist())]

balls = [ru_horizontal_idx, lu_horizontal_idx, rl_horizontal_idx, ll_horizontal_idx]

uaction = IDX[tuple(torch.tensor([[0,1,0],[1,0,1],[0,0,0]], dtype=torch.long).flatten().tolist())]
daction = IDX[tuple(torch.tensor([[0,0,0],[1,0,1],[0,1,0]], dtype=torch.long).flatten().tolist())]
saction = IDX[tuple(torch.tensor([[0,0,0],[1,1,1],[0,0,0]], dtype=torch.long).flatten().tolist())]
actions = [uaction, daction, saction]

def extend_with_action_col(state: torch.Tensor, action_idx: int, void_idx: int) -> torch.Tensor:
    mat = state.view(GRID_H, GRID_W)
    col = torch.full((GRID_H,), void_idx, dtype=state.dtype, device=state.device)
    col[-1] = action_idx
    ext = torch.cat([mat, col.unsqueeze(1)], dim=1)
    return ext.view(-1)

def undo_extend_with_action_col(extended_state: torch.Tensor) -> torch.Tensor:
    extended_state = extended_state.view(GRID_H, GRID_W + 1)
    original_mat = extended_state[:, :-1]
    return original_mat.reshape(-1)

# ===================== ENV =====================
class PongEnv:
    def __init__(self, device, grid_h=GRID_H, grid_w=GRID_W, patch_size=PATCH, ball_size=5, random_miss=True):
        self.r_paddle   = 0
        self.grid_h     = grid_h
        self.grid_w     = grid_w
        self.ball_r     = 0
        self.ball_c     = 1
        self.ball_d_c   = 1
        self.ball_d_r   = 1
        self.ball_idx   = 0
        self.reset_flag = False
        self.miss_counter = 0
        self.patch_size = patch_size
        self.ball_size  = ball_size
        self.height     = grid_h * patch_size
        self.width      = grid_w * patch_size
        self.device     = device
        self.random_miss = random_miss

    def render(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # paddle
        y0 = self.r_paddle * self.patch_size
        img[y0:y0 + self.patch_size, 0:5] = (0, 255, 0)
        # ball
        if self.ball_idx != 4:
            cell_x = self.ball_c * self.patch_size
            cell_y = self.ball_r * self.patch_size
            if   self.ball_idx == 0: px, py = cell_x + (self.patch_size - self.ball_size), cell_y
            elif self.ball_idx == 1: px, py = cell_x, cell_y
            elif self.ball_idx == 2: px, py = cell_x + (self.patch_size - self.ball_size), cell_y + (self.patch_size - self.ball_size)
            elif self.ball_idx == 3: px, py = cell_x, cell_y + (self.patch_size - self.ball_size)
            img[py:py + self.ball_size, px:px + self.ball_size] = (0, 0, 255)
        tensor = torch.from_numpy(img).permute(2, 0, 1).to(self.device)
        return tensor

    def getState(self):
        return self.render()

    def step(self, action: int):
        if action == 0:      self.r_paddle = max(self.r_paddle - 1, 0)
        elif action == 1:    self.r_paddle = min(self.r_paddle + 1, self.grid_h - 1)
        self.ball_c += self.ball_d_c
        self.ball_r += self.ball_d_r
        if self.ball_c >= self.grid_w - 1:
            self.ball_d_c = -1
        elif self.ball_c <= 1:
            if self.ball_idx != 4 and self.r_paddle != self.ball_r:
                self.reset_flag = True
                self.ball_idx = 4
                self.miss_counter += 1
            self.ball_d_c = 1
        if self.ball_r >= self.grid_h - 1:
            self.ball_d_r = -1
        elif self.ball_r <= 0:
            self.ball_d_r = 1
        if not self.reset_flag:
            if   self.ball_d_c == 1 and self.ball_d_r == -1: self.ball_idx = 0
            elif self.ball_d_c == -1 and self.ball_d_r == -1: self.ball_idx = 1
            elif self.ball_d_c == 1 and self.ball_d_r ==  1: self.ball_idx = 2
            elif self.ball_d_c == -1 and self.ball_d_r ==  1: self.ball_idx = 3
        else:
            self.reset_flag = False
        return self.render()

# ===================== BUFFERS =====================
class ReplayBuffer:
    """Token replay for ST (stores extended VQ-token frames)."""
    def __init__(self, observation_shape, capacity, device):
        self.device = device
        self.capacity = int(capacity)
        self.observations = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.bufferIndex = 0
        self.full = False
    def __len__(self): return self.capacity if self.full else self.bufferIndex
    def add(self, observation):
        self.observations[self.bufferIndex] = observation
        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or (self.bufferIndex == 0)
    def sample(self, batchSize, sequenceSize):
        lastIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastIndex > batchSize), "Not enough data in buffer"
        idx = np.random.randint(0, self.capacity if self.full else lastIndex, batchSize).reshape(-1, 1)
        seq = np.arange(sequenceSize).reshape(1, -1)
        idx = (idx + seq) % self.capacity
        obs = torch.as_tensor(self.observations[idx], dtype=torch.long, device=self.device)
        return obs.view(batchSize, -1)

class ImageReplayBuffer:
    """Image replay for VQ (stores recent CHW [0,1] frames)."""
    def __init__(self, capacity, device, h=IMG_H, w=IMG_W):
        self.capacity = int(capacity)
        self.device = device
        self.buffer = torch.empty((self.capacity, 3, h, w), dtype=torch.float32, device=self.device)
        self.ptr = 0
        self.full = False
    def __len__(self): return self.capacity if self.full else self.ptr
    @torch.no_grad()
    def add(self, img_chw_01: torch.Tensor):
        self.buffer[self.ptr].copy_(img_chw_01)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0: self.full = True
    def sample(self, batch_size):
        assert len(self) > 0
        idx = torch.randint(0, len(self), (batch_size,), device=self.device)
        return self.buffer[idx]

# ===================== VQ-ViT (no timm / no einops) =====================
class PatchEmbed(nn.Module):
    def __init__(self, img_height, img_width, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        ph, pw = to_2tuple(patch_size)
        self.seq_h = img_height // ph
        self.seq_w = img_width // pw
        self.seq_len = self.seq_h * self.seq_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(ph, pw), stride=(ph, pw))
        self.norm = norm_layer(embed_dim)
    def forward(self, x):  # x: (B, C, H, W)
        x = self.proj(x)                      # (B, D, H/ps, W/ps)
        B, D, Hps, Wps = x.shape
        x = x.view(B, D, Hps * Wps).transpose(1, 2).contiguous()  # (B, N, D)
        return self.norm(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, seq_h, seq_w, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.seq_h, self.seq_w = seq_h, seq_w
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        rotary = RotaryEmbedding(dim=self.head_dim//4 if self.head_dim>=4 else 1, freqs_for="pixel", max_freq=seq_h*seq_w)
        freqs = rotary.get_axial_freqs(seq_h, seq_w)
        self.register_buffer("rotary_freqs", freqs, persistent=False)
    def forward(self, x):  # x: (B, N, C), N=seq_h*seq_w
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        qkv = self.qkv(x)                                   # (B, N, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)                      # each (B, N, C)
        # -> (B, H, N, D)
        q = q.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        # reshape N -> (seq_h, seq_w) to apply axial rotary
        q = q.view(B, H, self.seq_h, self.seq_w, D)
        k = k.view(B, H, self.seq_h, self.seq_w, D)
        q = apply_rotary_emb(self.rotary_freqs, q)
        k = apply_rotary_emb(self.rotary_freqs, k)
        # back to (B, H, N, D)
        q = q.view(B, H, -1, D)
        k = k.view(B, H, -1, D)
        # scaled dot-product attention (PyTorch fused)
        attn = F.scaled_dot_product_attention(q, k, v)      # (B, H, N, D)
        out  = attn.permute(0, 2, 1, 3).contiguous().view(B, N, H*D)  # (B, N, C)
        return self.proj(out)

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, seq_h, seq_w, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads, seq_h, seq_w, qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp   = MlpTiny(in_features=dim, hidden_features=int(dim*mlp_ratio), out_features=dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AutoencoderKL(nn.Module):
    """VQ-VAE (VQ path only)."""
    def __init__(self,
        latent_dim:     int,
        input_height:   int = IMG_H,
        input_width:    int = IMG_W,
        patch_size:     int = PATCH,
        enc_dim:        int = 128,
        enc_depth:      int = 2,
        enc_heads:      int = 4,
        dec_dim:        int = 128,
        dec_depth:      int = 2,
        dec_heads:      int = 4,
        mlp_ratio:      float = 4.0,
        norm_layer     = nn.LayerNorm,
        codebook_size:  int  = VOCAB_SIZE,
        commitment_cost:float = 0.25,
    ):
        super().__init__()
        self.seq_h = input_height // patch_size
        self.seq_w = input_width // patch_size
        self.seq_len   = self.seq_h * self.seq_w
        self.patch_dim = 3 * patch_size * patch_size

        self.patch_embed = PatchEmbed(input_height, input_width, patch_size, 3, enc_dim, norm_layer)
        self.pos_emb     = nn.Parameter(torch.zeros(1, self.seq_len, enc_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.encoder = nn.ModuleList([
            AttentionBlock(enc_dim, enc_heads, self.seq_h, self.seq_w, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(enc_depth)
        ])
        self.enc_norm = norm_layer(enc_dim)

        self.decoder = nn.ModuleList([
            AttentionBlock(dec_dim, dec_heads, self.seq_h, self.seq_w, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(dec_depth)
        ])
        self.dec_norm  = norm_layer(dec_dim)
        self.predictor = nn.Linear(dec_dim, self.patch_dim)

        # VQ
        from vector_quantize_pytorch import VectorQuantize
        self.quant_conv      = nn.Linear(enc_dim, latent_dim)
        self.post_quant_conv = nn.Linear(latent_dim, dec_dim)
        self.vq = VectorQuantize(
            dim                     = latent_dim,
            codebook_size           = codebook_size,
            decay                   = 0.99,
            commitment_weight       = commitment_cost,
            use_cosine_sim          = True,
            threshold_ema_dead_code = 2
        )

        self.to_pixels = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias); nn.init.ones_(m.weight)
        self.apply(init_fn)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

    def encode(self, x):
        x = self.patch_embed(x)                 # (B, N, D)
        x = x + self.pos_emb
        for blk in self.encoder: x = blk(x)
        x = self.enc_norm(x)
        return self.quant_conv(x)               # (B, N, latent_dim)

    def quantize(self, z_e):
        return self.vq(z_e)                     # (z_q, idx_map, vq_loss)

    def decode(self, z):
        z = self.post_quant_conv(z)             # (B, N, dec_dim)
        z = z + self.pos_emb
        for blk in self.decoder: z = blk(z)
        z = self.dec_norm(z)
        patches = self.predictor(z)             # (B, N, patch_dim)
        return self.unpatchify(patches)

    def unpatchify(self, x):
        B = x.shape[0]
        # x: (B, N, 3*p*p) → (B, 3, H, W)
        N = x.shape[1]
        p = int(math.sqrt(self.patch_dim // 3))
        Hps, Wps = self.seq_h, self.seq_w
        x = x.view(B, Hps, Wps, 3 * p * p).permute(0,3,1,2).contiguous()   # (B, 3*p*p, Hps, Wps)
        x = x.view(B, 3, p, p, Hps, Wps).permute(0,1,4,2,5,3).contiguous() # (B,3,Hps,p,Wps,p)
        x = x.view(B, 3, Hps * p, Wps * p)                                  # (B,3,H,W)
        return self.to_pixels(x)

    def forward(self, x):
        z_e = self.encode(x)
        z_q, idx_map, vq_loss = self.quantize(z_e)
        recon = self.decode(z_q)
        return recon, vq_loss, idx_map

# ===================== ST MODEL (unchanged) =====================
class SpatioTemporalBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        dim_head = d // heads
        self.spatial_rotary = RotaryEmbedding(dim=dim_head//2, freqs_for="pixel")
        self.temporal_rotary = RotaryEmbedding(dim=dim_head,   freqs_for="lang")
        self.spatial  = SpatialAxialAttention(d, heads, dim_head, self.spatial_rotary)
        self.temporal = TemporalAxialAttention(d, heads, dim_head, self.temporal_rotary)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp   = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d*4),
            nn.GELU(),
            nn.Linear(d*4, d),
        )
    def forward(self, x):
        y = self.norm1(x); y = self.spatial(y); x = x + y
        y = self.norm2(x); y = self.temporal(y); x = x + y
        B,T,H,W,d = x.shape
        y = self.mlp(x.reshape(B*T*H*W, d)).reshape(B, T, H, W, d)
        return x + y

class STModel(nn.Module):
    def __init__(self, d=512, heads=4, num_layers=3, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.H, self.W = GRID_H, GRID_W + 1
        self.d = d
        self.token_emb    = nn.Embedding(vocab_size, d)
        self.time_embedder= TimestepEmbedder(hidden_size=d, frequency_embedding_size=d)
        self.pos2d        = nn.Parameter(torch.zeros(self.H*self.W, d))
        self.blocks       = nn.ModuleList([SpatioTemporalBlock(d, heads) for _ in range(num_layers)])
        self.head         = nn.Linear(d, vocab_size)
    def forward(self, seq):
        B,N = seq.shape
        assert N % (self.H*self.W) == 0
        T = N // (self.H*self.W)
        tok = self.token_emb(seq)
        sp  = self.pos2d.unsqueeze(0).repeat(B, T, 1).view(B, N, self.d)
        x = tok + sp
        idx= torch.arange(T, device=seq.device)
        te = self.time_embedder(idx)
        te = te.unsqueeze(1).expand(T, self.H*self.W, self.d).reshape(1, N, self.d).repeat(B,1,1)
        x = x + te
        x = x.view(B, T, self.H, self.W, self.d)
        for blk in self.blocks: x = blk(x)
        x = x.view(B, N, self.d)
        return self.head(x)

# ===================== UTILS =====================
def noisy_sample(seq: torch.LongTensor, frame_size: int) -> torch.LongTensor:
    N = seq.numel()
    T = N // frame_size
    frames = seq.view(T, frame_size).clone()
    e     = torch.rand(T, device=seq.device)
    mask  = torch.rand(T, frame_size, device=seq.device) < e.unsqueeze(1)
    rand  = torch.randint(0, VOCAB_SIZE, (T, frame_size), device=seq.device)
    frames[mask] = rand[mask]
    return frames.view(-1)

def save_checkpoint(st_model, vq_model, opt_st, opt_vq, tok_buf, img_buf, step, filename):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    state = {
        'step': step,
        'st_state': st_model.state_dict(),
        'vq_state': vq_model.state_dict(),
        'opt_st': opt_st.state_dict(),
        'opt_vq': opt_vq.state_dict(),
        'tok_observations': torch.from_numpy(tok_buf.observations).cpu(),
        'tok_index': tok_buf.bufferIndex,
        'tok_full': tok_buf.full,
        'img_buffer': img_buf.buffer.cpu(),
        'img_ptr': img_buf.ptr,
        'img_full': img_buf.full,
    }
    torch.save(state, filename)
    print(f"[ckpt] Saved at step {step} → {filename}")

def _pick_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None

def load_joint_checkpoint(path, st, vq, opt_st=None, opt_vq=None, tok_buf=None, img_buf=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=True)

    # 1) Try to find model state_dicts under flexible keys
    st_key = _pick_key(ckpt, ['st_state', 'st', 'st_model', 'state_dict_st', 'st_dict'])
    vq_key = _pick_key(ckpt, ['vq_state', 'vq', 'vq_model', 'state_dict_vq', 'vq_dict'])

    if st_key is None and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        st_sub = {k.split('st.',1)[1]: v for k,v in sd.items() if k.startswith('st.')}
        vq_sub = {k.split('vq.',1)[1]: v for k,v in sd.items() if k.startswith('vq.')}
        if st_sub: st.load_state_dict(st_sub, strict=True)
        if vq_sub: vq.load_state_dict(vq_sub, strict=True)
    else:
        if st_key is not None:
            st.load_state_dict(ckpt[st_key], strict=True)
        else:
            print(f"[load] ST weights not found in '{path}'. Keys: {list(ckpt.keys())}")

        if vq_key is not None:
            vq.load_state_dict(ckpt[vq_key], strict=True)
        else:
            print(f"[load] VQ weights not found in '{path}'. Keys: {list(ckpt.keys())}")

    # 2) Optimizers (optional)
    if opt_st is not None:
        opt_st_key = _pick_key(ckpt, ['opt_st', 'optimizer_st', 'st_opt'])
        if opt_st_key: 
            try: opt_st.load_state_dict(ckpt[opt_st_key])
            except Exception as e: print(f"[load] Skipped opt_st: {e}")

    if opt_vq is not None:
        opt_vq_key = _pick_key(ckpt, ['opt_vq', 'optimizer_vq', 'vq_opt'])
        if opt_vq_key:
            try: opt_vq.load_state_dict(ckpt[opt_vq_key])
            except Exception as e: print(f"[load] Skipped opt_vq: {e}")

    # 3) Buffers (if present)
    #if tok_buf is not None and 'tok_observations' in ckpt:
    #    tok_buf.observations = ckpt['tok_observations'].numpy()
    #    tok_buf.bufferIndex = ckpt['tok_index']
    #    tok_buf.full = ckpt['tok_full']
    #    print("[load] Loaded token buffer")

    if img_buf is not None and 'img_buffer' in ckpt:
        img_buf.buffer.copy_(ckpt['img_buffer'].to(device))
        img_buf.ptr = ckpt['img_ptr']
        img_buf.full = ckpt['img_full']
        print("[load] Loaded image buffer")

    step = ckpt.get('step', 0)
    print(f"[load] Loaded checkpoint from '{path}' (step={step})")
    return step

# ===================== MAIN =====================
def run():
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    dev = torch.device(DEVICE)

    # Models & opts
    st = STModel().to(dev)
    vq = AutoencoderKL(
        latent_dim     = 256,
        input_height   = IMG_H,
        input_width    = IMG_W,
        patch_size     = PATCH,
        enc_dim        = 128, enc_depth = 2, enc_heads = 4,
        dec_dim        = 128, dec_depth = 2, dec_heads = 4,
        mlp_ratio      = 4.0,
        codebook_size  = VOCAB_SIZE,
        commitment_cost= 0.25
    ).to(dev)

    opt_st = torch.optim.AdamW(st.parameters(), lr=LR_ST)
    opt_vq = torch.optim.AdamW(vq.parameters(), lr=LR_VQ_WARMUP)

    # Buffers
    frame_size_ext = GRID_H * (GRID_W + 1)
    tok_buf = ReplayBuffer(observation_shape=(frame_size_ext,), capacity=TOK_BUF_CAP, device=dev)
    img_buf = ImageReplayBuffer(capacity=IMG_BUF_CAP, device=dev, h=IMG_H, w=IMG_W)

    # Load checkpoint
    start_step = 0
    if LOAD_CKPT:
        ckpt_path = 'ckpts/joint_ckpt.pth'
        start_step = load_joint_checkpoint(ckpt_path, st, vq, opt_st, opt_vq, tok_buf, img_buf, device=dev)
        vq.eval()

    env_train = PongEnv(dev, random_miss=RANDOM_MODE)
    env_test  = PongEnv(dev, random_miss=RANDOM_MODE)

    # Initialize live with two consecutive real frames for better demo
    with torch.no_grad():
        current_state = env_test.getState()
        img01 = current_state.float() / 255.0
        _, _, idx_map = vq(img01.unsqueeze(0))
        z1 = idx_map.squeeze(0)
        action = random.randint(0, len(actions)-1)
        ext1 = extend_with_action_col(z1, actions[action], zero_idx)

        next_state = env_test.step(action)
        img02 = next_state.float() / 255.0
        _, _, idx_map2 = vq(img02.unsqueeze(0))
        z2 = idx_map2.squeeze(0)
        action2 = random.randint(0, len(actions)-1)
        ext2 = extend_with_action_col(z2, action2, zero_idx)

        live = torch.cat([ext1, ext2]).unsqueeze(0).to(dev)

    learning = False  # Set to False for debugging
    generate = True
    interactive = False

    criterion = nn.MSELoss()

    try:
        for step in range(start_step + 1, MAX_STEPS + 1):
            # --- collect current frame ---
            action = random.randint(0, len(actions)-1)
            current_state = env_train.getState()              # CHW uint8
            img01 = current_state.float() / 255.0             # [0,1]
            img_buf.add(img01)

            # get tokens for current frame
            recon_cur, vq_commit_loss_cur, idx_map_cur = vq(img01.unsqueeze(0))
            cur_z = idx_map_cur.squeeze(0)

            env_train.step(action)

            # extend with action col; add to token buffer
            s0_ext = extend_with_action_col(cur_z, actions[action], zero_idx)
            tok_buf.add(s0_ext.detach().cpu().numpy())

            if step == WARMUP_STEPS:
                for g in opt_vq.param_groups:
                    g["lr"] = LR_VQ_JOINT
                print(f"[Info] Lowered VQ LR to {LR_VQ_JOINT} at step {step}")

            # --- train ---
            if learning and len(tok_buf) >= 64:
                loss = torch.tensor(0.0, device=dev)

                # 1) VQ loss input: prefer replay batch when available, else fallback to current frame
                if len(img_buf) >= VQ_BATCH_SIZE:
                    vq_in = img_buf.sample(VQ_BATCH_SIZE)                 # (B,3,H,W), [0,1]
                else:
                    vq_in = img01.unsqueeze(0)                            # (1,3,H,W) — early bootstrapping

                # Forward VQ on chosen input
                recon_b, vq_commit_b, _ = vq(vq_in)
                loss_recon = criterion(recon_b, vq_in)                    # or MSE if you prefer
                loss_vq = LAMBDA_VQ_RECON * loss_recon + LAMBDA_VQ_COMMIT * vq_commit_b

                if step < WARMUP_STEPS:
                    # --- Warmup: train ONLY the VQ, skip ST completely (no forward/no grad) ---
                    opt_vq.zero_grad(set_to_none=True)
                    loss_vq.backward()
                    torch.nn.utils.clip_grad_norm_(vq.parameters(), 1.0)
                    opt_vq.step()
                    loss = loss_vq
                else:
                    # --- Joint phase: compute ST loss on token sequences + train both ---
                    # Sample a token sequence batch for ST
                    if len(tok_buf) >= 64:
                        n   = random.randint(SEQ_MIN, SEQ_MAX)
                        ex  = tok_buf.sample(ST_BATCH, n)
                        src = torch.stack([noisy_sample(e, frame_size=frame_size_ext) for e in ex], dim=0).to(dev)
                        tgt = torch.stack([e for e in ex], dim=0).to(dev)

                        logits = st(src)
                        loss_st = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt.view(-1))

                        opt_st.zero_grad(set_to_none=True)
                        opt_vq.zero_grad(set_to_none=True)
                        (loss_st + loss_vq).backward()
                        torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(vq.parameters(), 1.0)
                        opt_st.step()
                        opt_vq.step()
                        loss = loss_st + loss_vq
                    else:
                        # Not enough token data yet → at least keep improving VQ
                        opt_vq.zero_grad(set_to_none=True)
                        loss_vq.backward()
                        torch.nn.utils.clip_grad_norm_(vq.parameters(), 1.0)
                        opt_vq.step()
                        loss = loss_vq
            else:
                loss = torch.tensor(0.0, device=dev)

            # --- autoregressive demo ---
            if generate:
                with torch.no_grad():
                    logits = st(live)
                    preds  = logits.argmax(dim=-1)
                    last_idx = (GRID_H-1)*(GRID_W+1) + GRID_W
                    live[:, frame_size_ext:] = preds[:, frame_size_ext:]
                    live[0, last_idx] = preds[0, last_idx]

            # --- display (same style as working script) ---
            if SHOW_GUI and (step % 2 == 0):
                with torch.no_grad():
                    sl = live[0, :frame_size_ext]
                    sr = live[0, frame_size_ext:]
                    sl0 = undo_extend_with_action_col(sl)
                    sr0 = undo_extend_with_action_col(sr)
                    codebook = vq.vq.codebook
                    sl0_vectors = codebook[sl0]
                    sr0_vectors = codebook[sr0]
                    dl0 = vq.decode(sl0_vectors.unsqueeze(0))
                    dr0 = vq.decode(sr0_vectors.unsqueeze(0))
                    left  = (dl0.squeeze(0).permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
                    right = (dr0.squeeze(0).permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
                    sep   = np.full((left.shape[0], 10, 3), 128, dtype=np.uint8)
                    vis   = np.hstack([left, sep, right])

                vis = np.ascontiguousarray(vis)
                cv2.imshow('grid(L) | target(R)', vis)
                key = cv2.waitKey(1) & 0xFF
                if   key == 27: break
                elif key == ord(' '):
                    live = torch.cat([
                        torch.randint(0, VOCAB_SIZE, (frame_size_ext,)),
                        torch.randint(0, VOCAB_SIZE, (frame_size_ext,))
                    ]).unsqueeze(0).to(dev)
                elif key == ord('m'):
                    learning = not learning
                elif key == ord('g'):
                    generate = not generate
                elif key == ord('k'):
                    interactive = not interactive
                elif key == ord('s'):
                    save_checkpoint(st, vq, opt_st, opt_vq, tok_buf, img_buf, step, 'ckpts/joint_ckpt.pth')
                # --- interactive arrow keys: Up(82), Down(84), Right(83) ---
                elif interactive and key in [82, 84, 83]:
                    if   key == 82: chosen = 0  # up
                    elif key == 84: chosen = 1  # down
                    else:           chosen = 2  # stay/right

                    # step env_test with chosen action, encode with VQ, update 'live'
                    new_state = env_test.step(chosen)  # CHW uint8
                    with torch.no_grad():
                        _, _, zidx = vq((new_state.float() / 255.0).unsqueeze(0))
                    z = zidx.squeeze(0)  # (H*W)

                    state_ext = extend_with_action_col(z, actions[chosen], zero_idx)  # (H*(W+1))
                    live = torch.cat([
                        state_ext,
                        torch.randint(0, VOCAB_SIZE, (frame_size_ext,)).to(dev)
                    ]).unsqueeze(0)
                    # reflect chosen action token at last cell of the second frame
                    live[0, last_idx] = actions[chosen]

            # --- periodic live step when not interactive ---
            if not interactive and step % 10 == 0:
                last_idx = (GRID_H-1)*(GRID_W+1) + GRID_W
                if AUTOREGRESSIVE:
                    live = torch.cat([
                        live[0, frame_size_ext:],
                        torch.randint(0, VOCAB_SIZE, (frame_size_ext,)).to(dev)
                    ]).unsqueeze(0)
                    live[0, last_idx] = torch.randint(0, VOCAB_SIZE, (1,)).to(dev)
                else:
                    action = random.randint(0, len(actions)-1)
                    new_state = env_test.step(action)
                    with torch.no_grad():
                        _, _, zidx = vq((new_state.float()/255.0).unsqueeze(0))
                    z = zidx.squeeze(0)
                    state_ext = extend_with_action_col(z, actions[action], zero_idx)
                    live = torch.cat([
                        state_ext,
                        torch.randint(0, VOCAB_SIZE, (frame_size_ext,)).to(dev)
                    ]).unsqueeze(0)
                    live[0, last_idx] = torch.randint(0, VOCAB_SIZE, (1,)).to(dev)

            if step % 50 == 0:
                phase = "VQ-warmup" if step < WARMUP_STEPS else "Joint"
                print(f"step {step:6d} | loss {loss.item():.4f} | {phase} | buf(img,tok)=({len(img_buf)},{len(tok_buf)})")

            if SAVE_CKPT_EVERY and (step % SAVE_CKPT_EVERY == 0):
                save_checkpoint(st, vq, opt_st, opt_vq, tok_buf, img_buf, step, f'ckpts/joint_step{step}.pth')

    finally:
        if SHOW_GUI:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass
        if SAVE_CKPT_EVERY:
            save_checkpoint(st, vq, opt_st, opt_vq, tok_buf, img_buf, step, 'ckpts/joint_last.pth')

if __name__ == '__main__':
    run()