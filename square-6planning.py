import torch
from torch import nn
import random
from attention import SpatialAxialAttention, TemporalAxialAttention, TimestepEmbedder
from rotary_embedding_torch import RotaryEmbedding
import copy

# square_diffusion_transformer.py  (fixed greedy update)
"""
Discrete "diffusion" over a 3×3 grid using a Transformer encoder.
Now uses **greedy** argmax resampling, which converges quickly.
"""

import itertools
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def save_checkpoint(model, optimizer, step, filename='ckpt.pth'):
    torch.save({
        'step':  step,
        'model': model.state_dict(),
        'opt':   optimizer.state_dict(),
    }, filename)
    print("Saved checkpoint to square_diffusion.ckpt")

def load_checkpoint(model, optimizer, filename='ckpt.pth', device='cpu'):
    ckpt = torch.load(filename, map_location=device)
    model.load_state_dict( ckpt['model'] )
    optimizer.load_state_dict( ckpt['opt'] )
    return ckpt.get('step', None)

class ReplayBuffer(object):
    def __init__(self, observation_shape, capacity, device):
      
        self.device = device
        self.capacity = int(capacity)

        self.observations = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        
        self.bufferIndex = 0
        self.full = False
        
    def __len__(self):
        return self.capacity if self.full else self.bufferIndex

    def add(self, observation):
        self.observations[self.bufferIndex]     = observation

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def sample(self, batchSize, sequenceSize):
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastFilledIndex > batchSize), "not enough data in the buffer to sample"
        sampleIndex = np.random.randint(0, self.capacity if self.full else lastFilledIndex, batchSize).reshape(-1, 1)
        sequenceLength = np.arange(sequenceSize).reshape(1, -1)

        sampleIndex = (sampleIndex + sequenceLength) % self.capacity

        observations = torch.as_tensor(self.observations[sampleIndex], dtype=torch.long, device=self.device)
        
        return observations.reshape(batchSize,-1)

# ----------------------- utilities -----------------------------------------
PATTERNS, IDX = [], {}
for combo in itertools.combinations(range(9), 3):
    v = torch.zeros(9, dtype=torch.long); v[list(combo)] = 1
    IDX[tuple(v.tolist())] = len(PATTERNS); PATTERNS.append(v)
zero = torch.zeros(9, dtype=torch.long)
PATTERNS.insert(0, zero)
# rebuild IDX so that the void tuple → 0
IDX.clear()
for i, v in enumerate(PATTERNS):
    IDX[tuple(v.tolist())] = i


def pt(mat):
    return IDX[tuple([e for row in mat for e in row])]

def patch(i):
    return PATTERNS[i].view(3,3)

def to_img(tokens, zoom=32):
    patches=[patch(int(t)) for t in tokens]
    rows = [torch.cat(patches[r*4:(r+1)*4], 1) for r in range(3)]
    raw = torch.cat(rows,0).numpy().astype(np.uint8)
    grid = raw * 255
    img=cv2.resize(grid,(grid.shape[1]*zoom,grid.shape[0]*zoom),cv2.INTER_NEAREST)
    return cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2BGR)

# ----------------------- target --------------------------------------------
void = pt([[0,0,0],[0,0,0],[0,0,0]])
vertical   = pt([[0,1,0],[0,1,0],[0,1,0]])

saction = pt([[0,0,0],[0,0,0],[0,0,0]])
raction = pt([[0,1,0],[0,0,1],[0,1,0]])
uaction = pt([[0,1,0],[1,0,1],[0,0,0]])
laction = pt([[0,1,0],[1,0,0],[0,1,0]])
daction = pt([[0,0,0],[1,0,1],[0,1,0]])

actions= [saction,raction,uaction,laction,daction]

T1=torch.tensor([
    vertical,void,void,
    void,void,void,
    void,void,void],dtype=torch.long)
T2=torch.tensor([
    void,vertical,void,
    void,void,void,
    void,void,void],dtype=torch.long)
T3=torch.tensor([
    void,void,vertical,
    void,void,void,
    void,void,void],dtype=torch.long)
T4=torch.tensor([
    void,void,void,
    vertical,void,void,
    void,void,void],dtype=torch.long)
T5=torch.tensor([
    void,void,void,
    void,vertical,void,
    void,void,void],dtype=torch.long)
T6=torch.tensor([
    void,void,void,
    void,void,vertical,
    void,void,void],dtype=torch.long)
T7=torch.tensor([
    void,void,void,
    void,void,void,
    vertical,void,void],dtype=torch.long)
T8=torch.tensor([
    void,void,void,
    void,void,void,
    void,vertical,void],dtype=torch.long)
T9=torch.tensor([
    void,void,void,
    void,void,void,
    void,void,vertical],dtype=torch.long)

STATES = [T1, T2, T3,
          T4, T5, T6,
          T7, T8, T9]

class SpatioTemporalBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        dim_head = d // heads

        # rotary embeddings
        self.spatial_rotary = RotaryEmbedding(dim=dim_head//2, freqs_for="pixel")
        self.temporal_rotary = RotaryEmbedding(dim=dim_head, freqs_for="lang")

        # axis-wise attention modules
        self.spatial = SpatialAxialAttention(d, heads, dim_head, self.spatial_rotary)
        self.temporal = TemporalAxialAttention(d, heads, dim_head, self.temporal_rotary)

        # optional layernorms & MLP
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp   = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d*4),
            nn.GELU(),
            nn.Linear(d*4, d),
        )

    def forward(self, x):
        # x: (B, T, H, W, d)
        # --- spatial pass ---
        y = self.norm1(x)
        y = self.spatial(y)            # (B, T, H, W, d)
        x = x + y

        # --- temporal pass ---
        y = self.norm2(x)
        y = self.temporal(y)           # (B, T, H, W, d)
        x = x + y

        # --- MLP ---
        B,T,H,W,d = x.shape
        y = self.mlp(x.view(B*T*H*W, d)).view(B, T, H, W, d)
        return x + y

class STModel(nn.Module):
    def __init__(self, d=128, heads=4, num_layers=3, H=3, W=4, vocab_size=85):
        super().__init__()
        self.d = d; self.H = H; self.W = W

        # token embedding
        self.token_emb = nn.Embedding(vocab_size, d)

        self.time_embedder = TimestepEmbedder(hidden_size=d,
                                              frequency_embedding_size=d)

        # 2D spatial pos
        self.pos2d = nn.Parameter(torch.zeros(H*W, d))

        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(d, heads) for _ in range(num_layers)
        ])
        self.head   = nn.Linear(d, vocab_size)

    def forward(self, seq):
        B, N = seq.shape
        T    = N // (self.H * self.W)
        assert T * (self.H * self.W) == N

        # token + spatial pos
        tok = self.token_emb(seq)                            # (B, N, d)
        sp  = self.pos2d.unsqueeze(0).repeat(B, T, 1)        # (B, T, H*W, d)
        sp  = sp.view(B, N, self.d)
        tok = tok + sp

        # --- continuous time embeddings ---
        idx = torch.arange(T, device=seq.device)             # shape (T,)
        te  = self.time_embedder(idx)                        # (T, d)
        te  = te.unsqueeze(1).expand(T, self.H*self.W, self.d)   # (T, H*W, d)
        te  = te.reshape(1, N, self.d).repeat(B, 1, 1)           # (B, N, d)
        tok = tok + te

        # run ST blocks
        x = tok.view(B, T, self.H, self.W, self.d)
        for block in self.blocks:
            x = block(x)

        x = x.view(B, N, self.d)
        return self.head(x)
    
def extend_with_action_col(state: torch.Tensor, action_idx: int, void_idx: int):
    # state: (9,) flat → (3,3)
    mat = state.view(3,3)
    # new 4th column: [void; void; action]
    col = torch.tensor([void_idx, void_idx, action_idx],
                       dtype=state.dtype,
                       device=state.device).unsqueeze(1)  # (3,1)
    # concat → (3,4) → flatten (12,)
    return torch.cat([mat, col], dim=1).view(-1)


# ----------------------- training loop -------------------------------------

def noisy_sample(seq: torch.LongTensor) -> torch.LongTensor:
    """Add per‑token entropy e∈[0,1] the same way as before, now for length‑18."""
    e = torch.rand(len(seq))
    base = (e/85).unsqueeze(1).repeat(1,85)
    probs = base.clone(); rows=torch.arange(len(seq))
    probs[rows, seq] = 1.0 - e*84/85
    probs = probs / probs.sum(1,keepdim=True)
    return torch.multinomial(probs,1).squeeze(1)

def make_plan_vis(live, to_img, H, W, border_width=10):
    """
    live:    tensor of shape (1, T*H*W)
    to_img:  function mapping a length-(H*W) token vector → (h_px, w_px, 3) uint8 image
    H,W:     grid dims (e.g. 3,4)
    border_width: thickness of separator
    """
    B, NW = live.shape
    F      = H*W
    T      = NW // F

    # 1) render each of the T frames
    imgs = [ to_img(live[0, i*F:(i+1)*F]) for i in range(T) ]

    # 2) build a vertical “border” of the same height
    h_px = imgs[0].shape[0]
    border = np.full((h_px, border_width, 3), 128, dtype=np.uint8)

    # 3) interleave border between frames
    parts = []
    for img in imgs:
        if parts:
            parts.append(border)
        parts.append(img)

    # 4) hstack everything
    vis = np.hstack(parts)
    return vis


def map_transition(state: torch.Tensor, action: int) -> torch.Tensor:
    """
    state:  a (9,) LongTensor which is one of T1…T9
    action: 0=stay, 1=right, 2=up, 3=left, 4=down
    returns: new state (one of T1…T9)
    """

    # find the linear index 0..8 where the 'vertical' patch sits
    # (i.e. state[i] == vertical)
    pos = (state == vertical).nonzero(as_tuple=True)[0].item()
    r, c = divmod(pos, 3)

    if action == 1:      # move right
        c = min(c+1, 2)
    elif action == 2:    # move up
        r = max(r-1, 0)
    elif action == 3:    # move left
        c = max(c-1, 0)
    elif action == 4:    # move down
        r = min(r+1, 2)
    # action == 0 → stay in place

    return STATES[r*3 + c]

def run(max_steps=100000):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m=STModel().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    m.eval()
    
    step = load_checkpoint(m, opt, 'square_diffusion.ckpt', dev)
    # live sequence: frame0 | frame1
    B, H, W = 1, 3, 4
    F  = H*W
    T  = 4
    N  = T*F           # 36
    live = torch.empty((B, N), dtype=torch.long, device=dev)

    # 2) extend start and goal with a "stay" action in their action column 
    a = actions[1]  # zero‐th action = stay
    s0   = extend_with_action_col(T1, a, void)   # frame0
    g2   = extend_with_action_col(T6, a, void)   # frame2

    # 3) build the live tensor
    live[:, 0*F:1*F] = s0
    live[:, (T-1)*F:T*F] = g2

    preds = torch.randint(0, 85, (1, 12*T), device=dev)
    live[:,11] = preds[:,11]
    live[:,-1] = preds[:,-1]
    # initialize the middle frame randomly
    live[:, 1*F:(T-1)*F] = preds[:,1*F:(T-1)*F]
    generate = True
    
    try:
        for step in range(max_steps):
            print(f"step {step:5d}")

            if generate:
                logits = m(live)
                preds  = logits.argmax(dim=-1)
            else:
                preds = torch.randint(0, 85, (1, 12*T), device=dev)
            #live[:, 11:24] = preds[:, 11:24]
            live[:,11:(T-1)*F] = preds[:,11:(T-1)*F]
            live[:,-1] = preds[:,-1]

            if step % 5 == 0:
                
                vis = make_plan_vis(
                    live,       # your (1, T*H*W) tensor
                    to_img,     # your conversion fn
                    H=3, W=4,   # grid size
                    border_width=10
                )

                cv2.imshow('Plan: [start | predmid | goal]', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('g'):
                    generate = not generate

    finally:
        cv2.destroyAllWindows()

    

if __name__=='__main__':
    run()