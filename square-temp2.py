import torch
from torch import nn
from attention import SpatialAxialAttention, TemporalAxialAttention
from rotary_embedding_torch import RotaryEmbedding

# square_diffusion_transformer.py  (fixed greedy update)
"""
Discrete "diffusion" over a 3×3 grid using a Transformer encoder.
Now uses **greedy** argmax resampling, which converges quickly.
"""

import itertools
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# ----------------------- utilities -----------------------------------------
PATTERNS, IDX = [], {}
for combo in itertools.combinations(range(9), 3):
    v = torch.zeros(9, dtype=torch.long); v[list(combo)] = 1
    IDX[tuple(v.tolist())] = len(PATTERNS); PATTERNS.append(v)

def pt(mat):
    return IDX[tuple([e for row in mat for e in row])]

def patch(i):
    return PATTERNS[i].view(3,3)

def to_img(tokens, zoom=32):
    patches=[patch(int(t)) for t in tokens]
    rows=[torch.cat(patches[r*3:(r+1)*3],1) for r in range(3)]
    raw = torch.cat(rows,0).numpy().astype(np.uint8)
    grid = raw * 255
    img=cv2.resize(grid,(grid.shape[1]*zoom,grid.shape[0]*zoom),cv2.INTER_NEAREST)
    return cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2BGR)

# ----------------------- target --------------------------------------------
corner_TL=pt([[0,0,0],[0,1,1],[0,1,0]])
corner_TR=pt([[0,0,0],[1,1,0],[0,1,0]])
corner_BL=pt([[0,1,0],[0,1,1],[0,0,0]])
corner_BR=pt([[0,1,0],[1,1,0],[0,0,0]])
vertical =pt([[0,1,0],[0,1,0],[0,1,0]])
horizontal=pt([[0,0,0],[1,1,1],[0,0,0]])
#----------------------------------------------------------------------------
corner_TL2=pt([[1,0,0],[0,1,0],[0,0,1]])
corner_TR2=pt([[0,0,1],[0,1,0],[1,0,0]])
verticalR =pt([[0,0,1],[0,0,1],[0,0,1]])
verticalL =pt([[1,0,0],[1,0,0],[1,0,0]])
horizontalT=pt([[1,1,1],[0,0,0],[0,0,0]])
horizontalB=pt([[0,0,0],[0,0,0],[1,1,1]])

centre=vertical
T1=torch.tensor([
    corner_TL,horizontal,corner_TR,
    vertical,centre,vertical,
    corner_BL,horizontal,corner_BR],dtype=torch.long)
T2=torch.tensor([
    corner_TL2,horizontalB,corner_TR2,
    verticalR,centre,verticalL,
    corner_TR2,horizontalT,corner_TL2],dtype=torch.long)

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
    def __init__(self, d=128, heads=4, num_layers=3):
        super().__init__()
        self.d = d
        self.T = 2      # two frames
        self.H = self.W = 3

        # token & temporal embeddings
        self.token_emb = nn.Embedding(84, d)
        self.time_emb  = nn.Parameter(torch.zeros(self.T, d))

        # reserve a single pos-emb for the 9 spatial slots,
        # which we’ll reuse on both frames
        self.pos2d = nn.Parameter(torch.zeros(self.H*self.W, d))

        # build N spatio-temporal blocks
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(d, heads)
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(d, 84)

    def forward(self, seq):
        """
        seq: LongTensor of shape (B, 2*9) = concatenate(frame0, frame1)
        returns: (B, 18, 84)
        """
        B, N = seq.shape
        assert N == self.T*self.H*self.W

        # embed & add per-token positional
        tok = self.token_emb(seq)                  # (B, 18, d)
        tok = tok + self.pos2d.unsqueeze(0).repeat(B, self.T, 1).view(B, N, self.d)
        # add per-frame temporal embedding
        time_emb = self.time_emb.unsqueeze(1).repeat(1, self.H*self.W, 1)  # (2,9,d)
        time_emb = time_emb.view(1, N, self.d).repeat(B,1,1)              # (B,18,d)
        tok = tok + time_emb

        # reshape into (B, T, H, W, d)
        x = tok.view(B, self.T, self.H, self.W, self.d)

        # run our ST blocks
        for block in self.blocks:
            x = block(x)

        # flatten back to (B, 18, d) → logits
        out = x.view(B, N, self.d)
        return self.head(out)

# ----------------------- training loop -------------------------------------

def noisy_sample(seq: torch.LongTensor) -> torch.LongTensor:
    """Add per‑token entropy e∈[0,1] the same way as before, now for length‑18."""
    e = torch.rand(len(seq))
    base = (e/84).unsqueeze(1).repeat(1,84)
    probs = base.clone(); rows=torch.arange(len(seq))
    probs[rows, seq] = 1.0 - e*83/84
    probs = probs / probs.sum(1,keepdim=True)
    return torch.multinomial(probs,1).squeeze(1)

def run(max_steps=100000,lr=1e-4,seed=0):
    torch.manual_seed(seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m=STModel().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=lr)
    # live sequence: frame0 | frame1
    live = torch.cat([torch.randint(0,84,(9,)), torch.randint(0,84,(9,))]).unsqueeze(0).to(dev)
    learning = True
    generate = True
    try:
        for step in range(1, max_steps+1):
            src = torch.stack([
                noisy_sample(torch.cat([T1,T2])),
                noisy_sample(torch.cat([T2,T1]))
            ],0).to(dev)                     # (2,18)
            tgt = torch.stack([torch.cat([T1,T2]), torch.cat([T2,T1])],0).to(dev)

            logits = m(src)

            if learning:
                loss = F.cross_entropy(logits.view(-1,84), tgt.view(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                
            else:
                loss = torch.tensor(0.)

            if generate:
                with torch.no_grad():

                    # 2. use two-frame model on the whole sequence, take future frame
                    logits = m(live)                           # (1, 18, 84)
                    #probs = F.softmax(logits, -1)
                    preds = m(live).argmax(dim=-1)   # (1,18)
                    live  = preds   
                    #next = torch.multinomial(probs[0], 1).squeeze(1)[9:]  # (9,)
                    #live[:, 9:] = next


            if step % 5 == 0:
                print(f"step {step:5d} | loss {loss.item():.4f} | learning {learning}")
                vis = np.hstack([
                    to_img(live[0,:9]),
                    np.full((288,10),128,np.uint8)[:,:,None].repeat(3,2),
                    to_img(live[0,9:])
                ])
                cv2.imshow('grid (L)  |  target (R)', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:        # ESC
                    break
                elif key == ord(' '):  # reset
                    print("Reset tokens")
                    live = torch.cat([torch.randint(0,84,(9,)), torch.randint(0,84,(9,))]).unsqueeze(0).to(dev)
                elif key == ord('m'):  # toggle learning
                    learning = not learning
                elif key == ord('g'):
                    generate = not generate

            if step % 100 == 0:
                print('auto regressive update')
                live = torch.cat([live[0,9:], torch.randint(0,84,(9,)).to(dev)]).unsqueeze(0)


    finally:
        cv2.destroyAllWindows()

if __name__=='__main__':
    run()