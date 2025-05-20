# square_diffusion_transformer.py  (fixed greedy update)
"""
Discrete "diffusion" over a 3×3 grid using a Transformer encoder.
Now uses **greedy** argmax resampling, which converges quickly.
"""
from __future__ import annotations
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


TARGET=torch.tensor([
    horizontal,horizontal,vertical,
    horizontal,horizontal,vertical,
    horizontal,horizontal,vertical],dtype=torch.long)
TARGET2=torch.tensor([
    vertical,horizontal,horizontal,
    vertical,horizontal,horizontal,
    vertical,horizontal,horizontal],dtype=torch.long)
'''
TARGET=torch.tensor([
    corner_TL,horizontal,corner_TR,
    vertical,centre,vertical,
    corner_BL,horizontal,corner_BR],dtype=torch.long)
TARGET2=torch.tensor([
    corner_TL2,horizontalB,corner_TR2,
    verticalR,centre,verticalL,
    corner_TR2,horizontalT,corner_TL2],dtype=torch.long)
'''


# ----------------------- model ---------------------------------------------
from rotary_embedding_torch import RotaryEmbedding
from attention import SpatialAxialAttention

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # hidden dimension and attention heads
        self.d = 128
        heads = 4
        self.dim_head = self.d // heads

        # token embedding + 1D positional encoding
        self.emb = nn.Embedding(84, self.d)
        self.pos = nn.Parameter(torch.zeros(9, self.d))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # rotary embeddings for a 3×3 grid axial attention
        # use half the head dim so that 2D axial rotary (2 axes) fits into head_dim
        self.spatial_rotary_emb = RotaryEmbedding(
            dim=self.dim_head // 2,
            freqs_for="pixel",
            max_freq=256
        )

        # single Spatial Axial Attention block expects a T dimension
        self.spatial = SpatialAxialAttention(
            dim=self.d,
            heads=heads,
            dim_head=self.dim_head,
            rotary_emb=self.spatial_rotary_emb,
        )

        # output head projects back to token logits
        self.head = nn.Linear(self.d, 84)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        x: shape (B, 9) LongTensor of token indices
        returns: (B, 9, 84) logits for each of the 84 tokens per position
        """
        B, N = x.shape
        # embed tokens + add 1D positional embedding
        tok = self.emb(x) + self.pos        # (B, 9, d)

        # reshape into 2D grid and add a time/frame dim of 1
        tok2d = tok.view(B, 3, 3, self.d)   # (B, H=3, W=3, d)
        tok3d = tok2d.unsqueeze(1)         # (B, T=1, H, W, d)

        # apply spatial axial attention
        out3d = self.spatial(tok3d)        # (B, 1, 3, 3, d)

        # collapse T back to singleton for sequence
        out2d = out3d.squeeze(1)           # (B, 3, 3, d)

        # back to sequence of length 9
        out1d = out2d.reshape(B, N, self.d)   # (B, 9, d)

        # final projection to logits
        return self.head(out1d)            # (B, 9, 84)
    
def noisy_sample(target):
    e = torch.rand(9)                # entropy per position   ∈ [0,1]
    #import pdb
    #pdb.set_trace()
    e = torch.full_like(e, e[0])
    base = (e / 84).unsqueeze(1).repeat(1, 84)   # prob for any incorrect token
    probs = base.clone()
    rows  = torch.arange(9)
    probs[rows, target] = 1.0 - e * 83 / 84       # correct-token prob
    samples = torch.multinomial(probs, 1).squeeze(1)
    return samples          # shape (9,)


# ----------------------- training loop -------------------------------------

def run(max_steps=100000,lr=1e-4,seed=0):
    torch.manual_seed(seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m=Model().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=lr)
    tokens=torch.randint(0,84,(1,9),device=dev)
    learning = True
    generate = True
    target=TARGET
    try:
        for step in range(1, max_steps+1):
            src0 = noisy_sample(TARGET).unsqueeze(0)
            src1 = noisy_sample(TARGET2).unsqueeze(0)
            src   = torch.cat([src0, src1], dim=0).to(dev)   # (2,9)
            tgt   = torch.cat([TARGET.unsqueeze(0), TARGET2.unsqueeze(0)], 0).to(dev)  # (2,9)

            logits = m(src)   

            if learning:
                loss = F.cross_entropy(logits.view(-1,84), tgt.view(-1))
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                loss = torch.tensor(0.)

            if generate:
                with torch.no_grad():
                    live_logits = m(tokens)                       # (1, 9, 84)
                    temp = max(0.5, 1.0 * (1 - step / max_steps))
                    probs = F.softmax(live_logits / temp, dim=-1)
                    tokens = torch.multinomial(probs[0], 1).squeeze(1).unsqueeze(0)

            if step % 5 == 0:
                print(f"step {step:5d} | loss {loss.item():.4f} | learning {learning}")
                vis = np.hstack([
                    to_img(tokens.squeeze(0)),
                    np.full((288,10),128,np.uint8)[:,:,None].repeat(3,2),
                    to_img(target)
                ])
                cv2.imshow('grid (L)  |  target (R)', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:        # ESC
                    break
                elif key == ord(' '):  # reset
                    print("Reset tokens")
                    tokens = torch.randint(0,84,(1,9),device=dev)
                elif key == ord('m'):  # toggle learning
                    learning = not learning
                elif key == ord('g'):
                    generate = not generate
                elif key == ord('t'):
                    if target is TARGET:
                        target=TARGET2
                    else:
                        target=TARGET
    finally:
        cv2.destroyAllWindows()

if __name__=='__main__':
    run()
