from __future__ import annotations
import itertools
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# ----------------------- utilities -----------------------------------------
PATTERNS, IDX = [], {}
for combo in itertools.combinations(range(9), 3):
    v = torch.zeros(9, dtype=torch.long)
    v[list(combo)] = 1
    IDX[tuple(v.tolist())] = len(PATTERNS)
    PATTERNS.append(v)

VOID = len(PATTERNS)  # index for absorbing "void" token

def pt(mat):
    return IDX[tuple(e for row in mat for e in row)]

def patch(i):
    # return zero patch for VOID token, else the learned pattern
    if i == VOID:
        return torch.zeros(3, 3, dtype=torch.long)
    return PATTERNS[i].view(3, 3)

def to_img(tokens, zoom=32):
    patches = [patch(int(t)) for t in tokens]
    rows = [torch.cat(patches[r*3:(r+1)*3], 1) for r in range(3)]
    raw = torch.cat(rows, 0).numpy().astype(np.uint8)
    grid = raw * 255
    img = cv2.resize(grid, (grid.shape[1]*zoom, grid.shape[0]*zoom), cv2.INTER_NEAREST)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

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
'''
target_1 = torch.tensor([
    horizontal, horizontal, vertical,
    horizontal, horizontal, vertical,
    horizontal, horizontal, vertical
], dtype=torch.long)

target_2 = torch.tensor([
    vertical, horizontal, horizontal,
    vertical, horizontal, horizontal,
    vertical, horizontal, horizontal
], dtype=torch.long)
'''
target_1=torch.tensor([
    corner_TL,horizontal,corner_TR,
    vertical,centre,vertical,
    corner_BL,horizontal,corner_BR],dtype=torch.long)
target_2=torch.tensor([
    corner_TL2,horizontalB,corner_TR2,
    verticalR,centre,verticalL,
    corner_TR2,horizontalT,corner_TL2],dtype=torch.long)

# ----------------------- model ---------------------------------------------
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d = 128
        self.emb  = nn.Embedding(VOID+1, d)
        self.pos  = nn.Parameter(torch.zeros(9, d))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer      = nn.TransformerEncoderLayer(d, 8, batch_first=True)
        self.enc  = nn.TransformerEncoder(layer, num_layers=6)
        self.head = nn.Linear(d, VOID)

    def forward(self, x):
        tok = self.emb(x) + self.pos
        return self.head(self.enc(tok))  # (B, 9, VOID)

# ----------------------- diffusion utilities ------------------------------
def noisy_sample(target: torch.Tensor, full_void_p: float = 0.1, steps: int = 20) -> torch.Tensor:
    """
    Discrete absorbing diffusion with occasional full void and discrete noise levels.
    full_void_p: probability to see a fully voided input.
    steps: number of discrete noise levels (0/steps ... steps/steps).
    """
    # occasionally present fully-void input
    if torch.rand((), device=target.device) < full_void_p:
        return torch.full_like(target, VOID, device=target.device)
    # pick discrete noise fraction including 1.0 exactly
    t = torch.randint(0, steps+1, (1,), device=target.device).float() / steps
    mask = torch.rand(target.shape, device=target.device) > t
    void = torch.full_like(target, VOID, device=target.device)
    return torch.where(mask, target, void)

# ----------------------- training loop -------------------------------------
def run(max_steps=100000, lr=1e-4, seed=0):
    torch.manual_seed(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m   = Model().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)

    tokens = torch.full((1, 9), VOID, dtype=torch.long, device=dev)
    learning = True
    generate = True
    t1, t2 = target_1.to(dev), target_2.to(dev)

    try:
        for step in range(1, max_steps+1):
            # forward diffusion (noising) from both targets
            src0 = noisy_sample(t1).unsqueeze(0)
            src1 = noisy_sample(t2).unsqueeze(0)
            src  = torch.cat([src0, src1], dim=0)
            tgt  = torch.cat([t1.unsqueeze(0), t2.unsqueeze(0)], dim=0)

            # train on reconstructing exact targets
            logits = m(src)  # (2,9,VOID)
            if learning:
                loss = F.cross_entropy(logits.view(-1, VOID), tgt.view(-1))
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                loss = torch.tensor(0., device=dev)

            # reverse diffusion: denoise one void token per iteration
            if generate and (tokens == VOID).any():
                with torch.no_grad():
                    out_logits = m(tokens)              # (1,9,VOID)
                    vals, best = out_logits.max(dim=-1)  # (1,9)
                    mask_void = tokens == VOID
                    vals[~mask_void] = float('-inf')
                    idx = vals.argmax().item()
                    tokens[0, idx] = best[0, idx]

            if step % 5 == 0:
                print(f"step {step:5d} | loss {loss.item():.4f} | learning {learning}")

            # visualize: left evolving, right current target
            vis = np.hstack([
                to_img(tokens.squeeze(0).cpu()),
                np.full((288,10), 128, np.uint8)[:,:,None].repeat(3,2),
                to_img(t1.cpu() if generate else tokens.cpu().squeeze(0))
            ])
            cv2.imshow('grid (L)  |  target (R)', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:        # ESC
                break
            elif key == ord(' '):  # reset to full void
                tokens.fill_(VOID)
            elif key == ord('m'):  # toggle learning
                learning = not learning
            elif key == ord('g'):  # toggle generation
                generate = not generate
            elif key == ord('t'):  # switch target
                t1, t2 = t2, t1
    finally:
        cv2.destroyAllWindows()

if __name__=='__main__':
    run()
