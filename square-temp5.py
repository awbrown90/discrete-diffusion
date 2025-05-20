import torch
from torch import nn
import random
from attention import SpatialAxialAttention, TemporalAxialAttention
from rotary_embedding_torch import RotaryEmbedding
import copy

# square_diffusion_transformer.py  (fixed greedy update)
"""
Discrete "diffusion" over a 3×3 grid using a Transformer encoder.
Now uses **greedy** argmax resampling, which converges quickly.
"""

import itertools
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

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
horizontal = pt([[0,0,0],[1,1,1],[0,0,0]])
vertical   = pt([[0,1,0],[0,1,0],[0,1,0]])
laction = pt([[1,0,0],[0,1,0],[0,0,1]])
raction = pt([[0,0,1],[0,1,0],[1,0,0]])
saction = pt([[0,0,0],[0,0,0],[1,1,1]])

actions= [laction,saction,raction]

T1=torch.tensor([
    vertical,void,void,
    vertical,void,void,
    horizontal,horizontal,horizontal],dtype=torch.long)
T2=torch.tensor([
    void,vertical,void,
    void,vertical,void,
    horizontal,horizontal,horizontal],dtype=torch.long)
T3=torch.tensor([
    void,void,vertical,
    void,void,vertical,
    horizontal,horizontal,horizontal],dtype=torch.long)

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
        self.H, self.W = 3, 4

        # token & temporal embeddings
        self.token_emb = nn.Embedding(85, d)
        self.time_emb  = nn.Parameter(torch.zeros(self.T, d))

        # reserve a single pos-emb for the 9 spatial slots,
        # which we’ll reuse on both frames
        self.pos2d = nn.Parameter(torch.zeros(self.H*self.W, d))

        # build N spatio-temporal blocks
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(d, heads)
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(d, 85)

    def forward(self, seq):
        """
        seq: LongTensor of shape (B, 2*9) = concatenate(frame0, frame1)
        returns: (B, 18, 85)
        """
        B, N = seq.shape
        assert N == self.T*self.H*self.W

        # embed & add per-token positional
        tok = self.token_emb(seq)                  # (B, 24, d)
        tok = tok + self.pos2d.unsqueeze(0).repeat(B, self.T, 1).view(B, N, self.d)
        # add per-frame temporal embedding
        time_emb = self.time_emb.unsqueeze(1).repeat(1, self.H*self.W, 1)  # (2,12,d)
        time_emb = time_emb.view(1, N, self.d).repeat(B,1,1)              # (B,24,d)
        tok = tok + time_emb

        # reshape into (B, T, H, W, d)
        x = tok.view(B, self.T, self.H, self.W, self.d)

        # run our ST blocks
        for block in self.blocks:
            x = block(x)

        # flatten back to (B, 24, d) → logits
        out = x.view(B, N, self.d)
        return self.head(out)
    
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

def map_transition(state, action):
    if action == 1:
        return state
    if action == 0:
        return T1 if torch.equal(state, T2) or torch.equal(state, T1) else T2
    if action == 2:
        return T3 if torch.equal(state, T2) or torch.equal(state, T3) else T2



def run(max_steps=100000,lr=1e-4,seed=0):
    torch.manual_seed(seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m=STModel().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=lr)
    # live sequence: frame0 | frame1
    live = torch.cat([torch.randint(0,85,(12,)), torch.randint(0,85,(12,))]).unsqueeze(0).to(dev)

    buffer = ReplayBuffer(observation_shape=torch.randint(0,85,(12,)).shape, capacity=50000, device=dev)
    current_state = T1

    learning = True
    generate = True
    interactive = False

    try:
        for step in range(1, max_steps+1):

            # take a random action
            action = random.randint(0,len(actions)-1)
            new_state = map_transition(current_state,action)

            s0_ext = extend_with_action_col(current_state, actions[action], void)

            buffer.add(s0_ext)

            current_state = new_state

            if len(buffer) >= 16 and learning:
                examples = buffer.sample(8,2)

                src_batch = []
                tgt_batch = []
                
                for example in examples:
                    src_batch.append(noisy_sample(example))
                    tgt_batch.append(example)

                src = torch.stack(src_batch, dim=0).to(dev)
                tgt = torch.stack(tgt_batch, dim=0).to(dev)

                logits = m(src)

                loss = F.cross_entropy(logits.view(-1,85), tgt.view(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                
            else:
                loss = torch.tensor(0.)

            if generate:
                with torch.no_grad():
                    # forward once, get 24-token predictions
                    logits = m(live)                   # [1,24,85]
                    preds  = logits.argmax(dim=-1)     # [1,24]
                    live[:, :11] = preds[:, :11]  
                    live[:, 12:] = preds[:, 12:]


            if step % 5 == 0:
                print(f"step {step:5d} | loss {loss.item():.4f} | learning {learning} | interactive {interactive}")
                vis = np.hstack([
                    to_img(live[0,:12]),
                    np.full((288,10),128,np.uint8)[:,:,None].repeat(3,2),
                    to_img(live[0,12:])
                ])
                cv2.imshow('grid (L)  |  target (R)', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:        # ESC
                    break
                elif key == ord(' '):  # reset
                    print("Reset tokens")
                    live = torch.cat([torch.randint(0,85,(12,)), torch.randint(0,85,(12,))]).unsqueeze(0).to(dev)
                elif key == ord('m'):  # toggle learning
                    learning = not learning
                elif key == ord('g'):
                    generate = not generate
                elif key == ord('k'):
                    interactive = not interactive

                if interactive:
                    if key == 81:  # Left arrow
                        chosen_action = 0
                    elif key == 84:  # Down arrow
                        chosen_action = 1
                    elif key == 83:  # Right arrow
                        chosen_action = 2
                    else:
                        chosen_action = None
                    if chosen_action is not None:
                        print('auto regressive update')
                        live = torch.cat([live[0,12:], torch.randint(0,85,(12,)).to(dev)]).unsqueeze(0)
                        live[0,11] = actions[chosen_action]

            if not interactive and step % 10 == 0:
                chosen_action = torch.randint(0, len(actions), (1,)).item()
                print('auto regressive update')
                live = torch.cat([live[0,12:], torch.randint(0,85,(12,)).to(dev)]).unsqueeze(0)
                live[0,11] = actions[chosen_action]



    finally:
        cv2.destroyAllWindows()

if __name__=='__main__':
    run()