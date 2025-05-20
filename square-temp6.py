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

def run(max_steps=100000,lr=1e-4,seed=0):
    torch.manual_seed(seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m=STModel().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=lr)
    step = load_checkpoint(m, opt, 'square_diffusion.ckpt', dev)
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

            T=5
            if len(buffer) >= 8*T and learning:
                examples = buffer.sample(8,T)

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
                    #live[:, :11] = preds[:, :11]  
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
                elif key == ord('s'):
                    save_checkpoint(m, opt, step, 'square_diffusion.ckpt')

                if interactive:
                    if key == 81:  # Left arrow
                        chosen_action = 3
                    elif key == 82:  # Up arrow
                        chosen_action = 2
                    elif key == 83:  # Right arrow
                        chosen_action = 1
                    elif key == 84:  # Down arrow
                        chosen_action = 4
                    
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