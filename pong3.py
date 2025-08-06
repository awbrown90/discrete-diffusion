import torch
from torch import nn
import torch.nn.functional as F
import random
import itertools
import numpy as np
import cv2
from attention import SpatialAxialAttention, TemporalAxialAttention, TimestepEmbedder
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.tensorboard import SummaryWriter

# ------------------ CONFIGURABLE GRID DIMENSIONS ------------------
GRID_H = 5  # number of rows
GRID_W = 7  # number of columns
# -------------------------------------------------------------------
# ----------------------- patch-content patterns --------------------
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

def patch(i):
    return PATTERNS[i].view(3, 3)

def to_img(tokens, zoom=32):
    """Render GRID_H x (GRID_W+1) patches as an image."""
    ext_W = GRID_W + 1
    patches = [patch(int(t)) for t in tokens]
    rows = [torch.cat(patches[r*ext_W:(r+1)*ext_W], dim=1) for r in range(GRID_H)]
    raw = torch.cat(rows, dim=0).numpy().astype(np.uint8)
    grid = raw * 255
    img = cv2.resize(grid, (grid.shape[1]*zoom, grid.shape[0]*zoom), cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# ---------------- checkpoint save/load ----------------
def save_checkpoint(model, optimizer, step, filename='ckpt.pth'):
    torch.save({'step': step,
                'model': model.state_dict(),
                'opt': optimizer.state_dict()},
               filename)
    print(f"Saved checkpoint at step {step} to {filename}")

def load_checkpoint(model, optimizer, filename='ckpt.pth', device='cpu'):
    try:
        ckpt = torch.load(filename, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        return ckpt.get('step', None)
    except FileNotFoundError:
        return None
    
class PongEnv:
    def __init__(self,
                 device,
                 grid_h=5,
                 grid_w=7,
                 patch_size=20,
                 ball_size=5,
                 random_miss=True):
        # dynamics state
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

        # rendering params
        self.patch_size = patch_size
        self.ball_size  = ball_size
        self.height     = grid_h * patch_size
        self.width      = grid_w * patch_size

        self.device     = device
        self.random_miss = random_miss

    def render(self):
        # start with a black canvas
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # draw paddle: leftmost column, full patch_height × 5px wide
        y0 = self.r_paddle * self.patch_size
        img[y0:y0 + self.patch_size,
            0:5] = (0, 255, 0)   # green in RGB

        # draw ball if it’s in “active” state
        if self.ball_idx != 4:
            cell_x = self.ball_c * self.patch_size
            cell_y = self.ball_r * self.patch_size

            # choose corner based on ball_idx:
            # 0 = up-right, 1 = up-left, 2 = down-right, 3 = down-left
            if self.ball_idx == 0:
                px = cell_x + (self.patch_size - self.ball_size)
                py = cell_y
            elif self.ball_idx == 1:
                px = cell_x
                py = cell_y
            elif self.ball_idx == 2:
                px = cell_x + (self.patch_size - self.ball_size)
                py = cell_y + (self.patch_size - self.ball_size)
            elif self.ball_idx == 3:
                px = cell_x
                py = cell_y + (self.patch_size - self.ball_size)

            img[py:py + self.ball_size,
                px:px + self.ball_size] = (0, 0, 255)  # blue in RGB

        # convert to torch tensor, channels-first, on the correct device
        tensor = torch.from_numpy(img)                    # H×W×3 uint8
        tensor = tensor.permute(2, 0, 1).to(self.device)  # 3×H×W
        return tensor

    def getState(self):
        # exactly the same as before, except return the rendered image
        return self.render()
    
    def reset_misses(self):
        self.miss_counter = 0

    def get_misses(self):
        return self.miss_counter

    def step(self, action: int):
        # —— identical dynamics to your original code ——

        # apply paddle move
        if action == 0:          # up
            self.r_paddle = max(self.r_paddle - 1, 0)
        elif action == 1:        # down
            self.r_paddle = min(self.r_paddle + 1, self.grid_h - 1)

        # update ball position & detect misses
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

        # set ball_idx based on direction (if not in miss-reset)
        if not self.reset_flag:
            if   self.ball_d_c == 1 and self.ball_d_r == -1: self.ball_idx = 0
            elif self.ball_d_c == -1 and self.ball_d_r == -1: self.ball_idx = 1
            elif self.ball_d_c == 1 and self.ball_d_r == 1: self.ball_idx = 2
            elif self.ball_d_c == -1 and self.ball_d_r == 1: self.ball_idx = 3
        else:
            # on miss, clear the reset_flag for the next frame
            self.reset_flag = False

        # instead of returning a token vector, render and return the RGB image
        return self.render()

# ---------------------- replay buffer --------------------
class ReplayBuffer:
    def __init__(self, observation_shape, capacity, device):
        self.device = device
        self.capacity = int(capacity)
        self.observations = np.empty((self.capacity, *observation_shape),
                                     dtype=np.float32)
        self.bufferIndex = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.bufferIndex

    def add(self, observation):
        self.observations[self.bufferIndex] = observation
        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or (self.bufferIndex == 0)

    def sample(self, batchSize, sequenceSize):
        lastIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastIndex > batchSize), "Not enough data in buffer"
        idx = np.random.randint(0,
                                self.capacity if self.full else lastIndex,
                                batchSize).reshape(-1, 1)
        seq = np.arange(sequenceSize).reshape(1, -1)
        idx = (idx + seq) % self.capacity
        obs = torch.as_tensor(self.observations[idx],
                              dtype=torch.long,
                              device=self.device)
        return obs.view(batchSize, -1)

# --------------------- define tokens --------------------
zero_idx = IDX[tuple(zero.tolist())]

# vertical bar (3×3) – center column filled
vert = torch.zeros((3,3), dtype=torch.long)
vert[:,1] = 1
vertical_idx = IDX[tuple(vert.flatten().tolist())]

# horizontal bar (3×3) – center row filled
uhoriz = torch.zeros((3,3), dtype=torch.long)
uhoriz[0,:] = 1
uphorizontal_idx = IDX[tuple(uhoriz.flatten().tolist())]

ru_horiz = torch.zeros((3,3), dtype=torch.long)
ru_horiz[0,2] = 1
ru_horiz[0,1] = 1
ru_horiz[1,2] = 1
ru_horizontal_idx = IDX[tuple(ru_horiz.flatten().tolist())]

lu_horiz = torch.zeros((3,3), dtype=torch.long)
lu_horiz[0,0] = 1
lu_horiz[0,1] = 1
lu_horiz[1,0] = 1
lu_horizontal_idx = IDX[tuple(lu_horiz.flatten().tolist())]

rl_horiz = torch.zeros((3,3), dtype=torch.long)
rl_horiz[2,2] = 1
rl_horiz[2,1] = 1
rl_horiz[1,2] = 1
rl_horizontal_idx = IDX[tuple(rl_horiz.flatten().tolist())]

ll_horiz = torch.zeros((3,3), dtype=torch.long)
ll_horiz[2,0] = 1
ll_horiz[2,1] = 1
ll_horiz[1,0] = 1
ll_horizontal_idx = IDX[tuple(ll_horiz.flatten().tolist())]

# horizontal bar (3×3) – center row filled
horiz = torch.zeros((3,3), dtype=torch.long)
horiz[0,0] = 1
horiz[1,1] = 1
horiz[2,2] = 1
horizontal_idx = IDX[tuple(horiz.flatten().tolist())]

# horizontal bar (3×3) – center row filled
dhoriz = torch.zeros((3,3), dtype=torch.long)
dhoriz[2,:] = 1
dnhorizontal_idx = IDX[tuple(dhoriz.flatten().tolist())]

ball_idx = 0
balls = [ru_horizontal_idx, lu_horizontal_idx, rl_horizontal_idx, ll_horizontal_idx]

# action patches

uaction = IDX[tuple(torch.tensor([[0,1,0],
                                  [1,0,1],
                                  [0,0,0]],
                                 dtype=torch.long).flatten().tolist())]
daction = IDX[tuple(torch.tensor([[0,0,0],
                                  [1,0,1],
                                  [0,1,0]],
                                 dtype=torch.long).flatten().tolist())]
saction = IDX[tuple(torch.tensor([[0,0,0],
                                  [1,1,1],
                                  [0,0,0]],
                                 dtype=torch.long).flatten().tolist())]
actions = [uaction, daction, saction]

# ------------- extend with action column ----------------
def extend_with_action_col(state: torch.Tensor,
                           action_idx: int,
                           void_idx: int) -> torch.Tensor:
    mat = state.view(GRID_H, GRID_W)
    col = torch.full((GRID_H,), void_idx,
                     dtype=state.dtype, device=state.device)
    col[-1] = action_idx
    ext = torch.cat([mat, col.unsqueeze(1)], dim=1)
    return ext.view(-1)

# --------------------- model blocks ----------------------
class SpatioTemporalBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        dim_head = d // heads
        self.spatial_rotary = RotaryEmbedding(dim=dim_head//2, freqs_for="pixel")
        self.temporal_rotary = RotaryEmbedding(dim=dim_head, freqs_for="lang")
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
    def __init__(self, d=512, heads=4, num_layers=3,
                 vocab_size=len(PATTERNS)):
        super().__init__()
        self.H, self.W = GRID_H, GRID_W + 1
        self.d = d
        self.token_emb    = nn.Embedding(vocab_size, d)
        self.time_embedder= TimestepEmbedder(hidden_size=d,
                                             frequency_embedding_size=d)
        self.pos2d        = nn.Parameter(torch.zeros(self.H*self.W, d))
        self.blocks       = nn.ModuleList(
            [SpatioTemporalBlock(d, heads) for _ in range(num_layers)]
        )
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
        te = te.unsqueeze(1).expand(T, self.H*self.W, self.d) \
               .reshape(1, N, self.d).repeat(B,1,1)
        x = x + te

        x = x.view(B, T, self.H, self.W, self.d)
        for blk in self.blocks:
            x = blk(x)
        x = x.view(B, N, self.d)
        return self.head(x)

# ---------------- sampling util -----------------------
def noisy_sample(seq: torch.LongTensor, frame_size: int) -> torch.LongTensor:
    N = seq.numel()
    T = N // frame_size
    frames = seq.view(T, frame_size).clone()
    e     = torch.rand(T, device=seq.device)
    mask  = torch.rand(T, frame_size, device=seq.device) < e.unsqueeze(1)
    rand  = torch.randint(0, len(PATTERNS), (T, frame_size), device=seq.device)
    frames[mask] = rand[mask]
    return frames.view(-1)

# --------------------- main loop ----------------------
def run(max_steps=50000, lr=1e-4, seed=0):
    torch.manual_seed(seed)
    dev   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STModel().to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    _     = load_checkpoint(model, opt, 'pong2_ckpt.pth', dev)

    writer = SummaryWriter(log_dir='runs/pong_recon')

    extW  = GRID_H * (GRID_W + 1)
    # start live with two random frames
    live  = torch.cat([
        torch.randint(0, len(PATTERNS), (extW,)),
        torch.randint(0, len(PATTERNS), (extW,))
    ]).unsqueeze(0).to(dev)

    buffer = ReplayBuffer(observation_shape=(extW,), capacity=1_000, device=dev)

    random_mode = False

    env_train = PongEnv(dev, random_miss=random_mode)
    env_test = PongEnv(dev, random_miss=random_mode)

    autoregressive = False

    learning, generate, interactive = True, True, False

    # ——— NEW: prepare to collect logs ———
    log_entries = [] # will hold lines like "step,misses\n"

    try:
        for step in range(1, max_steps+1):
            # ——— NEW: every 250 steps, log + reset ———
            if step % 200 == 0:
                log_entries.append(f"{step},{env_test.get_misses()}\n")
                #env_test.reset_misses()
                
            action = random.randint(0, len(actions)-1)

            current_state = env_train.getState()
            
            # 1) map transition → new_state
            #new_state = map_transition(current_state, action)
            new_state = env_train.step(action)

            # 2) extend with action col
            s0_ext = extend_with_action_col(current_state,
                                            actions[action],
                                            zero_idx)
            
            # 3) add to buffer
            buffer.add(s0_ext.cpu().numpy())

            # 4) update current_state
            current_state = new_state

            # ——— train once we have enough data ———
            if len(buffer) >= 64 and learning:
                n = random.randint(2,8)
                ex = buffer.sample(32, n)
                src= torch.stack([noisy_sample(e, frame_size=extW)
                                  for e in ex], dim=0).to(dev)
                tgt= torch.stack([e for e in ex], dim=0).to(dev)
                logits = model(src)
                loss   = F.cross_entropy(
                            logits.view(-1, len(PATTERNS)),
                            tgt.view(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                writer.add_scalar('Train/CrossEntropy', loss.item(), step)
            else:
                loss = torch.tensor(0., device=dev)

            # ——— autoregressive generate ———
            if generate:
                with torch.no_grad():
                    logits = model(live)
                    preds  = logits.argmax(dim=-1)
                    
                    last_idx = (GRID_H-1)*(GRID_W+1) + GRID_W
                    live[:, extW:] = preds[:, extW:]
                    live[0, last_idx] = preds[0, last_idx]
                    
            # ——— display ———
            if step % 2 == 0:
                print(f"step {step} | loss {loss.item():.4f} | "
                      f"learning={learning} | interactive={interactive}")
                left  = to_img(live[0, :extW])
                right = to_img(live[0, extW:])
                sep   = np.full((left.shape[0], 10, 3),
                                128, dtype=np.uint8)
                vis   = np.hstack([left, sep, right])
                cv2.imshow('grid(L) | target(R)', vis)
                key = cv2.waitKey(1) & 0xFF
                if   key == 27: break
                elif key == ord(' '):
                    live = torch.cat([
                        torch.randint(0, len(PATTERNS), (extW,)),
                        torch.randint(0, len(PATTERNS), (extW,))
                    ]).unsqueeze(0).to(dev)
                elif key == ord('m'):
                    learning = not learning
                elif key == ord('g'):
                    generate = not generate
                elif key == ord('k'):
                    interactive = not interactive
                elif key == ord('s'):
                    save_checkpoint(model, opt, step, 'ckpt.pth')
                elif interactive and key in [82,84,83]:
                    # up/down arrow → pick action manually
                    if key==82:
                        chosen=0
                    elif key==84:
                        chosen=1
                    else:
                        chosen=2
                    
                    live = torch.cat([
                        live[0, extW:],
                        torch.randint(0, len(PATTERNS), (extW,)).to(dev)
                    ]).unsqueeze(0)
                    last_idx = (GRID_H-1)*(GRID_W+1) + GRID_W
                    live[0, last_idx] = actions[chosen]

            # ——— automated stepping if not interactive ———
            if not interactive and step % 10 == 0:
                last_idx = (GRID_H-1)*(GRID_W+1) + GRID_W
                if autoregressive:
                    live = torch.cat([
                        live[0, extW:],
                        torch.randint(0, len(PATTERNS), (extW,)).to(dev)
                    ]).unsqueeze(0)
                    live[0, last_idx] = torch.randint(0, len(PATTERNS), (1,)).to(dev)
                else:
                    action = live[0, last_idx]
                    #if action == actions[0]:
                    #    action=0
                    #elif action == actions[1]:
                    #    action=1
                    #elif action == actions[2]:
                    #    action=2
                    #else:
                    action = random.randint(0, len(actions)-1)
                    new_state = env_test.step(action)
                    state_ext = extend_with_action_col(new_state,
                                            actions[action],
                                            zero_idx)
                    live = torch.cat([
                        state_ext,
                        torch.randint(0, len(PATTERNS), (extW,)).to(dev)
                    ]).unsqueeze(0)
                    live[0, last_idx] = torch.randint(0, len(PATTERNS), (1,)).to(dev)

    finally:
        cv2.destroyAllWindows()
        # ——— NEW: write out log file at the very end ———
        save_log = False
        save_model = False
        if save_model:
            save_checkpoint(model, opt, step, 'k.pth')
        if save_log:
            if random_mode:
                file = 'miss_log_rand.txt'
            else:
                file = 'miss_log.txt'
            with open(file, 'w') as f:
                f.write("step,misses\n")
                f.writelines(log_entries)
            print(f"Wrote {len(log_entries)} entries to {file}")

if __name__ == '__main__':
    run()
