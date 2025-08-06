import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import matplotlib.pyplot as plt

# ------------------
# 1) PongEnv with one-frame miss logic
# ------------------
class PongEnv:
    def __init__(self, grid_h=5, grid_w=7, patch_size=20, ball_size=5):
        self.grid_h, self.grid_w = grid_h, grid_w
        self.patch_size, self.ball_size = patch_size, ball_size
        self.reset()

    def reset(self):
        self.r_paddle   = 0
        self.ball_r     = 0
        self.ball_c     = 1
        self.ball_d_c   = 1
        self.ball_d_r   = 1
        self.ball_idx   = 0    # 0..3 corners, 4 means "miss"
        self.reset_flag = False

    def render(self):
        H = self.grid_h * self.patch_size
        W = self.grid_w * self.patch_size
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # paddle (green)
        y0 = self.r_paddle * self.patch_size
        img[y0:y0+self.patch_size, 0:5] = (0,255,0)
        # ball (red) if not in miss state
        if self.ball_idx != 4:
            cx = self.ball_c * self.patch_size
            cy = self.ball_r * self.patch_size
            offsets = {
                0: (self.patch_size-self.ball_size, 0),
                1: (0, 0),
                2: (self.patch_size-self.ball_size, self.patch_size-self.ball_size),
                3: (0, self.patch_size-self.ball_size),
            }
            dx, dy = offsets[self.ball_idx]
            img[cy+dy:cy+dy+self.ball_size,
                cx+dx:cx+dx+self.ball_size] = (0,0,255)
        return torch.from_numpy(img).permute(2,0,1)  # CHW uint8

    def step(self, action: int):
        # move paddle
        if action == 0:
            self.r_paddle = max(self.r_paddle - 1, 0)
        else:
            self.r_paddle = min(self.r_paddle + 1, self.grid_h - 1)
        # update ball
        self.ball_c += self.ball_d_c
        self.ball_r += self.ball_d_r
        # left/right bounce + miss detection
        if self.ball_c >= self.grid_w - 1:
            self.ball_d_c = -1
        elif self.ball_c <= 1:
            if self.ball_idx != 4 and self.r_paddle != self.ball_r:
                self.reset_flag = True
                self.ball_idx   = 4
            self.ball_d_c = 1
        # top/bottom bounce
        if self.ball_r >= self.grid_h - 1:
            self.ball_d_r = -1
        elif self.ball_r <= 0:
            self.ball_d_r = 1
        # handle one-frame miss
        if self.reset_flag:
            self.reset_flag = False
        else:
            mapping = {(1,-1):0, (-1,-1):1, (1,1):2, (-1,1):3}
            self.ball_idx = mapping[(self.ball_d_c, self.ball_d_r)]
        return self.render()

# ------------------
# 2) ReplayBuffer + Dataset
# ------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.cap = capacity

    def add(self, frame: torch.Tensor):
        if len(self.buffer) >= self.cap:
            self.buffer.pop(0)
        self.buffer.append(frame.cpu())

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

class PongDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, steps_per_epoch: int):
        super().__init__()
        self.buffer = buffer
        self.steps  = steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps):
            batch = self.buffer.sample(32)
            yield torch.stack(batch, dim=0)  # [32,3,H,W] uint8

# ------------------
# 3) Training + Live Display via Matplotlib (with VQ-ViT)
# ------------------
def train_and_display(device, vis_interval=50):
    # 1) fill replay buffer
    buf = ReplayBuffer(capacity=5000)
    data_env = PongEnv()
    for _ in range(1000):
        frame = data_env.step(random.choice([0,1]))
        buf.add(frame)

    # 2) DataLoader
    loader = DataLoader(
        PongDataset(buf, steps_per_epoch=2000),
        batch_size=None,
        num_workers=0
    )

    # 3) lazy-import & build VQ-ViT AE
    from autoencoder_kl import AutoencoderKL
    model = AutoencoderKL(
        latent_dim     = 256,
        input_height   = 100,
        input_width    = 140,
        patch_size     = 20,
        enc_dim        = 128,
        enc_depth      = 2,
        enc_heads      = 4,
        dec_dim        = 128,
        dec_depth      = 2,
        dec_heads      = 4,
        mlp_ratio      = 4.0,
        use_variational=False,   # VQ-VAE
        use_vq         = True,
        codebook_size  = 84,     # your codebook size
        commitment_cost= 0.25
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    live_env  = PongEnv()

    # codebook and grid dims
    K = model.vq.codebook_size
    H, W = live_env.grid_h, live_env.grid_w

    # 4) Matplotlib interactive setup: 1×3 subplot
    plt.ion()
    fig, axes = plt.subplots(1,3, figsize=(9,3))
    for ax, title in zip(axes, ('Original','Reconstruction','VQ Indices')):
        ax.set_title(title); ax.axis('off')
    blank_frame = np.zeros((H*20, W*20, 3), np.uint8)
    blank_idx   = np.zeros((H, W), np.uint8)
    im_orig = axes[0].imshow(blank_frame)
    im_rec  = axes[1].imshow(blank_frame)
    im_idx  = axes[2].imshow(blank_idx, cmap='tab20', vmin=0, vmax=K-1)

    # 5) training loop
    for epoch in range(1, 51):
        running_loss = 0.0
        for batch_idx, batch in enumerate(loader, 1):
            x = batch.float().div(255.0).to(device)  # [32,3,100,140]

            # forward VQ-VAE
            recon_batch, vq_loss, idx_map = model(x)
            recon_loss = criterion(recon_batch, x)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % vis_interval == 0:
                #  original frame
                orig_t = live_env.step(random.choice([0,1]))
                orig_np = orig_t.permute(1,2,0).cpu().numpy()

                # reconstruction + idx for single frame
                inp = orig_t.unsqueeze(0).float().div(255.0).to(device)  # [1,3,H,W]
                with torch.no_grad():
                    recon_single, vq_single, idx_single = model(inp)
                # recon_single: [1,3,H,W] → image
                rec_img = recon_single.squeeze(0).permute(1,2,0).cpu().numpy()
                rec_img = (rec_img * 255).clip(0,255).astype(np.uint8)

                # idx_single: [1, H*W] → reshape to [H,W]
                idx_vec = idx_single.squeeze(0).cpu().numpy()
                idx_mat = idx_vec.reshape(H, W)

                # update displays
                im_orig.set_data(orig_np)
                im_rec .set_data(rec_img)
                im_idx .set_data(idx_mat)
                fig.canvas.draw()
                plt.pause(0.001)

        avg = running_loss / batch_idx
        print(f"Epoch {epoch:2d}  Recon Loss: {avg:.6f}")
        torch.save(model.state_dict(), "vqvit_pong.pth")

    plt.ioff()
    plt.close(fig)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_and_display(device)
