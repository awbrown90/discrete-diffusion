import math, random
from tqdm import trange
import torch
import torch.nn as nn
from vae import VAE
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

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

    def reset_misses(self):
        self.miss_counter = 0

    def get_misses(self):
        return self.miss_counter

class PongDataset(IterableDataset):
    def __init__(self, steps_per_epoch=10000):
        super().__init__()
        # now `device` is its own parameter
        self.env   = PongEnv(device=torch.device("cpu"), random_miss=True)
        self.steps = steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps):
            action = random.choice([0,1])
            img = self.env.step(action)       # already on the right device
            yield img

def train_vae(device):
    # 1) hyper-parameters
    batch_size   = 32
    lr           = 3e-4
    n_epochs     = 50
    recon_weight = 1.0      # weight for MSE loss
    vq_weight    = 1.0      # weight for VQ loss

    # 2) data loader
    dataset = PongDataset(steps_per_epoch=2000)
    loader  = DataLoader(dataset, batch_size=batch_size, num_workers=2)

    # 3) model, optimizer
    vae = VAE().to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)

    # 4) training loop
    for epoch in range(1, n_epochs+1):
        epoch_recon, epoch_vq, epoch_total = 0.0, 0.0, 0.0
        num_batches = 0
        for img_cpu in loader:
            # x: [B,3,H,W], floats in [0,1]
            # map to [–1,1] as your encode() expects
            x = img_cpu.to(device, non_blocking=True).float() / 255.0
            x_in = 2*(x - 0.5)

            # forward pass
            recon, vq_loss, _ = vae(x_in)
            # recon: [B,3,H,W], in [–1,1] via Tanh

            # 5) compute losses
            recon_loss = F.mse_loss(recon, x_in)
            loss       = recon_weight * recon_loss + vq_weight * vq_loss

            # 6) backward + step
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_recon += recon_loss.item()
            epoch_vq    += vq_loss.item()
            epoch_total += loss.item()
            num_batches += 1

        # logging
        print(f"Epoch {epoch:3d}  ∘  Recon={epoch_recon/num_batches:.4f}"
              f"  VQ={epoch_vq/num_batches:.4f}"
              f"  Total={epoch_total/num_batches:.4f}")

    # return trained model
    return vae

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_vae = train_vae(device)
