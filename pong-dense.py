import random
import cv2
import torch
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

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

def test_render_loop():
    dev = torch.device('cpu')
    env = PongEnv(dev, random_miss=False)

    cv2.namedWindow('pong', cv2.WINDOW_NORMAL)
    #from vae import VAE
    #vae = VAE()
    #vae.load_state_dict(torch.load("trained_vae_pong_dense.pth", weights_only=True))
    from autoencoder_kl import AutoencoderKL
    vqvit = AutoencoderKL(
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
        use_variational=False,
        use_vq         = True,
        codebook_size  = 84,
        commitment_cost= 0.25
    ).to(dev)
    vqvit.load_state_dict(torch.load("vqvit_pong.pth"))

    while True:

        # get the image tensor (3×H×W), convert to H×W×3 numpy
        img_t = env.getState() # torch.Size([3, 100, 140])
        
        x = img_t.unsqueeze(0)/255.0
        recon_single,_,idx = vqvit(x)
        rec_img = recon_single.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        img = (rec_img * 255).clip(0,255).astype(np.uint8)
        #y = vae(x)
        #print(idx)
        #pix = ((y + 1)/2 * 255.0).clamp(0,255).round().to(torch.uint8).squeeze()
        #pix = (y  * 255.0).clamp(0,255).round().to(torch.uint8).squeeze()
        #img = pix.permute(1,2,0).cpu().numpy()         # H×W×3
        #img = img_t.permute(1,2,0).cpu().numpy()         # H×W×3

        cv2.imshow('pong', img)
        key = cv2.waitKey(200) & 0xFF                     # 200 ms per frame
        if key == 27:                                      # ESC to quit
            break

        # randomly move the paddle up/down
        action = random.choice([0,1])
        env.step(action)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_render_loop()
