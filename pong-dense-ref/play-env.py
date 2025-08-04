import numpy as np
import cv2
from container import GameEnvironment
import torch
import math, itertools
from vae_fix import VAE_models

env = GameEnvironment()
obs = env.reset()
obs = env.step(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vqvae = VAE_models["vit-l-20-shallow-encoder"]().to(device)
vqvae_ckpt = torch.load('vqvae.pth', weights_only=True, map_location=device)
vqvae.load_state_dict(vqvae_ckpt)

h,w = 256,256

# build your 3×3 binary pattern lookup (84 = C(9,3))
PATTERNS = []
for combo in itertools.combinations(range(9), 3):
    v = torch.zeros(9, dtype=torch.uint8)
    v[list(combo)] = 1
    PATTERNS.append(v.view(3,3))

def to_img(tokens, zoom=32):
    N = len(tokens); n = int(math.sqrt(N))
    patches = [PATTERNS[int(t)] for t in tokens]
    rows = [torch.cat(patches[r*n:(r+1)*n], dim=1) for r in range(n)]
    grid = torch.cat(rows, dim=0).numpy() * 255   # shape (n*3,n*3)
    img  = cv2.resize(grid.astype(np.uint8),
                      (grid.shape[1]*zoom, grid.shape[0]*zoom),
                      interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Key mapping: arrow keys to action index
KEY_TO_ACTION = {
    81: 1,  # Left arrow
    82: 3,  # Up arrow
    83: 2,  # Right arrow
    84: 0,  # Down arrow
}

while True:
    
    x_images = (torch.from_numpy(obs)
                .permute(2,0,1)
                .unsqueeze(0)
                .float()
                .div(255.0)
                .to(device))
        
    x_images_normalized = x_images * 2 - 1

    recon_state, _, idx_map = vqvae(x_images_normalized)

    # recon to numpy
    rec_im = (recon_state[0]
            .clamp(0,1)
            .mul(255)
            .byte()
            .permute(1,2,0)
            .cpu()
            .numpy())
            
    tokens  = idx_map[0].cpu().numpy()  # length=64
    gridimg = to_img(tokens, zoom=16)

    h,w     = 256,256
    orig_r  = cv2.resize(obs,  (w,h), interpolation=cv2.INTER_AREA)
    recon_r = cv2.resize(rec_im, (w,h), interpolation=cv2.INTER_AREA)
    grid_r  = cv2.resize(gridimg,   (w,h), interpolation=cv2.INTER_NEAREST)

    combo = np.hstack([orig_r, recon_r, grid_r])
    cv2.imshow("env", cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(0) & 0xFF

    if key == 27:  # ESC to quit
        break

    # Get action from arrow key press
    if key in KEY_TO_ACTION:
        action = KEY_TO_ACTION[key]
        obs = env.step(action)

cv2.destroyAllWindows()