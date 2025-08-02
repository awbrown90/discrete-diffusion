import numpy as np
import matplotlib.pyplot as plt

# ---- CONFIG ----
FILENAME = 'miss_log_rand.txt'  # your text file
DELIMITER = ','            # comma-separated
SKIP_HEADER = 1            # skip the "step,misses" line

# ---- LOAD DATA ----
data = np.loadtxt(FILENAME, delimiter=DELIMITER, skiprows=SKIP_HEADER)
steps  = data[:, 0]
misses = data[:, 1]

# ---- PLOT ----
plt.figure()
plt.plot(steps, misses)
plt.xlabel('Step')
plt.ylabel('Misses')
plt.title('Misses over Steps')
plt.grid(True)
plt.tight_layout()
plt.show()
