import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Load grayscale image
img = data.astronaut()
img = rgb2gray(img)

# Compute statistics
mean_intensity = np.mean(img)
std_intensity = np.std(img)

print(f"Average intensity: {mean_intensity:.4f}")
print(f"Standard deviation: {std_intensity:.4f}")

# --- Initialize a line (open snake) ---
n_points = 40
r = np.linspace(20, 120, n_points)      # vertical coordinate (rows, y-axis)
c = np.linspace(270, 310, n_points)       # horizontal coordinate (cols, x-axis)
init = np.array([r, c]).T                # shape (N,2)

# Apply Gaussian blur (same as snake input)
img_blurred = gaussian(img, sigma=3, preserve_range=False)

# --- Active contour ---
snake = active_contour(
    img_blurred,
    init,
    alpha=0.0015,
    beta=1,
    gamma=0.001,
    w_edge = 1,
    w_line = 0
    ,boundary_condition="free"  # keeps ends open
)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img_blurred, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3, label="Initial line")
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3, label="Refined snake")
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
ax.legend()

plt.show()



# %% --- Plot original and blurred images ---

# Apply Gaussian blur (same as snake input)
img_blurred = gaussian(img, sigma=3, preserve_range=False)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(img, cmap=plt.cm.gray)
axs[0].set_title("Original image")
axs[0].plot(init[:, 1], init[:, 0], '--r', lw=2)
axs[0].set_xticks([]), axs[0].set_yticks([])

axs[1].imshow(img_blurred, cmap=plt.cm.gray)
axs[1].set_title("Blurred image (sigma=3)")
axs[1].plot(init[:, 1], init[:, 0], '--r', lw=2)
axs[1].set_xticks([]), axs[1].set_yticks([])

plt.show()
