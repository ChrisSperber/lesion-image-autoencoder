"""Create plot to visualise issues with linearity in latent space transformation."""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(9001)

# --- Linear point cloud ---
n_points = 100
x_linear = np.linspace(-2, 2, n_points)
y_linear = x_linear + np.random.normal(scale=0.3, size=n_points)  # small noise
linear_data = np.column_stack((x_linear, y_linear))

# --- Non-linear point cloud: y = x^2 ---
x_nonlinear = np.linspace(-2, 2, n_points)
y_nonlinear = ((x_nonlinear + 0.5) ** 2) * 0.4 + np.random.normal(
    scale=0.3, size=n_points
)
non_linear_data = np.column_stack((x_nonlinear, y_nonlinear))

# PCA models
pca_linear = PCA(n_components=2).fit(linear_data)
pca_non_linear = PCA(n_components=2).fit(non_linear_data)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# linear
axs[0, 0].scatter(
    linear_data[:, 0], linear_data[:, 1], color=(0.3, 0.3, 0.3), alpha=0.6
)
axs[0, 0].set_title("Linear data")
axs[0, 0].set_xlabel("Variable 1")
axs[0, 0].set_ylabel("Variable 2")
axs[0, 0].axis("equal")

# PCA on linear data
axs[0, 1].scatter(
    linear_data[:, 0], linear_data[:, 1], color=(0.3, 0.3, 0.3), alpha=0.6
)
pca_vec = pca_linear.components_.T * 2
axs[0, 1].quiver(
    0,
    0,
    pca_vec[0, 0],
    pca_vec[1, 0],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="red",
    label="1st component direction",
)
axs[0, 1].set_title("PCA captures linearity")
axs[0, 0].set_xlabel("Variable 1")
axs[0, 0].set_ylabel("Variable 2")
axs[0, 1].axis("equal")
axs[0, 1].legend()

# Non-linear
axs[1, 0].scatter(
    non_linear_data[:, 0], non_linear_data[:, 1], color=(0.3, 0.3, 0.3), alpha=0.6
)
axs[1, 0].set_title("Non-linear data")
axs[0, 0].set_xlabel("Variable 1")
axs[0, 0].set_ylabel("Variable 2")
axs[1, 0].axis("equal")

# PCA on non-linear data
axs[1, 1].scatter(
    non_linear_data[:, 0], non_linear_data[:, 1], color=(0.3, 0.3, 0.3), alpha=0.6
)
pca_vec_nl = pca_non_linear.components_.T * 2
axs[1, 1].quiver(
    0,
    0,
    pca_vec_nl[0, 0],
    pca_vec_nl[1, 0],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="red",
    label="1st component direction",
)
axs[1, 1].set_title("PCA fails to capture data")
axs[0, 0].set_xlabel("Variable 1")
axs[0, 0].set_ylabel("Variable 2")
axs[1, 1].axis("equal")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig(f"{Path(__file__).stem}.png", dpi=300)
plt.show()
# %%
