"""Model for deep, non-linear autoencoder."""

import torch
from autoencoder_utils.autoencoder_configs import TARGET_SHAPE_4CHANNEL
from torch import nn
from utils import N_LATENT_VARIABLES


class Conv3dAutoencoder(nn.Module):
    """A simple convolutional autoencoder for 3D lesion masks."""

    def __init__(
        self, input_shape=TARGET_SHAPE_4CHANNEL, latent_dim=N_LATENT_VARIABLES
    ):
        """Deep nonlinear autoencoder class.

        Args:
            input_shape (tuple): Shape of input tensor (C, D, H, W).
            latent_dim (int): Dimension of the compressed latent space.

        """
        super().__init__()
        self.input_shape = input_shape

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(
                1, 8, kernel_size=3, padding=1
            ),  # (1, 80, 96, 80) -> (8, 80, 96, 80)
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (8, 40, 48, 40)
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (16, 20, 24, 20)
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (32, 10, 12, 10)
        )

        # Latent space (flatten → linear → reshape)
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(32 * 10 * 12 * 10, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 32 * 10 * 12 * 10)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),  # -> (16, 20, 24, 20)
            nn.ReLU(),
            nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2),  # -> (8, 40, 48, 40)
            nn.ReLU(),
            nn.ConvTranspose3d(8, 1, kernel_size=2, stride=2),  # -> (1, 80, 96, 80)
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, x):  # noqa: D102
        batch_size = x.size(0)

        z = self.encoder(x)  # shape (B, 32, 10, 12, 10)
        z_flat = self.flatten(z)  # shape (B, latent_dim_in)
        z_latent = self.fc_enc(z_flat)  # shape (B, latent_dim)

        z_recon = self.fc_dec(z_latent).view(batch_size, 32, 10, 12, 10)
        x_recon = self.decoder(z_recon)
        return x_recon

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent vector (before decoding)."""
        z = self.encoder(x)
        z_flat = self.flatten(z)
        z_latent = self.fc_enc(z_flat)
        return z_latent
