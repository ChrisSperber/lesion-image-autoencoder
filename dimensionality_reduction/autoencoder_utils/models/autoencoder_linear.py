"""Model for simple linear autoencoder."""

import torch
from autoencoder_utils.autoencoder_configs import TARGET_SHAPE_4CHANNEL
from torch import nn
from utils import N_LATENT_VARIABLES


class LinearAutoencoder(nn.Module):
    """A simple linear autoencoder for 3D lesion masks.

    Encoder: Flattens the input and compresses to a latent space.
    Decoder: Reconstructs back to the original shape. Utilizes a sigmoid activation function to set
    all outputs between between 0 and 1.
    """

    def __init__(
        self, input_shape=TARGET_SHAPE_4CHANNEL, latent_dim=N_LATENT_VARIABLES
    ):
        """Linear autoencoder class.

        Args:
            input_shape (tuple): Shape of input tensor (C, D, H, W).
            latent_dim (int): Dimension of the compressed latent space.

        """
        super().__init__()
        self.input_shape = input_shape
        self.input_dim = torch.prod(torch.tensor(input_shape))

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.input_dim),
            nn.Sigmoid(),  # Output values in [0, 1]
        )

    def forward(self, x):  # noqa: D102
        batch_size = x.size(0)
        z = self.encoder(x)
        recon = self.decoder(z)
        recon = recon.view(batch_size, *self.input_shape)
        return recon

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent vector (before decoding)."""
        return self.encoder(x)
