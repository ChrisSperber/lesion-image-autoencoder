"""Configurations for autoencoder."""

from dataclasses import dataclass
from pathlib import Path

TARGET_SHAPE: tuple[int, int, int] = (80, 96, 80)
TARGET_SHAPE_4CHANNEL: tuple[int, int, int, int] = (1,) + TARGET_SHAPE

N_LATENT_VARIABLES: int = 64

AUTOENCODER_OUTPUTS_DIR: Path = Path(__file__).parent / "outputs"


@dataclass
class TrainingConfig:
    """Configuration container for training autoencoders.

    Attributes:
        device (str): Device to use for training ('cuda' or 'cpu').
        epochs (int): Maximum number of training epochs.
        lr (float): Learning rate for the optimizer.
        batch_size_linear (int): Batch size for training linear autoencoders.
        batch_size_deep (int): Batch size for training deep/nonlinear autoencoders.
        patience_early_stopping (int): Number of epochs without improvement before early stopping.
        patience_reduce_lr (int): Number of stagnant epochs before reducing learning rate.
        debug_mode (bool): If True, activates short runs with limited data and epochs.

    """

    device: str = "cuda"
    epochs: int = 500
    lr: float = 0.001
    batch_size_linear: int = 64
    batch_size_deep: int = 4
    patience_early_stopping: int = 10
    patience_reduce_lr: int = 3
    debug_mode: bool = False


autoencoder_config = TrainingConfig()
