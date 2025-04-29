"""Configurations for autoencoder."""

from pathlib import Path

TARGET_SHAPE = (80, 96, 80)
TARGET_SHAPE_4CHANNEL = (1,) + TARGET_SHAPE

# the number of latent variables is defined by the number of variables required to explain 75% of
# variance with PCA in the continuous data. This value is printed as an output of
# a_simple_feature_reduction.py
N_LATENT_VARIABLES = 75

BATCH_SIZE = 4
EPOCHS = 500  # high number as upper bound; early stopping should trigger much earlier
LEARNING_RATE = 0.001  # initial learning rate, common default for ADAM

DEVICE = "cuda"  # set to "cuda" or "cpu"

# For early stopping:
PATIENCE_EARLY_STOPPING = 10

# For ReduceLROnPlateau:
PATIENCE_REDUCE_LR_PLATEAU = 3

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
