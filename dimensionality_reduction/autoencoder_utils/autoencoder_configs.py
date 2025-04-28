"""Configurations for autoencoder."""

TARGET_SHAPE = (80, 96, 80)
TARGET_SHAPE_4CHANNEL = (1,) + TARGET_SHAPE

# the number of latent variables is defined by the number of variables required to explain 75% of
# variance with PCA in the continuous data. This value is printed as an output of
# a_simple_feature_reduction.py
N_LATENT_VARIABLES = 75
