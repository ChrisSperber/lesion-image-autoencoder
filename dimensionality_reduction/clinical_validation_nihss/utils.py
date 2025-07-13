"""Utils for clinical validation."""

import nibabel as nib
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from skopt import BayesSearchCV
from skopt.space import Real

BINARISATION_THRESHOLD_ORIG_LESION = 0.2

BAYESIAN_OPTIMISATION_ITERATIONS_ELASTIC_NET = 30
BAYESIAN_OPTIMISATION_ITERATIONS_SVR = 50
RNG_SEED = 9001


def load_masked_vectorised_images(
    lesion_path_list: list[str], min_lesion_threshold: int
) -> np.ndarray:
    """Load lesion segmentations from NIfTI files and return a 2D array of relevant features.

    Each image is flattened into a 1D vector (row), resulting in a 2D matrix
    of shape (n_subjects, n_voxels). Voxels with little to no lesion coverage are masked away.

    Args:
        lesion_path_list: List of file paths to 3D lesion NIfTI images. All images must have the
            same shape.
        min_lesion_threshold: minimum number of lesions per voxel to be included. Voxels that are
            lesioned less than this threshold are masked away from the output. Lesions are defined
            binarily as voxels with a p value above BINARISATION_THRESHOLD_ORIG_LESION

    Returns:
        2D NumPy array of shape (n_images, n_included_voxels).

    """
    images = []
    for path in lesion_path_list:
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        images.append(data.ravel())
    images_2d = np.stack(images)

    # set voxels with low p values to 0 to remove noise
    images_2d[images_2d < BINARISATION_THRESHOLD_ORIG_LESION] = 0

    # remove voxels with little to no information, i.e. which is rarely lesioned
    nonzero_counts = np.count_nonzero(images_2d, axis=0)
    voxels_to_keep = nonzero_counts >= min_lesion_threshold
    return images_2d[:, voxels_to_keep]


def train_test_split_indices(
    n: int, test_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate indices for train and test folds.

    Args:
        n (int): Total sample size
        test_ratio (float): Ratio of sample assigned to test fold.
        seed (int): RNG seed.

    Returns:
        Tuple: Arrays of training and test indices.

    """
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed
    )
    return train_idx, test_idx


def fit_elastic_net_bayes_opt(
    x: np.ndarray,
    y: np.ndarray,
    n_iter: int = BAYESIAN_OPTIMISATION_ITERATIONS_ELASTIC_NET,
) -> ElasticNet:
    """Fit ElasticNet using Bayesian optimization to tune hyperparameters.

    Features are standardised via make_pipeline to obtain comparable results with regularisation.

    Args:
        x (np.ndarray): Predictors
        y (np.ndarray): Target Variable
        n_iter (int, optional): Number of iterations for Bayesian optimisation.
            Defaults to BAYESIAN_OPTIMISATION_ITERATIONS.

    Returns:
        ElasticNet: Optimised elastic net model.

    """
    pipe = make_pipeline(StandardScaler(), ElasticNet(max_iter=5000))

    param_space = {
        "elasticnet__alpha": Real(1e-3, 1e2, prior="log-uniform"),
        "elasticnet__l1_ratio": Real(0.1, 0.9),
    }

    opt = BayesSearchCV(
        estimator=pipe,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=4,
        scoring="r2",
        n_jobs=-1,
        random_state=RNG_SEED,
        verbose=0,
    )
    opt.fit(x, y)
    return opt.best_estimator_


def fit_svr_bayes_opt(
    x: np.ndarray, y: np.ndarray, n_iter: int = BAYESIAN_OPTIMISATION_ITERATIONS_SVR
) -> SVR:
    """Fit SVR using Bayesian optimization to tune hyperparameters.

    Features are standardized via make_pipeline to improve convergence.

    Args:
        x (np.ndarray): Predictors.
        y (np.ndarray): Target Variable.
        n_iter (int, optional): Number of iterations for Bayesian optimisation.

    Returns:
        SVR: Optimized SVR model.

    """
    pipe = make_pipeline(StandardScaler(), SVR())

    param_space = {
        "svr__C": Real(1e-3, 1e3, prior="log-uniform"),
        "svr__epsilon": Real(1e-3, 1.0, prior="log-uniform"),
        "svr__gamma": Real(1e-4, 10.0, prior="log-uniform"),  # Only for 'rbf' kernel
    }

    opt = BayesSearchCV(
        estimator=pipe,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=4,
        scoring="r2",
        n_jobs=-1,
        random_state=RNG_SEED,
        verbose=0,
    )
    opt.fit(x, y)
    return opt.best_estimator_
