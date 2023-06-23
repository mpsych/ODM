# =============================================================================
# Variable Autoencoder (VAE) for outlier detection, wrapper for PyOD which
# can be found here:
# https://pyod.readthedocs.io/en/latest/pyod.models.html#pyod.models.vae.VAE
# ==============================================================================
import numpy as np
from pyod.models.vae import VAE
from utils import *


def vae(data_x, **hyperparams):
    """Variable Autoencoder (VAE) for outlier detection

    Parameters
    data_x : array-like of shape (n_samples, n_features)
        The input samples.
    hyperparams : dict
        Hyperparameters for VAE
    """
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Preprocess the data
    logging.info("Preprocessing data in vae...")
    if isinstance(data_x, np.ndarray):
        if len(data_x.shape) == 1:
            data_x = data_x.reshape(-1, 1)
        elif len(data_x.shape) == 3:
            data_x = data_x.reshape(data_x.shape[0], -1)
    elif isinstance(data_x, list):
        for i in range(len(data_x)):
            if len(data_x[i]) == 1:
                data_x[i] = np.pad(data_x[i], (0, len(data_x[0]) - 1), "constant")
            if len(data_x[i]) == 3:
                data_x[i] = np.pad(data_x[i], (0, len(data_x[0]) - 3), "constant")
        data_x = np.array(data_x)
    else:
        raise TypeError("Data type not supported.")

    logging.info("Data preprocessing complete.")

    # Initialize and train the VAE
    clf = VAE(**hyperparams)

    clf.fit(data_x)

    return clf.decision_scores_, clf.labels_
