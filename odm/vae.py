# =============================================================================
# Variable Autoencoder (VAE) for outlier detection, wrapper for PyOD which
# can be found here:
# https://pyod.readthedocs.io/en/latest/pyod.models.html#pyod.models.vae.VAE
# ==============================================================================
import logging

import keras
import numpy as np
import ast
from keras.losses import get
from pyod.models.vae import VAE
from keras.losses import mse
from utils import *


def vae(data_x):
    """Variable Autoencoder (VAE) for outlier detection

    Parameters
    data_x : array-like of shape (n_samples, n_features)
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
                data_x[i] = np.pad(data_x[i], (0, len(data_x[0]) - 1),
                                   "constant")
            if len(data_x[i]) == 3:
                data_x[i] = np.pad(data_x[i], (0, len(data_x[0]) - 3),
                                   "constant")
        data_x = np.array(data_x)
    else:
        raise TypeError("Data type not supported.")

    logging.info("Data preprocessing complete.")

    # Now fetch the hyperparameters from the configuration file,
    # with fallbacks for defaults
    logging.info("Fetching hyperparameters from config file...")
    # Fetch hyperparameters as strings
    raw_values = {
        param: config.get("HYPERPARAMS", param, fallback=None)
        for param in [
            "latent_dim",
            "hidden_activation",
            "output_activation",
            "loss",
            "optimizer",
            "epochs",
            "batch_size",
            "dropout_rate",
            "l2_regularizer",
            "validation_size",
            "preprocessing",
            "verbose",
            "contamination",
            "gamma",
            "capacity",
            "random_state",
            "encoder_neurons",
            "decoder_neurons"
        ]
    }

    # Prepare default values
    default_values = {
        "latent_dim": 2,
        "hidden_activation": "relu",
        "output_activation": "sigmoid",
        "loss": mse,
        "optimizer": "adam",
        "epochs": 100,
        "batch_size": 32,
        "dropout_rate": 0.2,
        "l2_regularizer": 0.1,
        "validation_size": 0.1,
        "preprocessing": True,
        "verbose": 1,
        "contamination": 0.1,
        "gamma": 1.0,
        "capacity": 0.0,
        "random_state": None,
        "encoder_neurons": None,
        "decoder_neurons": None
    }

    # Convert string values to correct types, with fallbacks for empty fields
    values = {}
    for param, raw_val in raw_values.items():
        if raw_val == '':
            # Use default value if field is empty
            values[param] = default_values[param]
        else:
            try:
                if param in ["latent_dim", "epochs", "batch_size", "verbose"]:
                    # These parameters should be integers
                    values[param] = int(raw_val)
                elif param in ["dropout_rate", "l2_regularizer",
                               "validation_size",
                               "contamination", "gamma", "capacity"]:
                    # These parameters should be floats
                    values[param] = float(raw_val)
                elif param in ["random_state", "encoder_neurons",
                               "decoder_neurons"]:
                    # These parameters should be evaluated as Python expressions
                    values[param] = ast.literal_eval(raw_val)
                elif param in ["loss"]:
                    # These parameters should be evaluated as Keras loss
                    # functions
                    values[param] = get(raw_val)
                else:
                    # All other parameters are kept as strings
                    values[param] = raw_val
            except Exception as e:
                logging.error(f"Error processing parameter {param}: {e}")
                return None
    logging.info("Hyperparameters fetched.")

    print_properties("Hyperparameters", **values)

    # Initialize and train the VAE
    # Create your VAE model with the above parameters
    clf = VAE(contamination=values["contamination"],
              gamma=values["gamma"],
              capacity=values["capacity"],
              latent_dim=values["latent_dim"],
              encoder_neurons=values["encoder_neurons"],
              decoder_neurons=values["decoder_neurons"],
              hidden_activation=values["hidden_activation"],
              output_activation=values["output_activation"],
              loss=values["loss"],
              optimizer=values["optimizer"],
              epochs=values["epochs"],
              batch_size=values["batch_size"],
              dropout_rate=values["dropout_rate"],
              l2_regularizer=values["l2_regularizer"],
              validation_size=values["validation_size"],
              preprocessing=values["preprocessing"],
              verbose=values["verbose"],
              random_state=values["random_state"])

    clf.fit(data_x)

    return clf.decision_scores_, clf.labels_
