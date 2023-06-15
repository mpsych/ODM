# =============================================================================
# Variable Autoencoder (VAE) for outlier detection, wrapper for PyOD which
# can be found here: https://pyod.readthedocs.io/en/latest/pyod.models.html#pyod.models.vae.VAE
# ==============================================================================
import numpy as np
import configparser
import ast
from keras.losses import get
from pyod.models.vae import VAE


def vae(data_x):
    """Variable Autoencoder (VAE) for outlier detection

    Parameters
    data_x : array-like of shape (n_samples, n_features)
    """
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Preprocess the data
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

    # Now fetch the hyperparameters from the configuration file, with fallbacks for defaults
    latent_dim = config.getint("HYPERPARAMS", "latent_dim", fallback=2)
    hidden_activation = config.get("HYPERPARAMS", "hidden_activation", fallback="relu")
    output_activation = config.get(
        "HYPERPARAMS", "output_activation", fallback="sigmoid"
    )
    loss = get(config.get("HYPERPARAMS", "loss", fallback="mse"))
    optimizer = config.get("HYPERPARAMS", "optimizer", fallback="adam")
    epochs = config.getint("HYPERPARAMS", "epochs", fallback=100)
    batch_size = config.getint("HYPERPARAMS", "batch_size", fallback=32)
    dropout_rate = config.getfloat("HYPERPARAMS", "dropout_rate", fallback=0.2)
    l2_regularizer = config.getfloat("HYPERPARAMS", "l2_regularizer", fallback=0.1)
    validation_size = config.getfloat("HYPERPARAMS", "validation_size", fallback=0.1)
    preprocessing = config.getboolean("HYPERPARAMS", "preprocessing", fallback=True)
    verbose = config.getint("HYPERPARAMS", "verbose", fallback=1)
    contamination = config.getfloat("HYPERPARAMS", "contamination", fallback=0.1)
    gamma = config.getfloat("HYPERPARAMS", "gamma", fallback=1.0)
    capacity = config.getfloat("HYPERPARAMS", "capacity", fallback=0.0)
    random_state = ast.literal_eval(
        config.get("HYPERPARAMS", "random_state", fallback="None")
    )
    encoder_neurons = ast.literal_eval(
        config.get("HYPERPARAMS", "encoder_neurons", fallback="None")
    )
    decoder_neurons = ast.literal_eval(
        config.get("HYPERPARAMS", "decoder_neurons", fallback="None")
    )

    # Check for None values (since configparser treats empty fields as strings)
    random_state = None if random_state == "None" else random_state
    encoder_neurons = None if encoder_neurons == "None" else encoder_neurons
    decoder_neurons = None if decoder_neurons == "None" else decoder_neurons

    # Initialize and train the VAE
    clf = VAE(
        encoder_neurons=encoder_neurons,
        decoder_neurons=decoder_neurons,
        latent_dim=latent_dim,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        loss=loss,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        l2_regularizer=l2_regularizer,
        validation_size=validation_size,
        preprocessing=preprocessing,
        verbose=verbose,
        random_state=random_state,
        contamination=contamination,
        gamma=gamma,
        capacity=capacity,
    )

    clf.fit(data_x)

    return clf.decision_scores_, clf.labels_
