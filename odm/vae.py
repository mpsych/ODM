import numpy as np


def vae(data_x, pyod_algorithm, **kwargs):
    """Detect outliers using PyOD algorithm. Default algorithm is HBOS.
    See PYOD documentation to see which arguments are available for each
    algorithm and to get description of each algorithm and how the
    arguments work with each algorithm.
    link: https://pyod.readthedocs.io/en/latest/pyod.html
    """
    return_decision_function = kwargs.get("return_decision_function", False)

    # # make sure data_x is a 2d array and not a 1d array or 3D array
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

    if pyod_algorithm == "VAE":
        from pyod.models.vae import VAE
        from keras.losses import mse

        if "VAE" in kwargs:
            clf = VAE(**kwargs["VAE"])
        else:
            latent_dim = kwargs.get("latent_dim", 2)
            hidden_activation = kwargs.get("hidden_activation", "relu")
            output_activation = kwargs.get("output_activation", "sigmoid")
            loss = kwargs.get("loss", mse)
            optimizer = kwargs.get("optimizer", "adam")
            epochs = kwargs.get("epochs", 100)
            batch_size = kwargs.get("batch_size", 32)
            dropout_rate = kwargs.get("dropout_rate", 0.2)
            l2_regularizer = kwargs.get("l2_regularizer", 0.1)
            validation_size = kwargs.get("validation_size", 0.1)
            preprocessing = kwargs.get("preprocessing", True)
            verbose = kwargs.get("verbose", 1)
            contamination = kwargs.get("contamination", 0.1)
            gamma = kwargs.get("gamma", 1.0)
            capacity = kwargs.get("capacity", 0.0)
            random_state = kwargs.get("random_state", None)
            encoder_neurons = kwargs.get("encoder_neurons", None)
            decoder_neurons = kwargs.get("decoder_neurons", None)
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

    else:
        raise ValueError("Algorithm not supported")

    clf.fit(data_x)

    if return_decision_function:
        return clf.decision_scores_, clf.labels_, clf.decision_function

    return clf.decision_scores_, clf.labels_
