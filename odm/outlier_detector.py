from datetime import datetime
import hashlib
import os
import time
import contextlib
from types import SimpleNamespace
import logging
import numpy as np

logger = logging.getLogger(__name__)

ALGORITHMS = [
    "VAE",
]

CACHE_PATH = r"/tmp/odm/cache_files/"
CACHE_FILE = r"ODM_CACHE.json"
LOG_DIR = r"/tmp/odm/log/"


class OutlierDetector:
    def __init__(
        self,
        run_id: str,
        algorithms: list = None,
        imgs: list = None,
        features: list = None,
        verbose: bool = False,
        timing: bool = False,
    ) -> None:
        """Initializes the OutlierDetector class"""
        t0 = time.time()
        self.__run_id = run_id
        self.__algorithms = ALGORITHMS if algorithms is None else algorithms
        self.__imgs = imgs
        self.__features = features
        if verbose is False:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            logging.getLogger("tensorflow").disabled = True
            logging.getLogger("pyod").disabled = True
        self.__errors = None

        if timing:
            logger.info(f"OD __init__ took {time.time() - t0} seconds")

    @staticmethod
    def detect_outliers(
        features,
        imgs,
        pyod_algorithm,
        timing=False,
        id_=None,
        redirect_output=False,
        return_decision_function=False,
        **kwargs,
    ):
        """Detect outliers using pyod algorithms"""

        t0 = time.time()

        errors = {}
        decision_scores = None

        # caselist contains the list of image paths
        caselist = kwargs.get("caselist", None)

        try:
            if redirect_output:
                print(f"Running {pyod_algorithm}...", end=" ")
                out_file = "outlier_output.txt"
                with open(out_file, "w") as f:
                    with contextlib.redirect_stdout(f):
                        if return_decision_function:
                            (
                                decision_scores,
                                labels,
                                t_decision_function,
                            ) = OutlierDetector._detect_outliers(
                                features, pyod_algorithm=pyod_algorithm, **kwargs
                            )

                            accuracy = t_decision_function
                        else:
                            decision_scores, labels = OutlierDetector._detect_outliers(
                                features, pyod_algorithm=pyod_algorithm, **kwargs
                            )
            else:
                print(f"Running {pyod_algorithm}...")

                if return_decision_function:
                    (
                        decision_scores,
                        labels,
                        t_decision_function,
                    ) = OutlierDetector._detect_outliers(
                        features, pyod_algorithm=pyod_algorithm, **kwargs
                    )

                    accuracy = t_decision_function

                else:
                    decision_scores, labels = OutlierDetector._detect_outliers(
                        features, pyod_algorithm=pyod_algorithm, **kwargs
                    )

        # if the algorithm causes an error, skip it and move on
        except Exception as e:
            logging.error(
                f"Error running {pyod_algorithm} on data {features} with exception {e}"
            )
            errors[pyod_algorithm] = e
            accuracy = -1
            labels = None

        kwargs_hash = dict_to_hash(kwargs)
        # get the date and time to use as file name
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # if the id is not None add it as part of the file name
        if id_ is not None:
            date_time = f"{date_time}_{str(id_)}"

        # put the feature type name with the number bad to use in the cache file
        filename = f"{date_time}_{pyod_algorithm}%_{kwargs_hash}"

        print(
            f"about to save and len of tscore and imgs is {len(decision_scores)} and {len(imgs)}"
        )

        # TODO: This is expecting the imgs to be a SimpleNamespace from Data API, need to
        #       change this so that it is no longer dependent on that
        OutlierDetector._save_outlier_path_log(filename, imgs, decision_scores)

        if timing:
            display_timing(t0, "running " + pyod_algorithm)

        return decision_scores, labels, accuracy

    @staticmethod
    def _detect_outliers(data_x, pyod_algorithm, **kwargs):
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

    @staticmethod
    def _validate_lengths(data_x, labels):
        """Validates the lengths of the data
        Parameters
        ----------
        data_x : list
            The data to validate
        labels : list, np.ndarray
            The labels to validate
        """
        if isinstance(labels, list):
            for label in labels:
                if len(data_x) != len(label):
                    raise ValueError(
                        "LengthError: The length of the data and labels must be the same"
                    )
        else:
            if len(data_x) != len(labels):
                raise ValueError("The length of the data and labels must be the same")

    @staticmethod
    def _save_outlier_path_log(filename, imgs, t_scores, caselist: str = None):
        """Saves the outlier path log
        Parameters
        ----------
        filename : str
            The cache key
        t_scores : list, np.array
            The training scores
        imgs : list
            The images
        caselist : str
            path to the caselist file, which contains the list of paths to the
            images
        """
        # validate the length of the scores and images
        OutlierDetector._validate_lengths(imgs, t_scores)

        # read the caselist file into a list
        with open(caselist, "r") as f:
            paths = [line.strip() for line in f]

        # the path to write the log to will be the LOG_DIR with the cache key
        # appended to it
        path = os.path.join(LOG_DIR, filename + ".txt")

        # create the log directory if it doesn't exist with all the parent directories
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # this will put all the data into order from highest to lowest score
        image_list = []
        if isinstance(t_scores, np.ndarray):
            for i in range(len(imgs)):
                image_list.append(SimpleNamespace(image=paths[i], score=t_scores[i]))
        elif isinstance(t_scores, list):
            if isinstance(t_scores[0], np.ndarray):
                for j in range(len(t_scores)):
                    for i in range(len(imgs)):
                        image_list.append(
                            SimpleNamespace(image=paths[i], score=t_scores[j][i])
                        )
            else:
                for i in range(len(imgs)):
                    image_list.append(
                        SimpleNamespace(image=paths[i], score=t_scores[i])
                    )

        image_list = sorted(image_list, key=lambda x: x.score, reverse=True)

        # write the paths in order they are in the image_list
        with open(path, "w") as f:
            for i in range(len(image_list)):
                f.write(image_list[i].image + "\n")

        # write the scores in order they are in the image_list
        path = os.path.join(LOG_DIR, filename + "_scores.txt")
        with open(path, "w") as f:
            for i in range(len(image_list)):
                f.write(str(image_list[i].score) + "\n")


def dict_to_hash(d):
    """Converts a dictionary to a hash and returns it as a string
    Parameters
    ----------
    d : dict
        The dictionary to be converted
    Returns
    -------
    hash : str
        The hash of the dictionary
    """
    return hashlib.md5(str(d).encode("utf-8")).hexdigest()


def display_timing(t0, label: str):
    """Prints the time it takes to perform a certain action

    Parameters
    ----------
    t0 : float
        the time when the action was performed
    label : str
        the label of the action
    """
    print(
        "{:<25s}{:<10s}{:>10f}{:^5s}".format(
            label, "...took ", time.time() - t0, " seconds"
        )
    )
