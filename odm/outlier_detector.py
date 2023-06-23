import logging
import os
import time
from vae import vae
import numpy as np


class OutlierDetector:
    """
    Class for outlier detection.
    """

    @staticmethod
    def detect_outliers(
            features: np.ndarray,
            verbose: bool = False,
            timing: bool = False
    ):
        """
        Detect outliers using PyOD's VAE algorithm.

        Parameters:
        -----------
        features : (list)
            List of features to be used for outlier detection.
        pyod_algorithm : (str)
            Name of the PyOD algorithm to be used for outlier detection.
        redirect_output : (bool)
            If True, redirect the output of the PyOD algorithm to a file.
        return_decision_function : (bool)
            If True, return the decision function of the PyOD algorithm.
        **kwargs : (dict)
            Keyword arguments to be passed to the PyOD algorithm.
        """
        t0 = time.time()
        decision_scores = []

        if verbose is False:
            print("Turning off verbose mode...")
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            logging.getLogger("tensorflow").disabled = True
            logging.getLogger("pyod").disabled = True

        errors = {}

        try:
            print(f"Running VAE...")
            decision_scores, labels = vae(features)

        # if the algorithm causes an error, skip it and move on
        except Exception as e:
            import traceback
            traceback.print_exc()
            logging.error(f"Error running VAE on data "
                          f"{features} with exception {e}")
            errors["VAE"] = e
            labels = None


        print(
            f"about to save and len of tscore and imgs is "
            f"{len(decision_scores)} and {len(features)}"
        )

        if timing:
            print(f"OD detect_outliers took {time.time() - t0} seconds")

        return decision_scores, labels
