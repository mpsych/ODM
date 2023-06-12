import logging
import os
import time
import contextlib
from vae import *


class OutlierDetector:
    """
    Class for outlier detection.
    """

    @staticmethod
    def detect_outliers(
        features: np.ndarray,
        pyod_algorithm: str,
        redirect_output: bool = False,
        return_decision_function: bool = False,
        timing: bool = False,
        **kwargs,
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
        verbose = kwargs.get("verbose", False)

        if verbose is False:
            print("Turning off verbose mode...")
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            logging.getLogger("tensorflow").disabled = True
            logging.getLogger("pyod").disabled = True

        errors = {}
        decision_scores = None

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
                            ) = vae(features, pyod_algorithm=pyod_algorithm, **kwargs)
                        else:
                            decision_scores, labels = vae(
                                features, pyod_algorithm=pyod_algorithm, **kwargs
                            )
            else:
                print(f"Running {pyod_algorithm}...")

                if return_decision_function:
                    (
                        decision_scores,
                        labels,
                        t_decision_function,
                    ) = vae(features, pyod_algorithm=pyod_algorithm, **kwargs)

                else:
                    decision_scores, labels = vae(
                        features, pyod_algorithm=pyod_algorithm, **kwargs
                    )

        # if the algorithm causes an error, skip it and move on
        except Exception as e:
            logging.error(
                f"Error running {pyod_algorithm} on data {features} with exception {e}"
            )
            errors[pyod_algorithm] = e
            labels = None

        print(
            f"about to save and len of tscore and imgs is {len(decision_scores)} and {len(features)}"
        )

        if timing:
            print(f"OD detect_outliers took {time.time() - t0} seconds")

        return decision_scores, labels
