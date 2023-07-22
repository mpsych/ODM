from typing import Tuple
from .vae import vae

import logging
import numpy as np
import os
import time


class OutlierDetector:
    """Class for outlier detection."""

    @staticmethod
    def detect_outliers(
        features: np.ndarray,
        log_to_terminal: bool = False,
        timing: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect outliers using PyOD's VAE algorithm.

        Parameters
        ----------
        features : list
            List of features to be used for outlier detection.
        log_to_terminal : bool, optional
            Whether to print verbose output. The default is False.
        timing : bool, optional
            Whether to time the function. The default is False.
        **kwargs : dict, Any
            hyperparameters for VAE
        """
        t0 = time.time()
        decision_scores = []

        if not log_to_terminal:
            logging.info("Turning off verbose mode...")
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            logging.getLogger("tensorflow").disabled = True
            logging.getLogger("pyod").disabled = True

        errors = {}

        try:
            logging.info("Running VAE...")
            decision_scores, labels = vae(features, **kwargs)

        except Exception as e:
            import traceback

            traceback.print_exc()
            logging.error(
                f"Error running VAE on data " f"{features} with exception {e}"
            )
            errors["VAE"] = e
            labels = None

        logging.info(
            f"len of decision_scores and imgs is "
            f"{len(decision_scores)} and {len(features)}"
        )

        if timing:
            logging.info(f"OD detect_outliers took {time.time() - t0} seconds")

        return decision_scores, labels
