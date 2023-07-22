from .normalizations import Normalize
from tqdm import tqdm
from types import SimpleNamespace
from typing import Union, List
import logging
import mahotas as mh
import numpy as np
import time


class Features:
    @staticmethod
    def histogram(
        pixels: Union[
            List[Union[SimpleNamespace, np.ndarray]], SimpleNamespace, np.ndarray
        ],
        bins: int = 256,
        timing: bool = False,
        norm_type: str = None,
    ) -> np.ndarray:
        """Create histogram of data

        Parameters
        ----------
        pixels : list or np.ndarray
            List of images or single image
        bins : int, optional
            Number of bins. The default is 256.
        norm_type : str, optional
            Type of normalization. The default is None.
        timing : bool, optional
            Whether to time the function. The default is False.

        Returns
        -------
        np.ndarray
            List of histograms
        """
        t0 = time.time()
        histograms = []
        if isinstance(pixels, list):
            for i in tqdm(range(len(pixels))):  # adding progress bar here
                if isinstance(pixels[i], SimpleNamespace):
                    tmp_pixels = pixels[i].pixels.copy()
                else:
                    tmp_pixels = pixels[i].copy()
                if norm_type is not None:
                    tmp_pixels = Normalize.get_norm(
                        pixels=tmp_pixels, norm_type=norm_type, timing=timing, bins=bins
                    )[0]
                histograms.append(mh.fullhistogram(tmp_pixels.astype(np.uint8)))
        else:
            tmp_pixels = pixels.copy()
            if norm_type is not None:
                tmp_pixels = Normalize.get_norm(
                    tmp_pixels, norm_type=norm_type, timing=timing, bins=bins
                )[0]
            histograms = mh.fullhistogram(tmp_pixels.astype(np.uint8))
        if timing:
            logging.info(f"Time to get histogram: {time.time() - t0}")
        return np.array(histograms)

    @staticmethod
    def get_features(
        data: Union[SimpleNamespace, np.ndarray],
        feature_type: str = "hist",
        norm_type: str = None,
        bins: int = 256,
        timing: bool = False,
    ) -> np.ndarray:
        """Get features of data

        Parameters
        ----------
        data : SimpleNamespace, np.ndarray
            Array of pixels
        feature_type : str, optional
            Type of feature to extract. The default is "histogram".
        norm_type : str, optional
            Type of normalization. The default is None.
        bins : int, optional
            Number of bins to use for normalization. The default is 256.
        timing: bool, optional
            Whether to time the function. The default is False.

        Returns
        -------
        np.ndarray
        """
        logging.info("Getting features")
        t0 = time.time()
        if feature_type in {"hist", "histogram"}:
            features = Features.histogram(
                data, norm_type=norm_type, timing=timing, bins=bins
            )
        else:
            raise ValueError("Feature type not supported")
        if timing:
            logging.info(f"Time to get features: {time.time() - t0}")
        return features
