import time
import types
import logging
import pydicom as dicom
import numpy as np
from typing import Union, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class Normalize:
    """
    Class to represent different normalization techniques
    """

    @staticmethod
    def extract_pixels(
        images: Union[np.ndarray, List[np.ndarray]], timing: bool = False
    ) -> List[np.ndarray]:
        """
        Extract pixels from images

        Parameters
        ----------
        images : Union[np.ndarray, List[np.ndarray]]
            Array of images to be normalized
        timing : bool, optional
            If true, the time needed to perform the normalization is printed.
            The default is False.

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            Extracted pixels
        """
        t0 = time.time()
        pixels = []
        if isinstance(images, list):
            if isinstance(images[0], np.ndarray):
                return images
            elif isinstance(images[0], types.SimpleNamespace):
                for image in images:
                    pixels.append(image.pixels)
            elif isinstance(images[0], dicom.dataset.FileDataset):
                for image in images:
                    pixels.append(image.pixel_array)
            else:
                raise TypeError("Unknown type of images")
        elif isinstance(images, np.ndarray):
            pixels = images  # was returning this as list before
        elif isinstance(images, types.SimpleNamespace):
            pixels = images.pixels
        else:
            raise TypeError("Unknown type of images")
        if timing:
            print("Extract pixels: {}".format(time.time() - t0))
        return pixels

    @staticmethod
    def _minmax_helper(pixels: np.ndarray, **kwargs) -> np.ndarray:
        """
        Helper function to normalize data using minmax method

        Parameters
        ----------
        pixels : Union[np.ndarray, List[np.ndarray]]
            Array of pixels to be normalized

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            Normalized pixels
        """
        bins = kwargs.get("bins", 256)
        max_val = np.max(pixels)
        min_val = np.min(pixels)
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels -= min_val
        normalized_pixels /= max_val - min_val
        normalized_pixels *= bins - 1
        return normalized_pixels

    @staticmethod
    def minmax(
        pixels: Union[np.ndarray, List[np.ndarray]], timing: bool = False, **kwargs
    ) -> Tuple[List[np.ndarray], None]:
        """
        The min-max approach (often called normalization) rescales the
        feature to a fixed range of [0,1] by subtracting the minimum value
        of the feature and then dividing by the range, which is then multiplied
        by 255 to bring the value into the range [0,255].

        Parameters
        ----------
        pixels : Union[np.ndarray, List[np.ndarray]]
            Array of pixels to be normalized
        timing : bool
            If true, the time needed to perform the normalization is printed
            The default is False.
        kwargs : dict
            Additional parameters to be passed to the normalization function.

        Returns
        -------
        Tuple[List[np.ndarray], Optional[Any]]
            Normalized pixels and None.
        """
        t0 = time.time()
        if isinstance(pixels, list):
            normalized_pixels = []
            for p in pixels:
                normalized_pixels.append(Normalize._minmax_helper(p, **kwargs))
        else:
            normalized_pixels = Normalize._minmax_helper(pixels, **kwargs)

        if timing:
            print("minmax: {}".format(time.time() - t0))
        return normalized_pixels, None

    @staticmethod
    def get_norm(
        pixels: Union[np.ndarray, List[np.ndarray]],
        norm_type: str,
        timing: bool = False,
        **kwargs,
    ) -> Tuple[List[np.ndarray], Optional[Any]]:
        """
        Normalize pixels

        Parameters
        ----------
        pixels : Union[np.ndarray, List[np.ndarray]]
            Array of pixels to be normalized
        norm_type : str
            Type of normalization. Options: 'minmax', 'min-max'.
        timing : bool, optional
            If true, the time needed to perform the normalization is printed.
            The default is False.
        kwargs : dict
            Additional parameters to be passed to the normalization function.

        Returns
        -------
        Tuple[List[np.ndarray], Optional[Any]]
            The first element is the normalized pixels.
            The second element is None.
        """
        t0 = time.time()
        pixels = Normalize.extract_pixels(pixels)
        if norm_type.lower() == "minmax" or norm_type.lower() == "min-max":
            normalized, filtered = Normalize.minmax(pixels, timing, **kwargs)
        else:
            raise ValueError("Invalid normalization type")

        if timing:
            print("get_norm: {}".format(time.time() - t0))

        return normalized, filtered
