import time
import types
import pydicom as dicom
import numpy as np
import mahotas as mh

DEBUG = False


# class to represent different normalization techniques

class Normalize:
    @staticmethod
    def extract_pixels(images, timing=False):
        """Extract pixels from images
        Parameters
        ----------
        images : numpy.ndarray | list of numpy.ndarray
            Array of images to be normalized
        timing : bool, optional
            If true, the time needed to perform the normalization is printed.
            The default is False.
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
    def _minmax_helper(pixels, **kwargs):
        """Helper function to normalize data using minmax method
        """
        bins = kwargs.get("bins", 256)
        max_val = np.max(pixels)
        min_val = np.min(pixels)
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels -= min_val
        normalized_pixels /= (max_val - min_val)
        normalized_pixels *= bins - 1

        return normalized_pixels

    @staticmethod
    def minmax(pixels,
               timing=False,
               **kwargs):
        """The min-max approach (often called normalization) rescales the
        feature to a fixed range of [0,1] by subtracting the minimum value
        of the feature and then dividing by the range, which is then multiplied
        by 255 to bring the value into the range [0,255].
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
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
    def _max_helper(pixels, **kwargs):
        """Helper function to normalize data using max method
        """
        bins = kwargs.get("bins", 256)
        max_val = np.max(pixels)
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels /= abs(max_val)
        normalized_pixels *= bins - 1
        return normalized_pixels

    @staticmethod
    def max(pixels,
            timing=False,
            **kwargs):
        """The maximum absolute scaling rescales each feature between -1 and 1
        by dividing every observation by its maximum absolute value.
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        if isinstance(temp_pixels, list):
            normalized_pixels = []
            for p in temp_pixels:
                normalized_pixels.append(Normalize._max_helper(p, **kwargs))
        else:
            normalized_pixels = Normalize._max_helper(temp_pixels, **kwargs)

        if timing:
            print("max: {}".format(time.time() - t0))
        return normalized_pixels, None

    @staticmethod
    def _gaussian_helper(pixels, **kwargs):
        """Helper function to normalize data using gaussian blur
        """
        sigma = kwargs.get("sigma", 20)
        normalized_pixels = mh.gaussian_filter(pixels, sigma=sigma)
        normalized_pixels /= normalized_pixels.max()
        return normalized_pixels

    @staticmethod
    def gaussian(pixels,
                 timing=False,
                 **kwargs):
        """Normalize by gaussian
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        bins = kwargs.get("bins", 256)
        if isinstance(temp_pixels, list):
            filtered = []
            for i, p in enumerate(temp_pixels):
                filtered.append(Normalize._gaussian_helper(p, **kwargs))
        else:
            filtered = Normalize._gaussian_helper(temp_pixels, **kwargs)

        normalized_pixels = filtered.copy()
        normalized_pixels *= bins - 1

        if timing:
            print("gaussian: {}".format(time.time() - t0))
        return normalized_pixels, filtered

    @staticmethod
    def _zscore_helper(pixels, **kwargs):
        """Helper function to normalize data using zscore method
        """
        bins = kwargs.get("bins", 255)
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels -= np.mean(normalized_pixels)
        normalized_pixels /= np.std(normalized_pixels)
        normalized_pixels *= bins

        return normalized_pixels

    @staticmethod
    def z_score(pixels,
                timing=False,
                **kwargs):
        """The z-score method (often called standardization) transforms the data
        into a distribution with a mean of 0 and a standard deviation of 1.
        Each standardized value is computed by subtracting the mean of the
        corresponding feature and then dividing by the standard deviation.
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        if isinstance(temp_pixels, list):
            normalized_pixels = []
            for p in temp_pixels:
                normalized_pixels.append(Normalize._zscore_helper(p, **kwargs))
        else:
            normalized_pixels = Normalize._zscore_helper(temp_pixels, **kwargs)

        if timing:
            print("zscore: {}".format(time.time() - t0))
        return normalized_pixels, None

    @staticmethod
    def _robust_helper(pixels, **kwargs):
        """
        Robust Scalar transforms x to x’ by subtracting each value of features
        by the median and dividing it by the interquartile range between the
        1st quartile (25th quantile) and the 3rd quartile (75th quantile).
        """
        bins = kwargs.get("bins", 256)
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels -= np.median(normalized_pixels)
        normalized_pixels /= np.percentile(normalized_pixels,
                                           75) - np.percentile(
            normalized_pixels, 25)
        normalized_pixels *= bins - 1

        return normalized_pixels

    @staticmethod
    def robust(pixels,
               timing=False,
               **kwargs):
        """In robust scaling, we scale each feature of the data set by subtracting
        the median and then dividing by the interquartile range. The interquartile
        range (IQR) is defined as the difference between the third and the first
        quartile and represents the central 50% of the data.
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        if isinstance(temp_pixels, list):
            normalized_pixels = []
            for p in temp_pixels:
                normalized_pixels.append(Normalize._robust_helper(p, **kwargs))
        else:
            normalized_pixels = Normalize._robust_helper(temp_pixels, **kwargs)

        if timing:
            print("robust: {}".format(time.time() - t0))

        return normalized_pixels, None

    @staticmethod
    def downsample(images,
                   output_shape=None,
                   flatten=False,
                   normalize=None,
                   timing=False,
                   **kwargs):
        """Downsample images to a given shape.
        Parameters
        ----------
        images : numpy.ndarray, list of numpy.ndarray
            Array of images to be downsampled
        output_shape : tuple
            Shape of the output images
        flatten : bool
            If true, the images are flattened to a 1D array
        normalize : str
            The type of normalization to perform on the images
        timing : bool
            If true, the time needed to perform the downsampling is printed
        Returns
        -------
        numpy.ndarray | list of numpy.ndarray
            Downsampled images
        """
        t0 = time.time()
        from skimage.transform import resize
        if output_shape is None:
            output_shape = (256, 256)
        images_copy = Normalize.extract_pixels(images)
        if isinstance(images_copy, list):
            resized = []
            for img in images_copy:
                if flatten:
                    resized.append(
                        np.array(resize(img, output_shape)).reshape(-1))
                else:
                    resized.append(np.array(resize(img, output_shape)))
                if timing:
                    print("downsample: {}".format(time.time() - t0))
        else:
            if flatten:
                resized = np.array(resize(images_copy, output_shape)).reshape(
                    -1)
            else:
                resized = np.array(resize(images_copy, output_shape))

        if normalize is not None:
            if isinstance(normalize, bool):
                normalize = "minmax"
            if isinstance(resized, list):
                for i in range(len(resized)):
                    resized[i] = Normalize.get_norm(resized[i],
                                                    normalize)[0]
            else:
                resized = Normalize.get_norm(resized, normalize)[0]

        if timing:
            print("downsample: {}".format(time.time() - t0))
        return np.asarray(resized), None

    @staticmethod
    def get_norm(pixels,
                 norm_type,
                 timing=False,
                 **kwargs):
        """Normalize pixels
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        norm_type : str
            Type of normalization. The default is 'min-max'.
            options -> attributes possible:
                'min-max',
                'max',
                'gaussian',
                'z-score',
                'robust',
                'downsample' -> output_shape
        timing : bool, optional
            If true, the time needed to perform the normalization is printed.
            The default is False.
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        pixels = Normalize.extract_pixels(pixels)
        if norm_type.lower() == 'max':
            normalized, filtered = Normalize.max(pixels,
                                                 timing,
                                                 **kwargs)
        elif norm_type.lower() == 'minmax' or norm_type.lower() == 'min-max':
            normalized, filtered = Normalize.minmax(pixels,
                                                    timing,
                                                    **kwargs)
        elif norm_type.lower() == 'gaussian':
            normalized, filtered = Normalize.gaussian(pixels,
                                                      timing,
                                                      **kwargs)
        elif norm_type.lower() == 'zscore' or norm_type.lower() == 'z-score':
            normalized, filtered = Normalize.z_score(pixels,
                                                     timing,
                                                     **kwargs)
        elif norm_type.lower() == 'robust':
            output_shape = kwargs.get('output_shape', (256, 256))
            normalized, filtered = Normalize.robust(pixels,
                                                    timing,
                                                    **kwargs)
        elif norm_type.lower() == 'downsample':
            output_shape = kwargs.get('output_shape', (256, 256))
            flatten = kwargs.get('flatten', True)
            normalized, filtered = Normalize.downsample(pixels,
                                                        output_shape=output_shape,
                                                        flatten=flatten,
                                                        normalize=norm_type,
                                                        **kwargs)
        else:
            raise ValueError('Invalid normalization type')

        if timing:
            print("get_norm: {}".format(time.time() - t0))

        return normalized, filtered
