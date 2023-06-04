import copy
import time
from types import SimpleNamespace
from .normalizations import Normalize
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np


class Features:

    @staticmethod
    def histogram(pixels, norm_type=None, timing=False, **kwargs):
        """Create histogram of data
        Returns
        -------
        np.ndarray
            List of histograms
        """
        t0 = time.time()
        histograms = []
        # if pixels is a list of images get list of histograms for each image
        if isinstance(pixels, list):
            for i in range(len(pixels)):
                # if pixels is a list of SimpleNamespace use the pixels attribute
                if isinstance(pixels[i], SimpleNamespace):
                    tmp_pixels = pixels[i].pixels.copy()
                else:  # else assume pixels is a list of np.ndarray
                    tmp_pixels = pixels[i].copy()
                # if normalization is specified normalize pixels before histogram
                if norm_type is not None:
                    tmp_pixels = Normalize.get_norm(tmp_pixels,
                                                    norm_type=norm_type,
                                                    timing=timing,
                                                    **kwargs)[0]
                # append histogram to list
                histograms.append(mh.fullhistogram(tmp_pixels.astype(np.uint8)))
        # if pixels is a single image get histogram
        else:
            tmp_pixels = pixels.copy()
            if norm_type is not None:
                tmp_pixels = Normalize.get_norm(tmp_pixels,
                                                norm_type=norm_type,
                                                timing=timing,
                                                **kwargs)[0]
            histograms = mh.fullhistogram(tmp_pixels.astype(np.uint8))

        if timing:
            print("Histogram: ", time.time() - t0)
        return histograms

    @staticmethod
    def orb(imgs,
            norm_type=None,
            timing=False,
            downsample=True,
            return_pixel_values=True,
            **kwargs):
        """Create ORB features of data
        Parameters:
        ----------
        imgs: np.ndarray | list of np.ndarray | SimpleNamespace
            array of pixels
        norm_type : str, optional
            Type of normalization. The default is minmax.
        downsample: bool, optional
            Whether to downsample the image to 128x128. The default is True.
        return_pixel_values: bool, optional
            Whether to return the keypoints or pixel values. The default is
            True.
        timing: bool, optional
            Whether to time the function. The default is False.
        **kwargs : dict
            Additional arguments for ORB
        Parameters passable through kwargs:
        ----------
        n_keypoints: int, optional
            Number of keypoints to be returned. The function will return the
            best n_keypoints according to the Harris corner response if more
            than n_keypoints are detected. If not, then all the detected
            keypoints are returned.
        fast_n: int, optional
            The n parameter in skimage.feature.corner_fast. Minimum number of
            consecutive pixels out of 16 pixels on the circle that should all be
            either brighter or darker w.r.t test-pixel. A point c on the circle
            is darker w.r.t test pixel p if Ic < Ip - threshold and brighter
            if Ic > Ip + threshold. Also stands for the n in FAST-n corner
            detector.
        fast_threshold: float, optional
            The threshold parameter in feature.corner_fast. Threshold used to
            decide whether the pixels on the circle are brighter, darker or
            similar w.r.t. the test pixel. Decrease the threshold when more
            corners are desired and vice-versa.
        harris_k: float, optional
            The k parameter in skimage.feature.corner_harris. Sensitivity factor
            to separate corners from edges, typically in range [0, 0.2]. Small
            values of k result in detection of sharp corners.
        downscale: float, optional
            Downscale factor for the image pyramid. Default value 1.2 is chosen
            so that there are more dense scales which enable robust scale
            invariance for a subsequent feature description.
        n_scales: int, optional
            Maximum number of scales from the bottom of the image pyramid to
            extract the features from.
        timing: bool, optional
            Whether to time the function. The default is False.
        Returns
        -------
        np.ndarray | list of np.ndarray
            List of keypoints for each image or list of pixel values for each
            image if return_pixel_values is True.
        """
        t0 = time.time()

        from skimage.feature import ORB

        n_keypoints = kwargs.get('n_keypoints', 50)
        fast_n = kwargs.get('fast_n', 9)
        fast_threshold = kwargs.get('fast_threshold', 0.08)
        harris_k = kwargs.get('harris_k', 0.04)
        downscale = kwargs.get('downscale', 1.2)
        n_scales = kwargs.get('n_scales', 8)
        output_shape = kwargs.get('output_shape', (512, 512))

        pixels = Normalize.extract_pixels(imgs)
        if downsample:
            pixels = Normalize.downsample(pixels, output_shape=output_shape)[0]

        if norm_type is not None:
            pixels = Normalize.get_norm(pixels,
                                        norm_type=norm_type,
                                        timing=timing,
                                        **kwargs)[0]

        descriptor_extractor = ORB(n_keypoints=n_keypoints,
                                   fast_n=fast_n,
                                   fast_threshold=fast_threshold,
                                   harris_k=harris_k,
                                   downscale=downscale,
                                   n_scales=n_scales)
        keypoints = []
        if isinstance(pixels, list) or isinstance(pixels, np.ndarray):
            for i in range(len(pixels)):
                descriptor_extractor.detect_and_extract(pixels[i])
                keypoints.append(descriptor_extractor.keypoints)
        else:
            descriptor_extractor.detect_and_extract(pixels)
            keypoints.append(descriptor_extractor.keypoints)

        if return_pixel_values:
            intensities = []
            for i in range(len(keypoints)):
                intensities.append(
                    Features._extract_pixel_intensity_from_keypoints(
                        keypoints[i], pixels[i]))
            keypoints = intensities

        if timing:
            print("ORB: {:.2f} s".format(time.time() - t0))
        return keypoints

    @staticmethod
    def sift(imgs,
             norm_type=None,
             timing=False,
             downsample=True,
             return_pixel_values=True,
             **kwargs
             ):
        """Create SIFT features of data
        Parameters:
        ----------
        imgs : np.ndarray | list of np.ndarray | SimpleNamespace
            array of pixels
        norm_type : str, optional
            Type of normalization. The default is minmax.
        downsample : bool, optional
            Downsample image. The default is False.
        return_pixel_values : bool, optional
            Return pixel values. The default is True.
        timing : bool, optional
            Print timing. The default is False.
        **kwargs : dict
            Additional arguments for sift
        Parameters passable through kwargs:
        ----------
        upsampling: int, optional
            Prior to the feature detection the image is upscaled by a factor of
            1 (no upscaling), 2 or 4. Method: Bi-cubic interpolation.
        n_octaves: int, optional
            Maximum number of octaves. With every octave the image size is
            halved and the sigma doubled. The number of octaves will be
            reduced as needed to keep at least 12 pixels along each dimension
            at the smallest scale.
        n_scales: int, optional
            Maximum number of scales in every octave.
        sigma_min: float, optional
            The blur level of the seed image. If upsampling is enabled
            sigma_min is scaled by factor 1/upsampling
        sigma_in: float, optional
            The assumed blur level of the input image.
        c_dog: float, optional
            Threshold to discard low contrast extrema in the DoG. It’s final
            value is dependent on n_scales by the relation:
            final_c_dog = (2^(1/n_scales)-1) / (2^(1/3)-1) * c_dog
        c_edge: float, optional
            Threshold to discard extrema that lie in edges. If H is the Hessian
            of an extremum, its “edgeness” is described by tr(H)²/det(H).
            If the edgeness is higher than (c_edge + 1)²/c_edge, the extremum
            is discarded.
        n_bins: int, optional
            Number of bins in the histogram that describes the gradient
            orientations around keypoint.
        lambda_ori: float, optional
            The window used to find the reference orientation of a keypoint has
            a width of 6 * lambda_ori * sigma and is weighted by a standard
            deviation of 2 * lambda_ori * sigma.
        c_max: float, optional
            The threshold at which a secondary peak in the orientation histogram
            is accepted as orientation
        lambda_descr: float, optional
            The window used to define the descriptor of a keypoint has a width
            of 2 * lambda_descr * sigma * (n_hist+1)/n_hist and is weighted by
            a standard deviation of lambda_descr * sigma.
        n_hist: int, optional
            The window used to define the descriptor of a keypoint consists of
            n_hist * n_hist histograms.
        n_ori: int, optional
            The number of bins in the histograms of the descriptor patch.
        timing: bool, optional
            Whether to time the function. The default is False.
        Returns
        -------
        np.ndarray | list of np.ndarray
            List of keypoints for each image or list of pixel values for each
            image if return_pixel_values is True.
        """
        t0 = time.time()

        from skimage.feature import SIFT

        upsampling = kwargs.get('upsampling', 1)
        n_octaves = kwargs.get('n_octaves', 1)
        n_scales = kwargs.get('n_scales', 1)
        sigma_min = kwargs.get('sigma_min', 1.3)
        sigma_in = kwargs.get('sigma_in', .5)
        c_dog = kwargs.get('c_dog', .7)
        c_edge = kwargs.get('c_edge', .05)
        n_bins = kwargs.get('n_bins', 10)
        lambda_ori = kwargs.get('lambda_ori', .5)
        c_max = kwargs.get('c_max', 1.5)
        lambda_descr = kwargs.get('lambda_descr', .5)
        n_hist = kwargs.get('n_hist', 1)
        n_ori = kwargs.get('n_ori', 1)
        output_shape = kwargs.get('output_shape', (256, 256))

        pixels = Normalize.extract_pixels(imgs)
        if downsample:
            pixels = Normalize.downsample(pixels, output_shape=output_shape)[0]

        if norm_type is not None:
            pixels = Normalize.get_norm(pixels,
                                        norm_type=norm_type,
                                        timing=timing,
                                        **kwargs)[0]

        descriptor_extractor = SIFT(upsampling=upsampling,
                                    n_octaves=n_octaves,
                                    n_scales=n_scales,
                                    sigma_min=sigma_min,
                                    sigma_in=sigma_in,
                                    c_dog=c_dog,
                                    c_edge=c_edge,
                                    n_bins=n_bins,
                                    lambda_ori=lambda_ori,
                                    c_max=c_max,
                                    lambda_descr=lambda_descr,
                                    n_hist=n_hist,
                                    n_ori=n_ori)

        keypoints = []
        if isinstance(pixels, list) or isinstance(pixels, np.ndarray):
            for i in range(len(pixels)):
                descriptor_extractor.detect_and_extract(pixels[i])
                keypoints.append(descriptor_extractor.keypoints)
        else:
            descriptor_extractor.detect_and_extract(pixels)
            keypoints.append(descriptor_extractor.keypoints)

        if return_pixel_values:
            intensities = []
            for i in range(len(keypoints)):
                intensities.append(
                    Features._extract_pixel_intensity_from_keypoints(
                        keypoints[i], pixels[i]))
            intensities = Features._fix_jagged_keypoint_arrays(intensities)
            keypoints = intensities

        if timing:
            print('SIFT: {:.2f} s'.format(time.time() - t0))

        return keypoints

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
        normalize : str | bool
            If not None, the images are normalized using the given method
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
            output_shape = (128, 128)
        pixels = Normalize.extract_pixels(images)
        if isinstance(pixels, list):
            resized = []
            for img in pixels:
                resized.append(resize(img, output_shape))
        else:
            resized = resize(pixels, output_shape)

        if flatten:
            resized = [img.flatten() for img in resized]

        if normalize is not None:
            resized = Normalize.get_norm(resized,
                                         norm_type=normalize,
                                         timing=timing,
                                         **kwargs)[0]

        if timing:
            print("downsample: {}".format(time.time() - t0))
        return resized

    @staticmethod
    def _extract_pixel_intensity_from_keypoints(keypoints, img):
        """Extract pixel intensities from keypoints
        Parameters
        ----------
        keypoints : np.ndarray
            array of keypoints
        img : np.ndarray
            image to extract intensities from
        Returns
        -------
        np.ndarray
            array of pixel intensities
        """
        intensities = []
        for i in range(len(keypoints)):
            x = int(keypoints[i][0])
            y = int(keypoints[i][1])
            intensities.append(img[x, y])
        return np.array(intensities)

    @staticmethod
    def _fix_jagged_keypoint_arrays(keypoints):
        """Normalize keypoint lengths by finding the smallest length of
        keypoints and then selecting a random distribution of keypoints of
        similar length in the rest of the keypoints.
        Parameters
        ----------
        keypoints : list of np.ndarray
            list of keypoints
        Returns
        -------
        np.ndarray
            array of keypoints
        """
        min_len = min([len(kp) for kp in keypoints])
        new_keypoints = []
        for kp in keypoints:
            if len(kp) > min_len:
                new_keypoints.append(
                    kp[np.random.choice(len(kp), min_len, replace=False)])
            else:
                new_keypoints.append(kp)
        return np.array(new_keypoints)

    @staticmethod
    def get_features(data,
                     feature_type="hist",
                     norm_type=None,
                     timing=False,
                     **kwargs):
        """Get features of data
        Parameters
        ----------
        data : SimpleNamespace, np.ndarray, list of np.ndarray, any
            array of pixels
        feature_type : str, optional
            Type of feature to extract. The default is "histogram".
        norm_type : str, optional
            Type of normalization. The default is None.
        timing: bool, optional
            Whether to time the function. The default is False.
        Returns
        -------
        np.ndarray, ski.feature.FeatureDetector
        """
        t0 = time.time()
        if feature_type == "hist" or feature_type == "histogram":
            features = Features.histogram(data,
                                          norm_type=norm_type,
                                          timing=timing,
                                          **kwargs)
        elif feature_type == "sift":
            rpv = kwargs.get('return_pixel_values', True)
            ds = kwargs.get('downsample', False)
            features = Features.sift(data,
                                     norm_type=norm_type,
                                     return_pixel_values=rpv,
                                     downsample=ds,
                                     timing=timing,
                                     **kwargs)
        elif feature_type == "orb":
            rpv = kwargs.get('return_pixel_values', True)
            ds = kwargs.get('downsample', True)
            features = Features.orb(data,
                                    norm_type=norm_type,
                                    return_pixel_values=rpv,
                                    downsample=ds,
                                    timing=timing,
                                    **kwargs)
        elif feature_type == "downsample":
            output_shape = kwargs.get('output_shape', (256, 256))
            flatten = kwargs.get('flatten', True)
            features = Normalize.downsample(data,
                                            output_shape=output_shape,
                                            flatten=flatten,
                                            norm_type=norm_type,
                                            timing=timing,
                                            **kwargs)[0]
        else:
            raise ValueError("Feature type not supported")
        if timing:
            print('Features: ', time.time() - t0)
        return features

    @staticmethod
    def show_image_and_feature(image,
                               features=None,
                               feature_types=None,
                               norm_type='min-max',
                               downsample=False,
                               output_shape=None,
                               train_scores=None,
                               label=None,
                               log=False,
                               **kwargs):
        """Displays an image next to the specified features.
        Parameters
        ----------
        image : SimpleNamespace
            array of pixel values
        features : list
            (default is None) list of features to display
        feature_types : list
            (default is 'hist') type of feature to display
        norm_type : str
            type of normalization to use, options are:
            'min-max' : normalize the image using the min and max values
            'max' : normalize the image using the max value
            'guassian' : normalize the image using a guassian distribution
            'z-score' : normalize the image using a z-score
            'robost' : normalize the image using a robust distribution
            'downsample' : downsample the image to 64x64
            (default is 'min-max')
        downsample : bool
            (default is False) downsample the image to 64x64 or to the shape
            specified by output_shape
        output_shape : tuple
            (default is None) shape of the output image
        train_scores : list
            (default is None) train scores of the features
        label : str
            (default is None) label of the image
        log : bool
            (default is False) whether to log the image
        """
        if feature_types is None:
            feature_types = []

        pixels = Normalize.extract_pixels(image)

        if output_shape is not None:
            downsample = True

        # normalize the image
        img = Normalize.get_norm(pixels,
                                 norm_type=norm_type,
                                 downsample=downsample,
                                 output_shape=output_shape,
                                 **kwargs)

        fig, ax = plt.subplots(1,
                               len(feature_types) + 1,
                               figsize=(10, 5))
        if label is not None:
            fig.suptitle(
                'SOPInstanceUID: ' + image.SOPInstanceUID + ' ' + 'Label: ' + label)
        else:
            fig.suptitle('SOPInstanceUID: ' + image.SOPInstanceUID)
        # add extra width between plots
        fig.subplots_adjust(wspace=0.4)

        ax[0].imshow(img, cmap='gray')
        # ax[0].set_title(image.SOPInstanceUID, size=8-len(feature_types))

        if 'sift' in feature_types:
            idx = feature_types.index('sift') + 1
            if features is None:
                kp = Features.sift(img)
                keypoints = kp.keypoints
            else:
                keypoints = features[idx - 1]
            ax[idx].imshow(img)
            x_points = keypoints[:, 1]
            y_points = keypoints[:, 0]
            ax[idx].scatter(x_points, y_points, facecolors='none',
                            edgecolors='r')
            label = Features._get_train_score('sift',
                                              feature_types,
                                              train_scores)
            ax[idx].set_title(label, size=8)

        if 'orb' in feature_types:
            idx = feature_types.index('orb') + 1
            if features is None:
                kp = Features.orb(img)
                keypoints = kp.keypoints
            else:
                keypoints = features[idx - 1]
            keypoints = keypoints[0].astype(int)
            img_ds = Normalize.downsample(img)
            ax[idx].imshow(img_ds[0])
            x_points = keypoints[:, 1]
            y_points = keypoints[:, 0]
            ax[idx].scatter(x_points, y_points, facecolors='none',
                            edgecolors='r')
            label = Features._get_train_score('orb',
                                              feature_types,
                                              train_scores)
            ax[idx].set_title(label, size=8)

        if 'hist' in feature_types or 'histogram' in feature_types:
            idx = feature_types.index('hist') + 1
            if features is None:
                y_axis = Features.histogram(img)
                idx = feature_types.index('hist') + 1
                y_axis = Features.histogram(img)
                if len(y_axis) < 256:
                    y_axis = np.append(y_axis, np.zeros(256 - len(y_axis)))
                x_axis = np.arange(0, 256, 1)
                ax[idx].set_ylim(0, 1)
                ax[idx].bar(x_axis, y_axis, color='b', log=True, width=10)
                ax[idx].set_xlim(.01, 255)
                ax[idx].set_ylim(.01, 10 ** 8)
                label = Features._get_train_score('hist',
                                                  feature_types,
                                                  train_scores)
                ax[idx].set_title(label, size=8)
            else:
                y_axis = features[idx - 1]
                print(y_axis)
                ax[idx].set_ylim(0, np.max(y_axis))
                ax[idx].bar(np.arange(0, len(y_axis)), y_axis, log=log)
                label = Features._get_train_score('hist',
                                                  feature_types,
                                                  train_scores)
                ax[idx].set_title(label, size=8)

        if 'downsample' in feature_types:
            idx = feature_types.index('downsample') + 1
            if features is None:
                img_ds = Normalize.downsample(img)
            else:
                img_ds = features[idx - 1]
            ax[idx].imshow(img_ds[0], cmap='gray')
            label = Features._get_train_score('downsample',
                                              feature_types,
                                              train_scores)
            ax[idx].set_title(label, size=8)

        plt.show()

    @staticmethod
    def view_image_and_features(images,
                                feature_types=None,
                                norm_type='min-max',
                                train_scores=None):
        """Displays an image next to its histogram.
        Parameters
        ----------
        images : list
            list of SimpleNamespace
        feature_types : list
            (default is 'hist') type of feature to display
        feature_labels : list
            (default is 'Histogram') label of the feature
        norm_type : str
            type of normalization to use, options are:
            'min-max' : normalize the image using the min and max values
            'max' : normalize the image using the max value
            'guassian' : normalize the image using a guassian distribution
            'z-score' : normalize the image using a z-score
            'robost' : normalize the image using a robust distribution
            (default is 'min-max')
        train_scores : list of lists
            (default is None) train scores of the features
        """
        # add the images and train scores a image list
        if feature_types is None:
            feature_types = []
        images_list = []
        for i in range(len(images)):
            ds = SimpleNamespace()
            ds.image = images[i]
            ds.train_scores = train_scores[i]
            images_list.append(ds)

        # loop over the images and use the show image and feature function
        for i in range(len(images)):
            ds = images_list[i]
            Features.show_image_and_feature(ds.image,
                                            feature_types=feature_types,
                                            norm_type=norm_type,
                                            train_scores=ds.train_scores)

    @staticmethod
    def _get_train_score(feature, feature_types, train_scores):
        """Get train scores of features
            Parameters
            ----------
            feature : str
                feature to get train scores of
            feature_types : list
               type of features to display
            train_scores : list
                (default is None) train scores of the feature
            Returns
            -------
            list
                train scores of the features
            """
        if train_scores is None:
            return feature
        if len(train_scores) == 1:
            return str(train_scores[0])
        else:
            return str(train_scores[feature_types.index(feature)])
