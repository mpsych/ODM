import glob
import io
import os
import pickle
import time

import numpy as np

from loaders import omama_loader as O
from data import Data
import pydicom as dicom
import matplotlib as mpl
import matplotlib.pyplot as plt
from types import SimpleNamespace
from PIL import Image


class DataHelper(object):
    @staticmethod
    def check_data_instance(config_num=2, timing=False):
        """Checks if the data instance is already created. If not, creates one.
        Parameters
        ----------
        config_num : int
            The option number corresponds to a set of paths to specific data
            which is loaded into the data instance. config_num options are:
            1: Full Omama dataset
            2: Omama dataset for 2D whitelist
            3: Omama dataset for 3D whitelist
            4: Omama dataset for 2D + 3D WL
        timing : bool
            If True, the time needed to create the data instance is printed.
        Returns
        -------
        data : Data
            The data instance.
        """
        t0 = time.time()
        if Data.instance is None:
            loader = O.OmamaLoader(config_num=config_num)
            Data(loader, load_cache=True)
            data = Data.instance
        else:
            data = Data.instance
        if timing:
            Data.timing(t0, 'check_data_instance')
        return data

    # --------------------------------------------------------------------------
    @staticmethod
    def check_type(image, data, timing=False):
        """Checks if the image is a numpy array or a PIL image.
        Parameters
        ----------
        image : int, str, SimpleNamespace
            The image to check.
        data : Data
            The data object to use.
        timing : bool
            If True, the time needed to check the type is printed.
        Returns
        --------
        Path : str
            The path to the image.
        """
        t0 = time.time()
        if isinstance(image, str) and '/' in image:
            # print('in check_type, image is a string')
            # if image ends in .txt, it is a whitelist
            if image.endswith('.txt'):
                # print('in check_type, image is a whitelist')
                with open(image, 'r') as f:
                    path_list = [line.rstrip() for line in f]
                # print the type of path_list print('in check_type, type of
                # path_list is: ', type(path_list))
                path = path_list
            else:
                path = image
        elif isinstance(image, int):
            # print('in check_type, image is an int')
            path = data.path(image_id=image)
        elif isinstance(image, str):
            # print('in check_type, image is a string')
            path = data.path(dicom_name=image)
        # check to see if image is a simple namespace object
        elif isinstance(image, SimpleNamespace):
            # print('in check_type, image is a simple namespace')
            path = image.filePath
        elif isinstance(image, dicom.dataset.FileDataset):
            # print('in check_type, image is a dicom dataset')
            path = image
        elif isinstance(image, np.ndarray):
            # print('in check_type, image is a numpy array')
            path = image
        elif isinstance(image, Data):
            # print('in check_type, image is a data object')
            path = image
        else:
            return 'Please specify either a path, image_id, dicom_name, ' \
                   'or a image object'
        if timing:
            Data.timing(t0, 'check_type')
        return path

    @staticmethod
    def get_pixels(image, data):
        """Gets the pixels from the image.
        Parameters
        ----------
        image : any
            The image to get the pixels from.
        Returns
        --------
        pixels : numpy.ndarray
            The pixels of the image.
        """
        if isinstance(image, np.ndarray):
            # print('image is a numpy array')
            return image
        elif isinstance(image, SimpleNamespace):
            # print('image is a simple namespace object')
            return image.pixels
        elif isinstance(image, dicom.dataset.FileDataset):
            # print('image is a dicom dataset')
            return image.pixel_array
        elif isinstance(image, list):
            # print('image is a list')
            if isinstance(image[0], np.ndarray):
                # print('image is a list of numpy arrays')
                return image
            elif isinstance(image[0], SimpleNamespace):
                # print('image is a list of simple namespace objects')
                pixels = []
                for i in image:
                    pixels.append(i.pixels)
                return np.array(pixels)
            elif isinstance(image[0], dicom.dataset.FileDataset):
                # print('image is a list of dicom datasets')
                pixels = []
                for i in image:
                    pixels.append(i.pixel_array)
                return np.array(pixels)
            elif isinstance(image[0], str):
                # print('image is a list of paths')
                pixels = []
                for i in image:
                    pixels.append(O.DataHelper.get_pixels(i, data))
                return np.array(pixels)
            else:
                return 'Please specify either a path, image_id, dicom_name, ' \
                       'or a image object'
        else:
            path = DataHelper.check_type(image, data)

        if isinstance(path, str):
            # print('Loading image from path using dicom...')
            image = dicom.read_file(path, force=True)
            return image.pixel_array
        elif isinstance(path, Data):
            # print('Loading image from data object...')
            pixels = []
            for d in path:
                pixels.append(d.pixel_array)
            return pixels
        elif isinstance(path, list):
            # print('Loading image from list of paths...')
            pixels = []
            for p in path:
                pixels.append(O.DataHelper.get_pixels(p, data))
            return pixels
        else:
            raise ValueError('Pixel array not found')

    # --------------------------------------------------------------------------
    @staticmethod
    def _add_bounding_box(image, pred, sop_uid):
        """Adds a bounding box to the image.
        Parameters
        ----------
        image : numpy.ndarray
            The image to add the bounding box to.
        pred : dict
            The coordinates of the bounding box.
        sop_uid : str
            The SOP_UID of the image.
        Returns
        -------
        image : numpy.ndarray
            The image with the bounding box.
        """
        if pred[sop_uid]['coords'] is None:
            print('No bounding box found for SOP_UID: ' + sop_uid)
            return image

        img_copy = image.copy()
        bb = DataHelper._get_coords(pred, sop_uid)
        img_copy[bb[1]:bb[3], bb[0]:bb[0] + 10] = 0
        img_copy[bb[1]:bb[3], bb[2]:bb[2] + 10] = 0
        img_copy[bb[1]:bb[1] + 10, bb[0]:bb[2]] = 0
        img_copy[bb[3]:bb[3] + 10, bb[0]:bb[2]] = 0

        return img_copy

    # --------------------------------------------------------------------------
    @staticmethod
    def _get_coords(pred, sop_uid):
        """Gets the coordinates of the bounding box.
        Parameters
        ----------
        pred : dict
            The predictions dictionary.
        sop_uid : str
            The SOP_UID of the image.
        Returns
        -------
        coords : list
            The coordinates of the bounding box.
        """
        if pred[sop_uid]['coords'] is None:
            print('No bounding box found for SOP_UID: ' + sop_uid)
            return None
        coords = pred[sop_uid]['coords']
        coords = [int(p) for p in coords]
        return coords

    # --------------------------------------------------------------------------
    @staticmethod
    def _rescale_image(ds: dicom.dataset.FileDataset):
        """Rescales the image to prepare for saving as a jpg or png
        Parameters
        ----------
        ds : dicom
            The image to rescale
        Returns
        -------
        Image : PIL.Image
            rescaled image
        """
        new_image = ds.pixel_array.astype(float)
        scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
        scaled_image = np.uint8(scaled_image)
        final_image = Image.fromarray(scaled_image)
        return final_image

    # --------------------------------------------------------------------------
    @staticmethod
    def get2D(N=1,
              data_loader=None,
              cancer=None,
              randomize=False,
              config_num=2,
              timing=False
              ):
        """Returns a list of N random 2D images from the dataset.
        Parameters
        ----------
        N : int
            (default is 1) Number of images to return
        data_loader : DataLoader
            (default is None) The data loader to use to load the data
        cancer : bool
            (default is False)
            If true will return only cancerous images, else will return only
             non-cancerous images
        randomize : bool
            (default is False)
            If true will return the images in a random order
        config_num : int
            (default is 1)
            The option number corresponds to a set of paths to specific data
            which is loaded into the data instance. config_num options are:
            1: Full Omama dataset
            2: Omama dataset for 2D whitelist
        timing : bool
            (default is False)
            If true will print the time it took to perform the action
        Return
        ------
        images : list
            N 2D images from the dataset
        """
        t0 = time.time()
        if data_loader is None:
            data_loader = O.OmamaLoader(config_num=config_num)
        else:
            data_loader = data_loader
        data = Data(data_loader, load_cache=True)
        if cancer is True:
            label = "IndexCancer"
        elif cancer is False:
            label = "NonCancer"
        else:
            label = None
        images = []
        gen = data.next_image(_2d=True, label=label, randomize=randomize,
                              timing=timing)
        for i in range(N):
            images.append(next(gen))
        if timing is True:
            Data.timing(t0, 'get2D')
        return images

    # --------------------------------------------------------------------------
    @staticmethod
    def get3D(N=1,
              data_loader=None,
              cancer=None,
              randomize=False,
              config_num=2,
              timing=False
              ):
        """ Returns a list of N random 3D images from the dataset.
        Parameters
        ----------
        N : int
            (default is 1) Number of images to return
        data_loader : DataLoader
            (default is None) The data loader to use to load the data
        cancer : bool
            (default is False)
            If true will return only cancerous images, else will return
            only non-cancerous images
        randomize : bool
            (default is False)
            If true will return the images in a random order
        config_num : int
            (default is 1)
            The option number corresponds to a set of paths to specific data
            which is loaded into the data instance. config_num options are:
            1: Full Omama dataset
            2: Omama dataset for 2D whitelist
        timing : bool
            (default is False)
            If true will print the time it took to perform the action
        Return
        ------
        images : list
            N 3D images from the dataset.
        """
        t0 = time.time()
        if data_loader is None:
            data_loader = O.OmamaLoader(config_num=config_num)
        else:
            data_loader = data_loader
        data = Data(data_loader, load_cache=True)
        if cancer is True:
            label = "IndexCancer"
        elif cancer is False:
            label = "NonCancer"
        else:
            label = None
        images = []
        gen = data.next_image(_3d=True, label=label, randomize=randomize,
                              timing=timing)
        for i in range(N):
            images.append(next(gen))
        if timing is True:
            Data.timing(t0, 'get3D')
        return images

    # --------------------------------------------------------------------------
    @staticmethod
    def view(image,
             prediction=None,
             slice_num=None,
             cmap='gray',
             vmin=None,
             vmax=None,
             normalize=False,
             interpolation='none',
             config_num=1,
             histogram=False,
             downsample=None,
             timing=False
             ):
        """Displays the image in the specified path or image_id
        Parameters
        ----------
        image : str, int, SimpleNamespace
            The image to display can be specified by its path, its id or by
            any of the objects returned by the DataHelper static methods or
            the Data get_image() method.
        prediction : dict
            (default is None)
            If specified, will add a bounding box to the image.
        slice_num : int
            (default is None)
            If specified will display the specified frame of the 3D image.
        cmap : str
            (default is 'gray') The color map to use. cmap can be the following:
            'gray', 'bone', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        vmin : float
            (default is None) The minimum value of the color map. Can set to
            'auto' to use the minimum value of the image.
        vmax : float
            (default is None) The maximum value of the color map. Can set to
            'auto' to use the maximum value of the image.
        normalize : bool
            (default is False)
            If true will normalize the image to the range [0, 1].
        interpolation : str
            (default is 'nearest') The interpolation to use when displaying
            the image. Interpolation can be 'nearest', 'bilinear', 'bicubic',
            or 'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
            'kaiser', 'quadratic', 'catrom', 'gaussian', 'bessel', 'mitchell',
            'sinc', 'lanczos' or 'none'.
        config_num : int
            (default is 1)
            The option number corresponds to a set of paths to specific data
            which is loaded into the data instance. config_num options are:
            1: Full Omama dataset
            2: Omama dataset for 2D whitelist
            ... see OmamaLoader for full list of config_num options
        histogram : bool
            (default is False) if True, the histogram of the image will be
            displayed.
        downsample : tuple
            (default is None) If specified, will downsample the image by the
            specified factor. The tuple should be of the form (x, y, z) where
            x, y, and z are the downsample factors for the x, y, and z
            dimensions respectively.
        timing : bool
            (default is False)
            If true will print the time it took to perform the action
        """
        t0 = time.time()
        is_nparray = False
        sop_uid = None
        data = DataHelper.check_data_instance(config_num=config_num)
        print("type of data: ", type(data))
        path = DataHelper.check_type(image, data)

        if isinstance(path, dicom.dataset.FileDataset):
            ds = path
        elif isinstance(path, np.ndarray):
            ds = path
            is_nparray = True
        else:
            ds = dicom.dcmread(path)

        if is_nparray is False:
            is_3d_flag = len(ds.pixel_array.shape) == 3
            # save the sop_uid for the image
            sop_uid = ds.SOPInstanceUID
        else:
            is_3d_flag = len(ds.shape) == 3

        if is_3d_flag:
            # 3D image
            if is_nparray is False:
                img = ds.pixel_array.copy()
            else:
                img = ds.copy()
            if slice_num is None:
                # displays the center fame by default
                if prediction is not None and is_nparray is False:
                    # get the slice number from the prediction
                    slice_num = prediction[sop_uid]['slice']
                else:
                    slice_num = int(img.shape[0] // 2)
                img = img[slice_num]
            else:
                img = img[slice_num]
        elif is_3d_flag is False:
            # 2D image
            if is_nparray is False:
                img = ds.pixel_array.copy()
            else:
                img = ds.copy()
        else:
            print('Not a valid image')
            return None
        # get the dicoms window center and width and use that for vmin and vmax
        if is_3d_flag is False and is_nparray is False:
            if isinstance(ds.WindowCenter, dicom.valuerep.DSfloat):
                wc = float(ds.WindowCenter)
            else:
                wc = float(ds.WindowCenter[1])
            if isinstance(ds.WindowWidth, dicom.valuerep.DSfloat):
                ww = float(ds.WindowWidth)
            else:
                ww = float(ds.WindowWidth[1])
        elif is_3d_flag and is_nparray is False:
            wc = ds.SharedFunctionalGroupsSequence[0].FrameVOILUTSequence[
                0].WindowCenter
            ww = ds.SharedFunctionalGroupsSequence[0].FrameVOILUTSequence[
                0].WindowWidth

        if vmin is None and is_nparray is False:
            vmin = wc - ww
        if vmax is None and is_nparray is False:
            vmax = wc + ww

        if prediction is not None:
            img = DataHelper._add_bounding_box(img,
                                               prediction,
                                               sop_uid)
        # get then minimum and maximum pixel values from the image
        if vmin == 'auto':
            vmin = img.min()
        if vmax == 'auto':
            vmax = img.max()
        # normalize the image
        if normalize is True:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            # we set vmin and vmax to None because if norm is being passed
            # into the imshow function it needs the vmin and vmax to be None.
            vmin = None
            vmax = None
        else:
            norm = None

        if downsample is not None:
            from skimage.transform import resize
            img = resize(img, downsample, order=0, preserve_range=True)
            img = img.astype(np.uint16)

        if histogram:
            if normalize:
                # perform max normalization on the image
                img = O.Normalize.minmax(img)
            img_hist = O.Features.histogram(img)
            plt.hist(img_hist)
        else:
            # display the image
            plt.figure(figsize=(10, 10))
            plt.imshow(img,
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       norm=norm,
                       interpolation=interpolation)
        if prediction is not None:
            # getting the bounding box coordinates for red overlay
            bb = DataHelper._get_coords(prediction, sop_uid)
            if bb is not None:
                plt.gca().add_patch(plt.Rectangle((bb[0], bb[1]),
                                                  bb[2] - bb[0],
                                                  bb[3] - bb[1],
                                                  fill=False,
                                                  edgecolor='r',
                                                  linewidth=2))
        # print the score of the prediction
        if prediction is not None:
            score = prediction[sop_uid]['score']
            plt.title('Score: {}'.format(score))

        if timing is True:
            Data.timing(t0, 'view')

    # --------------------------------------------------------------------------
    @staticmethod
    def view_grid(images,
                  slice_num=None,
                  ncols=None,
                  figsize=None,
                  cmap='gray',
                  show_indices=False,
                  index=0
                  ):
        """Plots a grid of images from a list of images.

        Parameters
        ----------
        images : any
            List of images to plot
        slice_num : int, optional
            (default is None)
            if specified, it will display the frame number of all the 3D images
        ncols : int
            (default is None)
            Number of columns in the grid
        figsize : tuple
            (default is None) Size of the figure
        cmap : str
            (default is 'gray') Color map to use
        show_indices : bool
            (default is False) If True, it will show the index of the image
            in the grid
        index : int
            (default is 0) Index of the show_indices to start at
        """
        data = DataHelper.check_data_instance()
        pixels = DataHelper.get_pixels(images, data)

        # images_copy = pixels.copy()
        imgs = pixels.copy()
        for i in range(len(imgs)):
            if len(imgs[i].shape) == 3:
                img = pixels[i]  # images[i].pixels
                if slice_num is None:
                    imgs[i] = (img[img.shape[0] // 2])
                else:
                    imgs[i] = (img[slice_num])

        if not ncols:
            factors = [i for i in range(1, len(imgs) + 1) if
                       len(imgs) % i == 0]
            ncols = factors[len(factors) // 2] if len(factors) else len(
                imgs) // 4 + 1
        nrows = int(len(imgs) / ncols) + int(len(imgs) % ncols)

        if figsize is None:
            figsize = (3 * ncols, 2 * nrows)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i in range(nrows):
            for j in range(ncols):
                if i * ncols + j < len(imgs):
                    ax[i, j].imshow(imgs[i * ncols + j], cmap=cmap)
                    if show_indices:
                        ax[i, j].axis('off')
                        ax[i, j].set_title(index)
                    index += 1
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------------
    @staticmethod
    def grid_view(images,
                  N=None,
                  slice_num=None,
                  ncols=None,
                  show_axis=False,
                  title=None,
                  plot_title=None,
                  h_pad=-0.5,
                  w_pad=5.0,
                  figsize=(20, 20),
                  cmap='gray',
                  top=0.55,
                  bottom=0.05,
                  left=0.05,
                  right=0.95,
                  dpi=500
                  ):
        """Plots a grid of images from a list of images.

        Parameters
        ----------
        images : list
            List of images to plot
        N : int
            (default is None)
            Number of images to plot from the list of images
        slice_num : int, optional
            (default is None)
            if specified, it will display the frame number of all the 3D images
        ncols : int
            (default is None)
            Number of columns in the grid
        show_axis : bool
            (default is False) if True, will display the axis of the image.
        title : str
            (default is None) Title of the plot
        plot_title : str
            (default is None) Title of each image in the grid
        h_pad : float
            (default is 0.5) Padding between subplots in the vertical direction
        w_pad : float
            (default is 0.5) Padding between subplots in the horizontal direction
        figsize : tuple
            (default is None) Size of the figure
        cmap : str
            (default is 'gray') Color map to use
        top : float
            (default is 0.99) The top of the subplots of the figure
        bottom : float
            (default is 0.01) The bottom of the subplots of the figure
        left : float
            (default is 0.01) The left side of the subplots of the figure
        right : float
            (default is 0.99) The right side of the subplots of the figure
        dpi : int
            (default is 500) Dots per inch
        """
        data = DataHelper.check_data_instance()
        pixels = DataHelper.get_pixels(images, data)
        if N is None or N > len(pixels):
            N = len(pixels)

        plt.rcParams['figure.dpi'] = dpi
        imgs = pixels[:N].copy()
        for i in range(len(imgs)):
            if len(imgs[i].shape) == 3:
                img = pixels[i]
                if slice_num is None:
                    imgs[i] = (img[img.shape[0] // 2])
                else:
                    imgs[i] = (img[slice_num])
            elif len(imgs[i].shape) == 2:
                img = pixels[i]
                imgs[i] = img
        if not ncols:
            factors = [i for i in range(1, N + 1) if
                       N % i == 0]
            ncols = factors[len(factors) // 2] if len(factors) else len(
                imgs) // 4 + 1
        nrows = int(N / ncols) + int(N % ncols)

        f, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()[:N]
        for img, ax in zip(imgs, axes.flatten()):
            if np.any(img):
                if len(img.shape) > 2 and img.shape[2] == 1:
                    img = img.squeeze()
                ax.imshow(img, cmap=cmap)
                if not show_axis:
                    ax.axis('off')
                if plot_title:
                    ax.set_title(plot_title)
        if title:
            f.suptitle(title)
        f.tight_layout(h_pad=h_pad, w_pad=w_pad)
        f.subplots_adjust(top=top, bottom=bottom, left=left, right=right)
        plt.show()
        plt.close()

    # --------------------------------------------------------------------------
    @staticmethod
    def get(image,
            view=False,
            config_num=1,
            timing=False
            ):
        """fetches the image being specified by the image_id, dicom_name, or path
        Parameters
        ----------
        image : str, int, SimpleNamespace
            The image to display can be specified by its path, its id or by
            any of the objects returned by the DataHelper static methods or
            the Data get_image() method.
        view : bool
            (default is False)
            If true will display the image
        config_num : int
            (default is 1)
            The option number corresponds to a set of paths to specific data
            which is loaded into the data instance. config_num options are:
            1: Full Omama dataset
            2: Omama dataset for 2D whitelist
        timing : bool
            (default is False)
            If true will print the time it took to perform the action
        Returns
        -------
            image : SimpleNamespace
        """
        # calls the get_image function and then prints the pixels of the image
        t0 = time.time()
        data = DataHelper.check_data_instance(config_num=config_num)
        path = DataHelper.check_type(image, data)
        dicom_name = os.path.basename(path)
        img = data.get_image(dicom_name=dicom_name,
                             timing=timing,
                             dicom_header=True)
        # call the view function to display the image if view is true
        if view is True:
            DataHelper.view(dicom_name, timing=timing)
        if timing is True:
            Data.timing(t0, 'get')
        return img

    # --------------------------------------------------------------------------
    @staticmethod
    def store(image,
              filename,
              timing=False
              ):
        """Stores the image as the specified file type.
        Parameters
        ----------
        image : str, int, SimpleNamespace
            The image to display can be specified by its path, its id or by
            any of the objects returned by the DataHelper static methods or
            the Data get_image() method.
        filename : str
            The name of the file to store the image in. Should include the
            file extension, such as image.png or image.jpg. If no file extension
            is specified, it will default to DCM (Dicom) file format.
        timing : bool
            (default is False)
            If true will print the time it took to perform the action
        """
        t0 = time.time()
        file_name = filename.lower()
        data = DataHelper.check_data_instance()
        path = DataHelper.check_type(image, data)
        if isinstance(path, dicom.dataset.FileDataset):
            ds = path
        else:
            ds = dicom.dcmread(path)
        if file_name.endswith('.dcm') is True or '.' not in file_name:
            if '.' not in file_name:
                filename = filename + '.DCM'
            ds.save_as(filename, write_like_original=False)
        elif file_name.endswith('.npy') is True:
            np.save(filename, ds.pixel_array)
        elif file_name.endswith('.npz') is True:
            np.savez_compressed(filename, ds.pixel_array)
        elif file_name.endswith('.png') is True:
            final_image = DataHelper._rescale_image(ds)
            final_image.save(filename)
        elif file_name.endswith('.jpg') is True:
            final_image = DataHelper._rescale_image(ds)
            final_image.save(filename)
        elif file_name.endswith('.tif') is True:
            final_image = DataHelper._rescale_image(ds)
            final_image.save(filename)
        elif file_name.endswith('.bmp') is True:
            final_image = DataHelper._rescale_image(ds)
            final_image.save(filename)
        else:
            raise ValueError('File type not supported')

        if timing is True:
            Data.timing(t0, 'store')

    # --------------------------------------------------------------------------
    @staticmethod
    def store_all(data: Data, save_path, timing=False):
        """ stores all the images from the omama.Data class instance into the
         specified save_path.

        Parameters
        ----------
        data : omama.Data
            The omama.Data class instance to store all the images from.
        save_path : str
            The path to store the images in.
        timing : bool
            (default is False)
            If true will print the time it took to perform the action
        """
        t0 = time.time()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        i = 0
        for img in data:
            print(type(img))
            path = data.image_paths[i]

            # get the last part of the path to use as the filename
            filename = path.split('/')[-1].replace('.', '_') + '.DCM'
            print(filename)
            # add the save path to the filename
            temp_save_path = os.path.join(save_path, filename)
            # store the image
            print(temp_save_path)
            DataHelper.store(img, temp_save_path, timing=timing)
            i += 1
        if timing is True:
            Data.timing(t0, 'store_all')

    # --------------------------------------------------------------------------
    @staticmethod
    def parse_sop_uid_from_paths(paths,
                                 substr_to_remove,
                                 timing=False):
        """Parse the SOPInstanceUID from a list of paths.
        Parameters
        ----------
        paths : list
            list of paths to dicom files
        substr_to_remove : str
            string to remove from the beginning of the file name
        timing : bool
            (default is False) If true will time execution of method,
            else will not
        Returns
        -------
        sop_uids : list
            list of SOPInstanceUIDs
        """
        t0 = time.time()
        sop_uids = []
        for path in paths:
            sop_uids.append(
                os.path.basename(path).replace(substr_to_remove, ""))
        if timing is True:
            Data.timing(t0, 'parse_sop_uid_from_paths')
        return sop_uids

    # --------------------------------------------------------------------------

    @staticmethod
    def show_histogram(image, norm_type='min-max'):
        """Displays a histogram of the images.
        Parameters
        ----------
        image : np.ndarray
            array of pixel values
        norm_type : str
            (default is 'min-max')
            type of normalization to use for the histogram
        """
        data = DataHelper.check_data_instance()
        pixels = DataHelper.get_pixels(image, data)
        # normalize the image
        img = O.Normalize.get_norm(pixels, norm_type=norm_type)

        y_axis = O.Features.histogram(img)
        x_axis = np.arange(0, 256, 1)

        fig, ax = plt.subplots()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(x_axis, y_axis, width=10, color='b', log=True)
        ax.set_xlim(.01, 255)
        ax.set_ylim(.01, 10 ** 8)
        plt.show()

    @staticmethod
    def file_to_list(file_path):
        """Reads a file and returns a list of the lines in the file.

        Parameters
        ----------
        file_path : str
            path to the file to read

        Returns
        -------
        list
            list of the lines in the file
        """
        # read each line turning into a list removing the newline character
        with open(file_path, 'r') as f:
            lines = [line.rstrip('\n') for line in f]
        return lines

    def merge_lists(*args):
        """Merges a list of lists into a single list.
        goes through each list taking element i from each list and adding it to the
        new list until all lists are exhausted.

        Parameters
        ----------
        *args : list
            list of lists to merge

        Returns
        -------
        list
            merged list
        """
        merged_list = []
        for i in range(len(args[0])):
            for j in range(len(args)):
                merged_list.append(args[j][i])
        return merged_list

    @staticmethod
    def list_to_caselist(list_of_paths, save_path, timing=False):
        """Converts a list of paths to a text file with one path per line.

        Parameters
        ----------
        list_of_paths : list
            list of paths to the cases
        save_path : str
            path to save the text file
        timing : bool
            (default is False) If true will time execution of method,
            else will not

        Returns
        -------
        str
            path to the text file
        """
        t0 = time.time()
        # create a new text file
        with open(save_path, 'w') as f:
            # go through each path
            for path in list_of_paths:
                # write the path to the text file add the newline character if not
                # the last path
                if path != list_of_paths[-1]:
                    f.write(path + '\n')
                else:
                    f.write(path)

        if timing is True:
            Data.timing(t0, 'list_to_caselist')
        return save_path

    @staticmethod
    def remove_b_from_a(a, b):
        """Removes all elements in list b from list a.

        Parameters
        ----------
        a : list
            list to remove elements from
        b : list
            list of elements to remove

        Returns
        -------
        list
            list with elements removed
        """
        # turn a and b into sets and then subtract b from a
        return list(set(a) - set(b))

    @staticmethod
    def add_b_to_a(a, b):
        """Adds all elements in list b to list a.

        Parameters
        ----------
        a : list
            list to add elements to
        b : list
            list of elements to add

        Returns
        -------
        list
            list with elements added
        """
        # turn a and b into sets and then add b to a
        return list(set(a) | set(b))

    @staticmethod
    def namespaces_to_dict(namespaces):
        """Converts list of namespaces to a dictionary.

        Parameters
        ----------
        namespaces : list
            list of namespaces

        Returns
        -------
        dict
            dictionary of the namespace
        """
        # create a dictionary
        d = {}
        # go through each namespace
        for i, ns in enumerate(namespaces):
            # add the namespace to the dictionary
            d[i] = ns.__dict__
        return d

    @staticmethod
    def image_collection_to_npy_file(colleciton_path, output_path, infile_type):
        """Converts an image collection to a numpy file.

        Parameters
        ----------
        colleciton_path : str
            path to the image collection
        output_path : str
            path to save the numpy file
        infile_type : str
            type of file in the collection
        """
        # get the list of files in the collection
        files = DataHelper.get_files_in_collection(colleciton_path, infile_type)
        # create a list to store the images
        images = []
        # go through each file
        for file in files:
            # read the image
            img = DataHelper.read(file)
            # add the image to the list
            images.append(img)
        # convert the list to a numpy array
        images = np.array(images)
        # save the numpy array        cls._init_labels()
        np.save(output_path, images)

    @staticmethod
    def get_files_in_collection(colleciton_path, infile_type):
        """Gets a list of files in an image collection.

        Parameters
        ----------
        collection_path : str
            path to the image collection
        infile_type : str
            type of file in the collection

        Returns
        -------
        list
            list of files in the collection
        """
        # get the list of files in the collection
        files = glob.glob(colleciton_path + '/*.' + infile_type)
        return files

    @staticmethod
    def read(file):
        """Reads an image file. using numpy"""
        # read a .TIF file
        img = np.array(Image.open(file))
        return img

    @staticmethod
    def data_to_binary_bin_hists(data):
        """
        convert the data to binary bins
        """
        # normalize the data
        data = O.Features.get_features(data,
                                       feature_type='hist',
                                       norm_type='minmax',
                                       bins=2)
        return data

    @staticmethod
    def combine_best_results_and_resave(pickle_paths,
                                          save_path):
        """Loads a list of pickle files which are dictionaries and goes through
        each dictionary comparing the roc_auc score for each algorithm. If the
        roc_auc score is better than the current best score it is saved to the
        results dictionary.

        Can be looped through as follows:
        for k,v in set1.items():
            print(set1[k][0]['evaluation']['roc_auc'])

        Parameters
        ----------
        pickle_paths : list, str
            list of paths to the pickle files
        save_path : str
            path to save the combined results
        """
        if isinstance(pickle_paths, str):
            # if it is a string it is a path to a root diretory containing the
            # pickle files, get the list of pickle files
            pickle_paths = glob.glob(pickle_paths + '/*.pkl')
        print(f'Length of pickle_paths: {len(pickle_paths)}')
        # create a dictionary to store the results
        results = {}
        # go through each pickle file
        for i, pickle_path in enumerate(pickle_paths):
            # load the pickle file
            with open(pickle_path, 'rb') as f:
                d = pickle.load(f)
            # go through each algorithm
            for k, v in d.items():
                # if the algorithm is not in the results dictionary add it
                if k not in results:
                    results[k] = v
                # if the algorithm is in the results dictionary compare the
                # roc_auc score
                else:
                    # if the roc_auc score is better than the current best
                    # score replace the current best score
                    if v[0]['evaluation']['roc_auc'] > results[k][0]['evaluation']['roc_auc']:
                        results[k] = v
            print(f'Finished {i+1} of {len(pickle_paths)}')
        # save the results
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

    @staticmethod
    def build_gt(n, indices):
        """
        build a ground truth matrix of length n starting at all zeros and,
        and for each index in indices, set the value to 1.
        """
        # create a ground truth array of zeros
        gt = np.zeros(n)
        # set the indices to 1
        gt[indices] = 1
        return gt
