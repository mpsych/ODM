import os
import pickle
import re
import time
import pprint
import random
from collections import Counter
import pydicom as dicom
from types import SimpleNamespace
from typing import Generator
from omama.loaders.data_loader import DataLoader


class Data:
    """Singleton Data class used to represent and explore dicom data

    Attributes
    ----------
    total_2d_noncancer : int
        total number of 2D dicoms with a label of NonCancer
    total_2d_cancer : int
        total number of 2D dicoms with a label of IndexCancer
    total_2d_preindex : int
        total number of 3D dicoms with a label of PreIndexCancer
    total_3d_noncancer : int
        total number of 3D dicoms with a label of NonCancer
    total_3d_cancer : int
        total number of 3D dicoms with a label of IndexCancer
    total_3d_preindex : int
        total number of 3D dicoms with a label of PreIndexCancer
    """
    dl = None  # DataLoader object
    # total number of 2D dicoms with a label of NonCancer
    total_2d_noncancer = 0
    # total number of 2D dicoms with a label of IndexCancer
    total_2d_cancer = 0
    # total number of 3D dicoms with a label of PreIndexCancer
    total_2d_preindex = 0
    # total number of 3D dicoms with a label of NonCancer
    total_3d_noncancer = 0
    # total number of 3D dicoms with a label of IndexCancer
    total_3d_cancer = 0
    # total number of 3D dicoms with a label of PreIndexCancer
    total_3d_preindex = 0
    # the sorted images from all studies. This is the list that image_id is
    # derived from
    _image_paths = []
    # the sorted study paths to all studies. This is the list study_id is
    # derived from
    _study_paths = []
    # dictionary of the label mappings
    _label_mapping_dict = {}
    # dictionary of the generated stats about the data contained in the Data
    # instance
    _stats = {}
    # dictionary of all the labels contained in the Data object instance
    _labels = {}
    # dictionary of flags used while fetching filtered data
    _filtered_flags = {}
    # dictionary of sop_uid to cancer labels
    pat_id_to_label_dict = {}

    _caselist_path = None
    _cache = None
    _load_cache = None
    _print_data = None
    instance = None

    def __new__(
            cls,
            data_loader: DataLoader,
            cache: bool = False,
            load_cache: bool = False,
            cache_paths: list = None,
            print_data: bool = False,
            timing: bool = False
    ):
        """Initializes the Data class and prepares the data for exploratory
        jupyter notebook sessions

        Parameters
        ----------
        data_loader : DataLoader
            object used to tell the Data class where to find the data
        cache : bool
            (default False) whether to cache the data
        load_cache : bool
            (default False) whether to load the data from the cache
        print_data : bool
            (default is False)
            prints out the detailed dictionary of information about all the
            studies and the combined data as well
        timing : float
            sets the timers in the initializing methods to true, and prints out
            how long each one took
            (default is False)
        """
        t0 = time.time()
        # allow for multiple instances of the Singleton class as long as different
        # data_loader objects are passed in
        if cls.instance is None or cls.dl != data_loader:
            cls.destroy()
            cls.dl = data_loader
            cls.instance = super().__new__(cls)
            if cache_paths is not None:
                cls.dl.cache_paths = cache_paths
            cls._caselist_path = cls.dl.caselist_path
            cls._cache = cache
            cls._load_cache = load_cache
            cls._print_data = print_data
            print("DataLoader type is: ", type(cls.dl))
            # initialize the data
            cls._init(timing=timing)
        if timing:
            cls.timing(t0, "total __init__")
        if cls._print_data:
            pprint.pprint(cls._stats)

        return cls.instance

    @classmethod
    def __str__(cls):
        """Prints out the class name and the number of studies and images in
        the data
        """
        # save the pprint.pprint into a string
        s = pprint.pformat(cls._stats)
        return f"{cls.__name__}({s})"

    @classmethod
    def __len__(cls):
        """Returns the length of all the dicoms in the Data object instance

        Returns
        -------
        length of the Data object instance : int
            the length of all the dicoms in the Data object instance
        """
        return len(cls._image_paths)

    @classmethod
    def __getitem__(
            cls,
            index
    ):
        """Returns the dicom at the given index, or a slice of the dicoms if
        a slice is given

        Parameter
        ---------
        index : int
            the index of the dicom to return

        Returns
        -------
        dicom : dicom.dicom.DicomImage
            the dicom at the given index
        """
        # if isinstance(subscript, slice):
        #     # do your handling for a slice object:
        #     print(subscript.start, subscript.stop, subscript.step)
        # else:
        #     # Do your handling for a plain index
        #     print(subscript)
        # check if the index is a slice
        # return dicom.filereader.dcmread(cls._image_paths[index])

        if isinstance(index, slice):
            return [dicom.filereader.dcmread(path) for path in
                    cls._image_paths[index]]
        else:
            return dicom.filereader.dcmread(cls._image_paths[index])

    @classmethod
    def __iter__(cls):
        """Returns an iterator for the Data object instance

        Returns
        -------
        iterator : iterator
            an iterator for the Data object instance
        """
        # use the image path to read the dicom there and return pixel data
        return (dicom.filereader.dcmread(
            path, specific_tags=cls.dl.dicom_tags) for path in cls._image_paths)

    @classmethod
    def __call__(cls, *args, **kwargs):
        if cls not in cls.instance:
            cls.instance[cls] = cls(*args, **kwargs)
        return cls.instance[cls]

    @classmethod
    def __instancecheck__(cls):
        return cls.instance

    @property
    def image_paths(self):
        return self._image_paths

    @classmethod
    def destroy(cls):
        """Destroys the Data object instance"""
        cls.instance = None
        cls.dl = None
        # reset the class variables
        cls._study_paths = []
        cls._image_paths = []
        cls._map = {}
        cls._stats = {}
        cls._cache = False
        cls._load_cache = False
        cls._print_data = False
        cls._caselist_path = None
        cls._cache_paths = None

    # --------------------------------------------------------------------------
    @classmethod
    def _init(
            cls,
            timing=False
    ):
        """ Initializes the Data object instance with the data contained in the
        data_loader object

        Parameters
        ----------
        timing : bool
            (default is False) if True, the time it takes to initialize the Data
            object instance is printed
        """
        t0 = time.time()
        cls._init_study_paths(timing=timing)
        cls._init_image_paths(timing=timing)
        cls._init_map(timing=timing)
        if cls._load_cache and not cls._cache:
            cls._init_cache(timing=timing)
        elif cls._cache and not cls._load_cache:
            cls._generate_label_map(timing=timing)
            cls._cache_labels(timing=timing)
        elif cls._cache and cls._load_cache:
            cls._init_cache(timing=timing)
            cls._cache_labels(timing=timing)
        else:
            cls._generate_label_map(timing=timing)
        cls._init_labels(timing=timing)
        cls._init_stats(timing=timing)

        if timing:
            cls.timing(t0, "total _init")

    # --------------------------------------------------------------------------
    @classmethod
    def _init_study_paths(
            cls,
            timing=False
    ):
        """ Initializes the study paths of the Data object instance. O(n) time

        Parameters
        ----------
        timing : bool
            (default is False) if True, the time it takes to initialize the
            study paths is printed
        """
        t0 = time.time()
        study_list = []
        if cls._caselist_path is None:
            if cls.dl.data_paths is None or len(cls.dl.data_paths) == 0:
                print("no study files loaded")
                return
            # load the sorted study from list of paths
            for path in cls.dl.data_paths:  # iterate through each path
                study = sorted(
                    os.listdir(path))  # get list of studies in the path, sorted
                study_list.append(study)  # add the study to the list of studies
            # loop to create the list of study paths
            for i in range(0, len(study_list)):  # iterate through each study
                study_id = 0  # initialize the study id
                for _ in study_list[i]:
                    #  append the root path to each study
                    study_list[i][study_id] = \
                        cls.dl.data_paths[i] + study_list[i][study_id]
                    study_id = study_id + 1
            cls._study_paths = sorted(
                [item for sublist in study_list for item in sublist])
        else:
            with open(cls._caselist_path, 'r') as f:
                # read each line which is a path
                for path in f:
                    path = path.strip()
                    if path != '':
                        study_path = os.path.dirname(path)
                        cls._study_paths.append(study_path)
            # remove any duplicates and sort the list
            cls._study_paths = sorted(list(set(cls._study_paths)))
        if timing is True:
            cls.timing(t0, 'load_studies')

    # --------------------------------------------------------------------------
    @classmethod
    def _init_map(
            cls,
            timing=False
    ):
        """ Initializes the map of labels from the csv files, which is used
        to build a sop_uid to label dictionary. O(n) time complexity.

        Parameters
        ----------
        timing : bool
            (default is False) if True, the time it takes to initialize the
            map is printed
        """
        t0 = time.time()

        if cls.dl.csv_paths is None or len(cls.dl.csv_paths) == 0:
            print("no csv files loaded")
            return
        # go through each csv file and load the map from it
        for path in cls.dl.csv_paths:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    values = line.split(',')
                    study_id = values[0]
                    lat = values[1]
                    label = values[2].strip()
                    cls._label_mapping_dict[study_id + '_' + lat] = label
        if timing is True:
            cls.timing(t0, 'csv_to_map')

    # --------------------------------------------------------------------------
    @classmethod
    def _init_image_paths(
            cls,
            timing=False
    ):
        """ Initializes the image paths of the Data object instance. O(n) where
        n is the number of subdirectories in each study path.

        Parameters
        ----------
        timing : bool
            (default is False) if True, the time it takes to initialize the
            image paths is printed
        """
        t0 = time.time()
        if cls._caselist_path is None:
            # use the path to each study and get the list of images
            for current_path in cls.dl.data_paths:
                # walk through each study
                for path, currentDirectory, files in os.walk(current_path):
                    # iterate through each image in the study
                    for file in files:
                        # if it is a 2D or 3D image, add it to the list of images
                        if file.startswith(cls.dl.dicom_2d_substring) or \
                                file.startswith(cls.dl.dicom_3d_substring):
                            cls._image_paths.append(path + '/' + str(file))
                cls._image_paths = sorted(cls._image_paths)
        else:
            # open the caselist file and read each line which is a path and
            # add it to the list of image paths and then sort the list
            with open(cls._caselist_path, 'r') as f:
                for path in f:
                    path = path.strip()
                    if path != '':
                        cls._image_paths.append(path)
            cls._image_paths = sorted(cls._image_paths)
        if timing is True:
            cls.timing(t0, 'load_images')

    # --------------------------------------------------------------------------
    @classmethod
    def _generate_label_map(
            cls,
            timing=False
    ):
        """ Generates the label map of the Data object instance. O(n) where n
        is the number of images in the Data object instance.

        Parameters
        ----------
        timing : bool
            (default is False) if True, the time it takes to generate the
            label map is printed
        """
        t0 = time.time()
        image_id = 0
        for path in cls._image_paths:
            if os.path.basename(path).startswith(cls.dl.dicom_2d_substring) or \
                    cls.dl.dicom_2d_substring == '':
                ds = dicom.filereader.dcmread(path,
                                              stop_before_pixels=True,
                                              specific_tags=[
                                                  "ImageLaterality",
                                                  cls.dl.patient_identifier,
                                                  cls.dl.cancer_identifier,
                                                  "StudyInstanceUID"])
                image_laterality = ds.get("ImageLaterality")
            else:
                ds = dicom.filereader.dcmread(path,
                                              stop_before_pixels=True,
                                              specific_tags=[
                                                  "SharedFunctionalGroupsSequence",
                                                  cls.dl.patient_identifier,
                                                  cls.dl.cancer_identifier,
                                                  "StudyInstanceUID"])
                image_laterality = \
                    ds.SharedFunctionalGroupsSequence[0].FrameAnatomySequence[
                        0].FrameLaterality
            can_id = ds.get(cls.dl.cancer_identifier)
            # make dictionary of labels based off of ImageID as index
            label = cls._label(can_id, image_laterality)
            pat_id = ds.get(cls.dl.patient_identifier)
            cls.pat_id_to_label_dict[pat_id] = label
            cls._labels[image_id] = label

            image_id += 1
            if timing is True:
                if image_id % 1000 == 0:
                    cls.timing(t0, str(image_id))
        if timing is True:
            cls.timing(t0, 'load_labels')

    # --------------------------------------------------------------------------
    @classmethod
    def _init_labels(
            cls,
            timing=False
    ):
        """ Initializes the labels of the Data object instance. O(n)

        Parameters
        ----------
        timing : bool
            (default is False) if True, the time it takes to initialize the
            labels is printed
        """
        t0 = time.time()
        cls.total_2d_cancer = 0
        cls.total_3d_cancer = 0
        cls.total_2d_preindex = 0
        cls.total_3d_preindex = 0
        cls.total_2d_noncancer = 0
        cls.total_3d_noncancer = 0
        # loop over the image_paths and generate the counts of all the possible
        # filtered scenarios
        for i in range(0, len(cls._image_paths)):
            path = cls._image_paths[i]
            label = cls._labels[i]
            if os.path.basename(path).startswith(cls.dl.dicom_2d_substring) or \
                    cls.dl.dicom_2d_substring == '':
                if label == 'IndexCancer':
                    cls.total_2d_cancer += 1
                elif label == 'PreIndexCancer':
                    cls.total_2d_preindex += 1
                elif label == 'NonCancer':
                    cls.total_2d_noncancer += 1
            else:
                if label == 'IndexCancer':
                    cls.total_3d_cancer += 1
                elif label == 'PreIndexCancer':
                    cls.total_3d_preindex += 1
                elif label == 'NonCancer':
                    cls.total_3d_noncancer += 1
        if timing is True:
            cls.timing(t0, 'generate_counts')

    # --------------------------------------------------------------------------
    @property
    def labels(cls):
        """ Returns the labels of the Data object instance
        """
        return cls._labels

    @classmethod
    def _init_stats(
            cls,
            timing=False
    ):
        """ Initializes the statistics of the Data object instance

        Parameters
        ----------
        timing : bool
            (default is False) if True, the time it takes to initialize the
            statistics is printed
        """
        t0 = time.time()
        cls._stats = {}
        tot_all_dicoms = 0
        tot_all_2d = 0
        tot_all_3d = 0
        files_2d_per_study = {}
        files_3d_per_study = {}
        # loop over the image_paths and generate the counts of all the possible
        # scenarios
        for path in cls._image_paths:
            for study_folder in cls.dl.study_folder_names:
                if study_folder in path:
                    # counting the 2d Dicom files per study
                    if os.path.basename(path).startswith(
                            cls.dl.dicom_2d_substring) or \
                            cls.dl.dicom_2d_substring == '':
                        tot_all_2d += 1
                        files_2d_per_study[study_folder] = \
                            files_2d_per_study.get(study_folder, 0) + 1
                    #  counting the 3d Dicom files per study
                    elif os.path.basename(path).startswith(
                            cls.dl.dicom_3d_substring) or \
                            cls.dl.dicom_3d_substring == '':
                        tot_all_3d += 1
                        files_3d_per_study[study_folder] = \
                            files_3d_per_study.get(study_folder, 0) + 1
                    tot_all_dicoms += 1
        # Using the counts from above, generate the statistics for each study
        # as well as the global statistics
        for i in range(0, len(cls.dl.data_paths)):
            folder = str(cls.dl.study_folder_names[i])
            if folder in files_2d_per_study:
                _2D = files_2d_per_study[folder]
            else:
                _2D = 0
            if folder in files_3d_per_study:
                _3D = files_3d_per_study[folder]
            else:
                _3D = 0
            total = _2D + _3D
            case = {'total': total, '3D': _3D, '2D': _2D}
            cls._stats[folder] = case
            # using SimpleNamespace to give dot notation access to the stats
            cls._stats[folder] = SimpleNamespace(**cls._stats[folder])
        cls._stats['total_all_dicoms'] = tot_all_dicoms
        cls._stats['total_2d_all'] = tot_all_2d
        cls._stats['total_3d_all'] = tot_all_3d
        cls._stats['total_2d_cancer'] = cls.total_2d_cancer
        cls._stats['total_2d_preindex'] = cls.total_2d_preindex
        cls._stats['total_2d_noncancer'] = cls.total_2d_noncancer
        cls._stats['total_3d_cancer'] = cls.total_3d_cancer
        cls._stats['total_3d_preindex'] = cls.total_3d_preindex
        cls._stats['total_3d_noncancer'] = cls.total_3d_noncancer
        cls._stats['total_cancer'] = cls.total_2d_cancer + cls.total_3d_cancer
        cls._stats[
            'total_preindex'] = cls.total_2d_preindex + cls.total_3d_preindex
        cls._stats[
            'total_noncancer'] = cls.total_2d_noncancer + cls.total_3d_noncancer
        cls._stats['total_no_label'] = tot_all_dicoms - (
                cls.total_3d_noncancer + cls.total_2d_noncancer +
                cls.total_3d_preindex + cls.total_2d_preindex +
                cls.total_3d_cancer + cls.total_2d_cancer)
        # Using SimpleNamespace to give nested dot notation access to the stats
        cls._stats = SimpleNamespace(**cls._stats)
        if timing is True:
            cls.timing(t0, 'generate_stats')

    # --------------------------------------------------------------------------
    @classmethod
    def _save_pickle(
            cls,
            path,
            data,
            timing=False
    ):
        """ Saves the data to a pickle file

        Parameters
        ----------
        path : str
            the path and name of the pickle file to save the data to
        data : dict
            the data to save to the pickle file
        timing : bool
            (default is False) if True, the time it takes to save the data is
            printed
        """
        t0 = time.time()
        if not os.path.isfile(path):
            with open(path, 'wb') as file:
                pickle.dump(data, file)
            file.close()
        else:
            pickle_file = open(path, 'wb')
            pickle.dump(data, pickle_file)
            pickle_file.close()

        if timing is True:
            cls.timing(t0, 'save_pickle')

    # --------------------------------------------------------------------------
    @classmethod
    def _load_pickle(
            cls,
            path,
            timing=False
    ):
        """ Loads a pickle file from the given path

        Parameters
        ----------
        path : str
            the path to the pickle file
        timing : bool
            (default is False) if True, the time it takes to load the pickle file
            is printed
        """
        t0 = time.time()
        pickle_file = open(path, 'rb')
        result = pickle.load(pickle_file)
        pickle_file.close()
        if timing is True:
            cls.timing(t0, 'load_pickle')
        return result

    # --------------------------------------------------------------------------
    @classmethod
    def _cache_labels(
            cls,
            timing=False
    ):
        """ Caches the _labels and the sopuid_to_labels of the Data object
        instance

        Parameters
        ----------
        timing : bool
            (default is False) if True, the time it takes to cache the labels is
             printed
        """
        t0 = time.time()

        cls._save_pickle(cls.dl.cache_paths[0], cls._labels)
        cls._save_pickle(cls.dl.cache_paths[1], cls.pat_id_to_label_dict)

        if timing is True:
            cls.timing(t0, 'save_pickle')

    # --------------------------------------------------------------------------
    @classmethod
    def _init_cache(
            cls,
            timing=False
    ):
        """ Initializes the cache of the Data object instance

        Parameters
        ----------
        timing : bool
            (default is False) if True, the time it takes to initialize the
            cache is printed
        """
        t0 = time.time()
        cls._labels = cls._load_pickle(cls.dl.cache_paths[0])
        cls.pat_id_to_label_dict = cls._load_pickle(cls.dl.cache_paths[1])
        if timing is True:
            cls.timing(t0, 'load_pickle')

    # --------------------------------------------------------------------------
    @classmethod
    def path(
            cls,
            image_id: int = None,
            dicom_name: str = None,
            path: str = None,
            study_instance_uid: str = None,
            study_key: int = None,
            timing=False
    ) -> str:
        """ Returns the path to the image, given one of the several ids

        Parameters
        ----------
        image_id : int
            (default is None) the id of the image
        dicom_name : str
            (default is None) the name of the dicom file
        path : str
            (default is None) the path of the image
        study_instance_uid : str
            (default is None) the study instance uid of the image
        study_key : int
            (default is None) the key of the study
        timing : bool
            (default is False) if True, the time it takes to find the path is printed

        Returns
        -------
        path : str
            the path of the image
        """
        t0 = time.time()
        if image_id is not None:
            path = cls._image_paths[image_id]
        elif dicom_name is not None:
            path = ''.join([s for s in cls._image_paths if dicom_name in s])
        elif path is not None:
            path = path
        elif study_instance_uid is not None:
            file_id = [idx for idx, s in enumerate(cls._study_paths) if
                       study_instance_uid in s][0]
            path = cls._study_paths[file_id]
        elif study_key is not None:
            path = cls._study_paths[study_key]
        else:
            return "Error: No valid image_id, dicom_name, path, " \
                   "study_instance_uid or study_key was specified "
        if timing is True:
            cls.timing(t0, 'get_path')
        return path

    # --------------------------------------------------------------------------
    @classmethod
    def _dicom(
            cls,
            dicom_name: str = None,
            image_id: int = None,
            dicom_path: str = None,
            pixels=False,
            dicom_header=False,
            timing=False
    ):
        """ Returns the dicom object of the image with the given path

        Parameters
        ----------
        dicom_name : str
            (default is None) the name of the dicom file
        image_id : int
            (default is None) the id of the image
        dicom_path : str
            (default is None) the path of the dicom file
        pixels : bool
            (default is False) if True, the pixel data of the image is returned
        dicom_header : bool
            (default is False) if True, the dicom header of the image is
            returned
        timing : bool
            (default is False) if True, the time it takes to load the dicom
            is printed

        Returns
        -------
        dicom : pydicom.dicom.Dataset
            the dicom object of the image
        """
        t0 = time.time()

        if image_id is not None:
            img_path = cls.path(image_id=image_id)
            if cls._validate_path(img_path) is False:
                return "path not valid", None
        elif dicom_path is not None:
            if cls._validate_path(dicom_path) is False:
                return "path not valid", None
            img_path = dicom_path
        elif dicom_name is not None:
            img_path = cls.path(dicom_name=dicom_name)
            if cls._validate_path(img_path) is False:
                return "path not valid", None
        else:
            print(
                "Error: No valid dicom_name, image_id, dicom_path or "
                "dicom_name was specified")
            return "None", None
        if dicom_header and pixels:
            ds = dicom.filereader.dcmread(img_path, force=True)
        elif pixels:
            ds = dicom.filereader.dcmread(img_path, force=True,
                                          specific_tags=cls.dl.dicom_tags)
        else:
            ds = dicom.filereader.dcmread(img_path, force=True,
                                          stop_before_pixels=True)
        if timing is True:
            cls.timing(t0, '_get_dicom')
        return ds

    # --------------------------------------------------------------------------
    @classmethod
    def _validate_path(
            cls,
            path,
            _2d=True,
            _3d=True,
    ) -> bool:
        """ Checks if the path is valid

        Parameters
        ----------
        path : str
            the path to check
        _2d : bool
            (default is True) if True, the path must be a 2D image
        _3d : bool
            (default is True) if True, the path must be a 3D image

        Returns
        -------
        valid : bool
            if True, the path is valid and can be used, otherwise it is not
        """
        if _2d and _3d:
            if os.path.basename(path).startswith(cls.dl.dicom_3d_substring) or \
                    os.path.basename(path).startswith(
                        cls.dl.dicom_2d_substring) or \
                    cls.dl.dicom_3d_substring == '' or \
                    cls.dl.dicom_2d_substring == '':
                return True
        elif _2d:
            if os.path.basename(path).startswith(cls.dl.dicom_2d_substring) or \
                    cls.dl.dicom_2d_substring == '':
                return True
        elif _3d:
            if os.path.basename(path).startswith(cls.dl.dicom_3d_substring) or \
                    cls.dl.dicom_3d_substring == '':
                return True
        return False

    # --------------------------------------------------------------------------
    @classmethod
    def _files_in_series(
            cls,
            path=None,
            study_instance_uid=None,
            _2d=True,
            _3d=True,
            labels=None,
            manufacturer=None,
            timing=False
    ) -> list:
        """ Returns the files in the series with the given path

        Parameters
        ----------
        path : str,
            (default is None) the path of the series
        study_instance_uid : str
            (default is None) the study instance uid of the series
        timing : bool
            (default is False) if True, the time it takes to load the dicom
            is printed

        Returns
        -------
        files : list
            the files in the series
        """
        t0 = time.time()
        if path is not None:
            file_names = os.listdir(path)
        else:
            idx = [idx for idx, s in enumerate(cls._study_paths) if
                   study_instance_uid in s][0]
            file_names = os.listdir(cls._study_paths[idx])
        file_names = [f for f in file_names if cls._validate_path(f, _2d, _3d)]
        if labels is not None:
            temp = []
            for f in file_names:
                img_idx = cls._image_id(dicom_name=f, timing=timing)
                if img_idx != -1:
                    temp.append(f)
            file_names = temp.copy()
        if manufacturer is not None:
            temp = []
            for f in file_names:
                ds = cls._dicom(dicom_name=f,
                                pixels=False,
                                dicom_header=False,
                                timing=timing)
                if ds.Manufacturer == manufacturer:
                    temp.append(f)
            file_names = temp.copy()
        if timing is True:
            cls.timing(t0, 'get_file_names_in_series')
        return file_names

    # --------------------------------------------------------------------------
    @classmethod
    def _image_id(
            cls,
            dicom_name,
            timing=False
    ) -> int:
        """ Returns the image id of the image with the given dicom name

        Parameters
        ----------
        dicom_name : str
            the dicom name of the image
        timing : bool
            (default is False) if True, the time it takes to load the dicom
            is printed

        Returns
        -------
        image_id : int
            the image id of the image
        """
        t0 = time.time()
        index = -1
        temp_index = 0
        for img in cls._image_paths:
            if dicom_name in img:
                index = temp_index
                break
            temp_index += 1
        if timing is True:
            cls.timing(t0, 'image_id')
        return index

    # --------------------------------------------------------------------------
    @classmethod
    def _study_key(
            cls,
            study_instance_uid,
            timing=False
    ) -> int:
        """ Returns the study key of the study with the given study instance uid

        Parameters
        ----------
        study_instance_uid : str
            the study instance uid of the study
        timing : bool
            (default is False) if True, the time it takes to load the dicom
            is printed

        Returns
        -------
        study_key : int
            the study key of the study
        """
        t0 = time.time()
        index = [idx for idx, s in enumerate(cls._study_paths) if
                 study_instance_uid in s][0]
        if timing is True:
            cls.timing(t0, 'study_key')
        return index

    # --------------------------------------------------------------------------
    @classmethod
    def _label(
            cls,
            cancer_id,
            image_laterality,
            timing=False
    ) -> str:
        """ Returns the label of the image with the given study instance uid and
         image laterality

        Parameters
        ----------
        cancer_id : str
            the study instance uid of the study
        image_laterality : str
            the image laterality of the image
        timing : bool
            (default is False) if True, the time it takes to load the dicom
            is printed

        Returns
        -------
        label : str
            the label of the image
        """
        t0 = time.time()
        key = str(cancer_id) + '_' + str(image_laterality)
        if key in cls._label_mapping_dict:
            label = cls._label_mapping_dict[key]
        elif str(cancer_id) + '_' + 'None' in cls._label_mapping_dict:
            label = cls._label_mapping_dict[
                str(cancer_id) + '_' + 'None']
        elif ((
                      str(cancer_id) + '_' + 'L' in cls._label_mapping_dict) or (
                      str(cancer_id) + '_' + 'R' in cls._label_mapping_dict) or
              (str(cancer_id) + '_' + 'B' in cls._label_mapping_dict)):
            label = "NonCancer"
        else:
            label = 'No Label Information'

        if timing is True:
            cls.timing(t0, 'get_label')
        return label

    # --------------------------------------------------------------------------
    @classmethod
    def to_text_file(cls, file_path=None, file_name=None):
        """ Writes each path from the image_paths list to a text file each
        separated by a new line

        Parameters
        ----------
        file_name : str
            the name of the file to write to
        file_path : str
            the path to the file to write to

        Returns
        -------
        file_path : str
            the path to the file that was written to
        """
        if file_path is None:
            # get current working directory
            file_path = os.getcwd() + '/'
        if file_name is None:
            file_name = 'image_paths.txt'
        with open(file_path + file_name, 'w') as f:
            for path in cls._image_paths:
                # check if it is the last line in the file and if it is, don't
                # add a new line
                if path != cls._image_paths[-1]:
                    f.write(path + '\n')
                else:
                    f.write(path)
        return file_path + file_name

    # --------------------------------------------------------------------------
    @classmethod
    def get_label(cls, cancer_id, timing=False):
        """ Returns the label of the image with the given sop uid

        Parameters
        ----------
        cancer_id : str
            the sop uid of the image
        timing : bool
            (default is False) if True, the time it takes to load the dicom
            is printed

        Returns
        -------
        label : str
            the label of the image
        """
        t0 = time.time()
        if cancer_id in cls.pat_id_to_label_dict:
            label = cls.pat_id_to_label_dict[cancer_id]
        else:
            label = 'No Label Information'
        if timing is True:
            cls.timing(t0, 'get_label')
        return label

    # --------------------------------------------------------------------------
    @classmethod
    def get_study(
            cls,
            study_instance_uid: str = None,
            study_key: int = None,
            verbose=False,
            timing=False
    ) -> SimpleNamespace:
        """ Returns the study with the given study instance uid or study key

        Parameters
        ----------
        study_instance_uid : str
            (default is None, this or the ID need to be set)
            Uses the StudyInstanceUID to locate the study information and returns
            data as dictionary
        study_key : int
            (default is None, this or the study_instance_uid needs to be set)
            Uses the location of a study which is based off of its index in the
            sorted list of all studies
        verbose : bool
            (default is False) If True, will include additional information
            about the study such as the number of 2D and 3D dicoms in study
        timing : bool
            (default is False) Sets timing flag, if true will time execution time
             of method, else will not

        Returns
        ------
        study : SimpleNamespace
            {'directory': XX, 'images': [imagefile1, imagefile2, ...]}
            iff (verbose==True) => info': {'3D_count': XX, '2D_count': XX ..and others}
        """
        _2d = True
        _3d = True
        lab = []
        t0 = time.time()
        if study_instance_uid is not None:
            uid_path = cls.path(study_instance_uid=study_instance_uid,
                                timing=timing)
            file_id = cls._study_key(uid_path, timing=timing)
        elif study_key is not None:
            uid_path = cls.path(study_key=study_key, timing=timing)
            study_instance_uid = os.path.basename(uid_path)
            file_id = study_key
        else:
            raise ValueError(
                'Either study_instance_uid or study_key must be set')
        if cls._filtered_flags != {}:  # if there are filters
            _2d = cls._filtered_flags['2D']
            _3d = cls._filtered_flags['3D']
            lab = cls._filtered_flags['labels']
            man = cls._filtered_flags['manufacturers']
        image_files = cls._files_in_series(path=uid_path,
                                           _2d=_2d,
                                           _3d=_3d,
                                           labels=lab,
                                           timing=timing
                                           )
        if cls.dl.dicom_3d_substring != '':
            files_3d = [i for i in image_files if
                        cls.dl.dicom_3d_substring in i]
        else:
            # if there is no 3d substring, then all files are 3d
            files_3d = [image_files for i in image_files]
        if cls.dl.dicom_2d_substring != '':
            files_2d = [i for i in image_files if
                        cls.dl.dicom_2d_substring in i]
        else:
            # if there is no 2d substring, then all files are 2d
            files_2d = [image_files for i in image_files]
        # get a list of image ids from each of the files
        image_ids = [cls._image_id(i) for i in image_files]
        # remove all image_ids tha are -1
        image_ids = [i for i in image_ids if i != -1]
        # use the images ids to get the image labels
        image_labels = [cls._labels[i] for i in image_ids]
        # count each of the types in the image labels
        label_counts = Counter(image_labels)

        if verbose:
            info = {'total': len(files_3d) + len(files_2d),
                    '3D count': len(files_3d), '2D count': len(files_2d),
                    'label counts': label_counts}
        else:
            info = None

        study = {'directory': uid_path, 'study_uid': study_instance_uid,
                 'study_key': file_id, 'images': image_files,
                 'info': info}
        dot_dictionary = SimpleNamespace(**study)
        if timing is True:
            cls.timing(t0, 'get_study')
        return dot_dictionary

    # --------------------------------------------------------------------------
    @classmethod
    def get_image(
            cls,
            image_id=None,
            dicom_name=None,
            path=None,
            pixels=True,
            dicom_header=False,
            timing=False
    ) -> SimpleNamespace:
        """ Returns the image with the given image id or dicom name or path

        Parameters
        ----------
        image_id : int
            (default is None, this or the Dicom_ID or path need to be set)
            Uses the image_id to fetch Dicom and returns the specified data
        dicom_name : str
            (default is None, this or the image_id or path need to be set)
            Uses the dicom_name to fetch Dicom, and returns the specified  data
        path : str
            (default is None, this or the image_id or dicom_name need to be set)
            Uses the path to fetch Dicom, and returns the specified data
        pixels : bool
            (default is True) If True will return the pixel array of Dicom,
            else will return None
        dicom_header : bool
            (default is False) If True will return the dicom header as a
            dictionary, else will return None
        timing : bool
            (default is False)
            If true will time execution of method, else will not

        Returns
        ------
        image : SimpleNamespace
            [np.array just the pixels or None if pixels=False, info]
            info is dictionary:
            {'label': LABELS.Cancer, 'filepath':... 'shape'...}
        """
        t0 = time.time()
        ds = cls._dicom(
            dicom_name=dicom_name,
            image_id=image_id,
            dicom_path=path,
            pixels=pixels,
            dicom_header=dicom_header,
            timing=timing
        )
        img = None
        study_instance_uid = None
        img_path = cls.path(image_id=image_id, dicom_name=dicom_name,
                            path=path)
        if os.path.basename(img_path).startswith(cls.dl.dicom_3d_substring):
            ds1 = dicom.filereader.dcmread(img_path, stop_before_pixels=True,
                                           specific_tags=[
                                               "PatientID",
                                               "InstanceNumber",
                                               "SharedFunctionalGroupsSequence",
                                               "StudyInstanceUID"])
            study_instance_uid = ds1.StudyInstanceUID
            image_laterality = \
                ds1.SharedFunctionalGroupsSequence[0].FrameAnatomySequence[
                    0].FrameLaterality
            image_shape = (
                int(ds.NumberOfFrames), int(ds.Rows), int(ds.Columns))
        else:
            ds1 = dicom.filereader.dcmread(img_path, stop_before_pixels=True,
                                           specific_tags=[
                                               "PatientID",
                                               "InstanceNumber",
                                               "ImageLaterality",
                                               "StudyInstanceUID"])
            study_instance_uid = ds1.StudyInstanceUID
            image_laterality = ds1.get("ImageLaterality")
            image_shape = (int(ds.Rows), int(ds.Columns))

        # img_index = cls._image_id(os.path.basename(img_path))
        # label = cls._labels[int(img_index)]

        # get the dicom name
        sop_uid = os.path.basename(img_path)
        if cls.dl.dicom_3d_substring != '' and cls.dl.dicom_2d_substring != '':
            # remove the dicom_3d_substring or dicom_2d_substring from the beginning
            sop_uid = sop_uid.replace(cls.dl.dicom_3d_substring + '.', "")
            sop_uid = sop_uid.replace(cls.dl.dicom_2d_substring + '.', "")

        # get the image label
        label = cls.get_label(sop_uid)

        if pixels is True:
            img = ds.pixel_array
        if dicom_header is False:
            ds = None
        dictionary = {'filePath': img_path,
                      'SOPInstanceUID': sop_uid,
                      'StudyInstanceUID': study_instance_uid,
                      'PatientID': ds1.PatientID,
                      'InstanceNumber': ds1.InstanceNumber,
                      'label': label,
                      'imageLaterality': image_laterality,
                      'shape': image_shape,
                      'metadata': ds,
                      'pixels': img}
        dot_dictionary = SimpleNamespace(**dictionary)
        if timing:
            cls.timing(t0, 'get_image')
        return dot_dictionary

    # --------------------------------------------------------------------------
    @classmethod
    def next_image(
            cls,
            _2d=False,
            _3d=False,
            label=None,
            randomize=False,
            timing=False
    ) -> Generator[SimpleNamespace, None, None]:
        """Generator to filter and return iteratively all the filtered images in
         the dataset

         Parameters
         ----------
         _2d : bool
            (default is False, this or _3D or both needs to be set to True)
            Filter flag used to add 2D images to returned set of Dicoms
         _3d : bool
            (default is False, this or _2D or both needs to be set to True)
            Filter flag used to add 3D images to returned set of Dicoms
         label : str
            (default is None which is all labeled images)
            options are IndexCancer, NonCancer, PreIndexCancer
         randomize : bool
            (default is False) If True will randomize the order of the
            returned images
         timing : bool
            (default is False) If true will time execution of method,
            else will not

         Returns
         -------
         image : generator
            [np.array just the pixels, info]
            info is dictionary: {'label': LABELS.Cancer, 'filepath':... 'shape'...}

        """
        t0 = time.time()
        generator_map = {}
        image_index_list = {}
        if label is not None:
            for i in range(len(cls._labels)):
                if cls._labels[i] == label:
                    image_index_list[i] = cls._image_paths[i]
        else:
            for i in range(len(cls._labels)):
                image_index_list[i] = cls._image_paths[i]

        if _2d and _3d:
            for k, v in image_index_list.items():
                generator_map[k] = v
        elif _2d:
            for k, v in image_index_list.items():
                if os.path.basename(v).startswith(cls.dl.dicom_2d_substring):
                    generator_map[k] = v
                elif cls.dl.dicom_2d_substring == '':
                    generator_map[k] = v
        else:
            for k, v in image_index_list.items():
                if os.path.basename(v).startswith(cls.dl.dicom_3d_substring):
                    generator_map[k] = v

        # shuffle the generator
        dicoms = list(generator_map.items())
        if randomize is True:
            random.shuffle(dicoms)

        for k, v in dicoms:
            img_path = str(v)
            test_label = cls._labels[int(k)]
            if test_label == label or label is None:
                ds = cls.get_image(path=img_path)
                try:
                    yield ds
                except StopIteration:
                    break
            else:
                continue
        if timing:
            cls.timing(t0, 'next_image_all')

    # --------------------------------------------------------------------------
    @classmethod
    def filter_data(
            cls,
            _2d=True,
            _3d=True,
            shapes: list = None,
            row_max: int = None,
            row_min: int = None,
            col_max: int = None,
            col_min: int = None,
            frames_max: int = None,
            frames_min: int = None,
            age_range: list = None,
            labels: list = None,
            studies: list = None,
            manufacturers: list = None,
            timing=False
    ):
        """Filters the dataset to include only Dicoms that match the specified
        filter flags. Configures internal class data in a way that all the Data
        methods work the same, but only will be on the filtered data.
        To reset the filtered data to the original dataset, use the
        reset_data method.

        Parameters
        ----------
        _2d : int
            (default is True, includes 2D images)
            Filter flag used to remove 2D images
        _3d : str
            (default is True, includes 3D images)
            Filter flag used to remove 3D images
        shapes: list['(X1,Y1)', '(X2,Y2)', ..., (Xn,Yn,Zn)']
            (default is None, includes all shapes)
            specify the shapes of the images to include
        row_max: int
            (default is None, includes all rows)
            Filter flag used to remove images with rows greater than the
            specified value
        row_min: int
            (default is None, includes all rows)
            Filter flag used to remove images with rows less than the
            specified value
        col_max: int
            (default is None, includes all columns)
            Filter flag used to remove images with columns greater than the
            specified value
        col_min: int
            (default is None, includes all columns)
            Filter flag used to remove images with columns less than the
            specified value
        frames_max: int
            (default is None, includes all frames)
            Filter flag used to remove images with frames greater than the
            specified value
        frames_min: int
            (default is None, includes all frames)
            Filter flag used to remove images with frames less than the
            specified value
        age_range: list[min, max]
            (Default is None, include all ages)
            specify the age range in years to filter images by age
        labels: list
            (default is None, includes all labels)
            specify the labels of the images to include
        studies: list
            (default is None, includes all studies)
            specify the studies of the images to include
        manufacturers: list
            (default is None, includes all manufacturers)
            specify the manufacturers of the images to include
        timing: bool
            (default is False) If true will time execution of method,
            else will not
        """
        t0 = time.time()
        # build the filtered_flags dictionary
        cls._filtered_flags = {
            '2D': _2d,
            '3D': _3d,
            'shapes': shapes,
            'row_max': row_max,
            'row_min': row_min,
            'col_max': col_max,
            'col_min': col_min,
            'frames_max': frames_max,
            'frames_min': frames_min,
            'age_range': age_range,
            'labels': labels,
            'studies': studies,
            'manufacturers': manufacturers
        }
        label_to_path_map = {}
        dimension_map = {}
        filtered_map = {}
        temp_labels = {}
        # remove the labels that are not in the filtered_flags first
        if labels is not None:
            for label in labels:
                for i in range(len(cls._labels)):
                    if cls._labels[i] == label:
                        temp_labels[i] = cls._labels[i]
                        label_to_path_map[i] = cls._image_paths[i]
        else:
            for i in range(len(cls._labels)):
                temp_labels[i] = cls._labels[i]
                label_to_path_map[i] = cls._image_paths[i]
        # remove the dimensions that are not in the filtered_flags
        if _2d and _3d:
            for k, v in label_to_path_map.items():
                if studies is not None:
                    for study in studies:
                        if study in v:
                            dimension_map[k] = v
                else:
                    dimension_map[k] = v
        elif _2d:
            for k, v in label_to_path_map.items():
                if os.path.basename(v).startswith(cls.dl.dicom_2d_substring) or \
                        cls.dl.dicom_2d_substring == '':
                    if studies is not None:
                        for study in studies:
                            if study in v:
                                dimension_map[k] = v
                    else:
                        dimension_map[k] = v
        else:
            for k, v in label_to_path_map.items():
                if os.path.basename(v).startswith(cls.dl.dicom_3d_substring) or \
                        cls.dl.dicom_3d_substring == '':
                    if studies is not None:
                        for study in studies:
                            if study in v:
                                dimension_map[k] = v
                    else:
                        dimension_map[k] = v
        # next filter by the shapes of the images
        if shapes is not None:
            pattern = ''
            for shape in shapes:
                print(f'Filtering for shape: {shape}')
                if shape != shapes[-1]:
                    pattern += shape + '|'
                else:
                    pattern += shape
            count = 0
            for k, v in dimension_map.items():
                img_path = str(v)
                test_label = cls._labels[int(k)]
                if labels is None or labels.__contains__(test_label):
                    ds = dicom.filereader.dcmread(
                        img_path,
                        stop_before_pixels=True,
                        force=True,
                        specific_tags=['Columns', 'Rows', 'NumberOfFrames'])
                    rows = ds.Rows
                    cols = ds.Columns
                    if os.path.basename(img_path).startswith(
                            cls.dl.dicom_3d_substring):
                        frames = ds.NumberOfFrames
                        shape = str(
                            '(' + str(rows) + ',' + str(cols) + ',' + str(
                                frames) + ')')
                    else:
                        shape = str('(' + str(rows) + ',' + str(cols) + ')')
                    if re.search(pattern, shape) is not None:
                        filtered_map[k] = v
                    if timing:
                        count += 1
                        if count % 100 == 0:
                            percentage_done = int(
                                count / len(dimension_map) * 100)
                            print(f'{percentage_done}% done')

        elif shapes is None and (col_max is not None and row_max is not None) or \
                (col_min is not None and row_min is not None):
            count = 0
            for k, v in dimension_map.items():
                img_path = str(v)
                test_label = cls._labels[int(k)]
                if labels is None or labels.__contains__(test_label):
                    ds = dicom.filereader.dcmread(
                        img_path,
                        stop_before_pixels=True,
                        force=True,
                        specific_tags=['Columns', 'Rows', 'NumberOfFrames'])
                    rows = ds.Rows
                    cols = ds.Columns
                    frames = 1
                    if os.path.basename(img_path).startswith(
                            cls.dl.dicom_3d_substring):
                        frames = ds.NumberOfFrames
                    if (col_min is None or col_min <= cols) and \
                            (col_max is None or cols <= col_max) and \
                            (row_min is None or row_min <= rows) and \
                            (row_max is None or rows <= row_max) and \
                            (frames_min is None or frames_min <= frames) and \
                            (frames_max is None or frames <= frames_max):
                        filtered_map[k] = v
                    if timing:
                        count += 1
                        percentage_done = count / len(dimension_map) * 100
                        if count % 1000 == 0:
                            print(f'{percentage_done}% done')
        else:
            filtered_map = dimension_map

        # filter by the age range
        temp_map = {}
        count = 0
        if age_range is not None:
            # turning years into months to match how age is stored in dicom
            age_range[0] = int(age_range[0] * 12)
            age_range[1] = int(age_range[1] * 12)
            for k, v in filtered_map.items():
                img_path = str(v)
                test_label = cls._labels[int(k)]
                if labels is None or labels.__contains__(test_label):
                    ds = dicom.filereader.dcmread(
                        img_path,
                        stop_before_pixels=True,
                        force=True,
                        specific_tags=['PatientAge', 'PatientSex'])
                    if ds.PatientAge is not None and ds.PatientAge != '':
                        age = int(ds.PatientAge[:-1])
                        if age_range[0] <= age <= age_range[1]:
                            temp_map[k] = v
                    if timing:
                        count += 1
                        percentage_done = count / len(filtered_map) * 100
                        if count % 1000 == 0:
                            print(f'{percentage_done}% done')
            filtered_map = temp_map

        # filter by manufacturer
        temp_map = {}
        count = 0
        if manufacturers is not None:
            for k, v in filtered_map.items():
                img_path = str(v)
                test_label = cls._labels[int(k)]
                if labels is None or labels.__contains__(test_label):
                    ds = dicom.filereader.dcmread(
                        img_path,
                        stop_before_pixels=True,
                        force=True,
                        specific_tags=['Manufacturer'])
                    if ds.Manufacturer is not None and ds.Manufacturer != '':
                        if manufacturers.__contains__(ds.Manufacturer):
                            temp_map[k] = v
                    if timing:
                        count += 1
                        percentage_done = count / len(filtered_map) * 100
                        if count % 1000 == 0:
                            print(f'{percentage_done}% done')
            filtered_map = temp_map

        cls._image_paths = []
        cls._labels = {}
        cls._study_paths = []
        study_path_map = {}
        image_id = 0
        count = 0
        for k, v in filtered_map.items():
            img_path = str(v)
            cls._image_paths.append(img_path)
            cls._labels[image_id] = temp_labels[int(k)]
            image_id += 1
            study_path = os.path.dirname(img_path)
            if study_path_map.__contains__(study_path):
                continue
            else:
                study_path_map[study_path] = image_id
                cls._study_paths.append(study_path)
            if timing:
                count += 1
                percentage_complete = count / len(filtered_map) * 100
                if count % 1000 == 0:
                    print('Now Processing filtered images: ' + str(
                        percentage_complete) + '%')
        cls._init_stats()
        if timing:
            cls.timing(t0, 'filter_images')

    @classmethod
    def reset_data(cls, timing=False):
        """Resets the data to the original data
        """
        t0 = time.time()
        cls._filtered_flags = {}
        cls._image_paths = []
        cls._labels = {}
        cls._study_paths = []
        cls._init()
        if timing:
            cls.timing(t0, 'reset_data')

    # --------------------------------------------------------------------------
    @classmethod
    def partition_data(cls,
                       num_partitions,
                       randomized=True,
                       save_paths_to_files=True,
                       save_directory=None,
                       file_name_prefix=None,
                       timing=False):
        """Partitions the data into the specified number of partitions.

        Parameters
        ----------
        num_partitions : int
            The number of partitions to split the data into.
        randomized : bool
            If True, the data will be randomized before partitioning.
        save_paths_to_files : bool
            If True, the paths to the images will be saved to a file for each
            partition.
        save_directory : str
            The directory to save the files to. If None, the files will be
            saved to the current directory.
        file_name_prefix : str
            The prefix to use for the file names. If None, the prefix will be
            'partition_'.
        timing : bool
            If True, timing information will be printed.

        Returns
        -------
        A list of lists of image paths for each partition.
        """
        t0 = time.time()
        image_paths = cls._image_paths.copy()
        if randomized:
            random.shuffle(image_paths)

        if file_name_prefix is None:
            file_name_prefix = 'partition_'
        num_images = len(image_paths)
        num_images_per_partition = int(num_images / num_partitions)
        partitioned_paths = []
        for i in range(num_partitions):
            partitioned_paths.append([])
        for i in range(num_images):
            partition_id = int(i / num_images_per_partition)
            partitioned_paths[partition_id].append(image_paths[i])
        if save_paths_to_files:
            if save_directory is None:
                # get the current directory to save the files to
                save_directory = os.path.dirname(os.path.realpath(__file__))
            print('Saving partitioned paths to files in: ' + save_directory)
            for i in range(num_partitions):
                partition_path = os.path.join(save_directory,
                                              file_name_prefix + str(
                                                  i) + '.txt')
                with open(partition_path, 'w') as f:
                    for path in partitioned_paths[i]:
                        f.write(path + '\n')
        if timing:
            cls.timing(t0, 'partition_data')
        return partitioned_paths

    # --------------------------------------------------------------------------
    @classmethod
    def random_sample(cls, size, timing=False):
        """Randomly samples the data.

        Parameters
        ----------
        size : int
            The number of images to sample.
        timing : bool
            If True, timing information will be printed.

        Returns
        -------
        A list of image paths.
        """
        t0 = time.time()
        image_paths = cls._image_paths.copy()
        random.shuffle(image_paths)
        if timing:
            cls.timing(t0, 'random_sample')
        return image_paths[:size]

    # --------------------------------------------------------------------------
    @classmethod
    def set_dicom_tags(cls, tags):
        """Sets the tags to be used when reading dicom files with python
        iterators and generators.

        Parameters
        ----------
        tags : list of str
            The tags to be used when reading dicom files with python iterators
            and generators.
        """
        cls.dl.dicom_tags = tags

    @staticmethod
    def timing(
            t0,
            label: str
    ):
        """ Prints the time it takes to perform a certain action

        Parameters
        ----------
        t0 : float
            the time when the action was performed
        label : str
            the label of the action
        """
        print('{:<25s}{:<10s}{:>10f}{:^5s}'
              .format(label, '...took ', time.time() - t0, ' seconds'))
