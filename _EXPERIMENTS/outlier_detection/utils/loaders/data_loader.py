# class to hold all the specific data that is used in the data class
import csv
import configparser


class DataLoader:
    """DataLoader class to hold all the specific path location data that is
    used in the data class"""

    def __init__(
        self,
        data_paths: list,
        csv_paths: list,
        study_folder_names: list,
        cache_paths: list,
        dicom_tags: list,
        dicom_2d_substring: str,
        dicom_3d_substring: str,
        patient_identifier: str,
        cancer_identifier: str,
        caselist_path: str = None,
        config_num: int = 0,
        config_loc: str = None,
    ):
        self._data_paths = data_paths
        self._csv_paths = csv_paths
        self._study_folder_names = study_folder_names
        self._cache_paths = cache_paths
        self._dicom_tags = dicom_tags
        self._dicom_2d_substring = dicom_2d_substring
        self._dicom_3d_substring = dicom_3d_substring
        self._patient_identifier = patient_identifier
        self._cancer_identifier = cancer_identifier
        self._caselist_path = caselist_path
        self._config_num = config_num
        self._config_loc = config_loc

        """
        Initialize the DataLoader object which is used in the Data class

        Parameters
        ----------
        data_paths : list: str
            paths to the data
        csv_paths : list: str
            paths to the csv files
        study_folder_names : list: str
            names of the study folders
        pickle_path : str
            path to the pickle file which is used for saving and loading cache
        dicom_tags : list: str
            tags that are used to extract the data from the dicom files
        dicom_2d_substring : str
            substring that is used to find the 2d dicom files
        dicom_3d_substring : str
            substring that is used to find the 3d dicom files
        whitelist_path : str
            path to the whitelist file
        """

    def __str__(self):
        """Returns a string representation of the DataLoader object"""
        return (
            "DataLoader(data_paths={}, csv_paths={}, pickle_path={}, "
            "dicom_tags={}, dicom_2d_substring={}, dicom_3d_substring={}, "
            "whitelist_path={}, config_num={}, config_csv_path={} )".format(
                self._data_paths,
                self._csv_paths,
                self._cache_paths,
                self._dicom_tags,
                self._dicom_2d_substring,
                self._dicom_3d_substring,
                self._caselist_path,
                self._patient_identifier,
                self._cancer_identifier,
                self._config_num,
                self._config_loc,
            )
        )

    @property
    def data_paths(self):
        return self._data_paths

    @data_paths.setter
    def data_paths(self, value):
        self._data_paths = value

    @property
    def csv_paths(self):
        return self._csv_paths

    @csv_paths.setter
    def csv_paths(self, value):
        self._csv_paths = value

    @property
    def study_folder_names(self):
        return self._study_folder_names

    @study_folder_names.setter
    def study_folder_names(self, value):
        self._study_folder_names = value

    @property
    def patient_identifier(self):
        return self._patient_identifier

    @patient_identifier.setter
    def patient_identifier(self, value):
        self._patient_identifier = value

    @property
    def cancer_identifier(self):
        return self._cancer_identifier

    @cancer_identifier.setter
    def cancer_identifier(self, value):
        self._cancer_identifier = value

    @property
    def cache_paths(self):
        return self._cache_paths

    @cache_paths.setter
    def cache_paths(self, value):
        self._cache_paths = value

    @property
    def dicom_tags(self):
        return self._dicom_tags

    @dicom_tags.setter
    def dicom_tags(self, value):
        self._dicom_tags = value

    @property
    def dicom_2d_substring(self):
        return self._dicom_2d_substring

    @dicom_2d_substring.setter
    def dicom_2d_substring(self, value):
        self._dicom_2d_substring = value

    @property
    def dicom_3d_substring(self):
        return self._dicom_3d_substring

    @dicom_3d_substring.setter
    def dicom_3d_substring(self, value):
        self._dicom_3d_substring = value

    @property
    def caselist_path(self):
        return self._caselist_path

    @caselist_path.setter
    def caselist_path(self, value):
        self._caselist_path = value

    @property
    def config_num(self):
        return self._config_num

    @config_num.setter
    def config_num(self, value):
        self._config_num = value

    @property
    def config_csv_path(self):
        return self._config_loc

    @config_csv_path.setter
    def config_csv_path(self, value):
        self._config_loc = value

    @staticmethod
    def load_config_data_from_csv(config_num, config_csv_path):
        """Loads the cache paths and the case list path from the csv file. If the
        config_num is not found in the csv file, the function returns None for
        both the cache_paths and the case_list_path. If the create_new flag is
        set to True, and the config_num is not found in the csv file, the
        function will create a new row in the csv file with the config_num and
        the cache_paths and case_list_path set using the paths from the
        last row in the csv file, with using datetime to make the names of new
        cache paths and case list path unique.
        """
        cache_paths = None
        case_list_path = None
        with open(config_csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == str(config_num):
                    cache_paths = row[1]
                    sop_uid_cache_path = row[2]
                    case_list_path = row[3]
                    cache_paths = [cache_paths, sop_uid_cache_path]
                    break

        if case_list_path == "None":
            case_list_path = None

        if cache_paths is None and case_list_path is None:
            # throw error saying there is no data to load
            raise ValueError("No data found for config_num: {}".format(config_num))

        return cache_paths, case_list_path

    @staticmethod
    def load_config_data_from_ini(config_num, config_ini_loc):
        """
        Loads the cache paths and the case list path from the ini file.
        """
        config = configparser.ConfigParser()
        config.read(config_ini_loc)
        cache_paths = []
        case_list_path = None
        for key in config[str(config_num)]:
            if key == "cache_path1":
                cache_paths.append(config[str(config_num)][key])
            elif key == "cache_path2":
                cache_paths.append(config[str(config_num)][key])
            elif key == "caselist_path":
                case_list_path = config[str(config_num)][key]
            else:
                continue

        if case_list_path == "None":
            case_list_path = None

        if cache_paths is None and case_list_path is None:
            # throw error saying there is no data to load
            raise ValueError("No data found for config_num: {}".format(config_num))

        return cache_paths, case_list_path
