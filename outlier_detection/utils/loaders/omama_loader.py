from data_loader import DataLoader
from types import SimpleNamespace

__author__ = "Ryan Zurrin"

label_types = {'CANCER': 'IndexCancer', 'NONCANCER': 'NonCancer',
               'PREINDEX': 'PreIndexCancer'}
Label = SimpleNamespace(**label_types)


class OmamaLoader(DataLoader):
    """
    OmamaLoader class is a subclass of the super class DataLoader, and holds
    information that is specific to the Omama dataset stored at our compute
    cluster location. I will show an example of how to build a child class of
    your own, all you should have to change if you are using the Omama
    dataset, is the location of where you are storing your data. If you are
    using your own mammogram dicom data you will need to make sure your csv
    file is formatted as specified in the DataLoader class.
    """

    def __init__(self,
                 data_paths=None,
                 csv_paths=None,
                 study_folder_names=None,
                 dicom_tags=None,
                 dicom_2d_substring=None,
                 dicom_3d_substring=None,
                 cache_paths=None,
                 patient_identifier=None,
                 cancer_identifier=None,
                 caselist_path=None,
                 config_num=None,
                 config_loc=None
                 ):
        """
        Initialize the OmamaLoader class.

        Parameters
        ----------
        data_paths : list: str
            paths to the data
        csv_paths : list: str
            paths to the csv files
        study_folder_names : list: str
            names of the study folders
        dicom_tags : list: str
            tags that are used to extract the data from the dicom files
        dicom_2d_substring : str
            substring that is used to find the 2d dicom files
        dicom_3d_substring : str
            substring that is used to find the 3d dicom files
        cache_paths : list: str
            paths to the pickle files which are used for caching
        config_num : int
            The option number corresponds to a set of paths to specific data.
            config_num options are:
            1: Full Omama dataset -> 967991 images
            2: Omama dataset for 2D whitelist -> 176494 images
            3: Omama dataset for 3D whitelist -> 57572 images
            4: Omama dataset for 2D + 3D WL -> 234066 images
            5: Test dataset A for outlier detection -> 100 images with 8% errors
            6: Test dataset B for outlier detection -> 100 images with 13% errors
            7: Test dataset C for outlier detection -> 100 images with 24% errors
            8: Partition 0 for outlier detection megarun -> 44123 images
            9: Partition 1 for outlier detection megarun -> 44123 images
            10: Partition 2 for outlier detection megarun -> 44123 images
            11: Partition 3 for outlier detection megarun -> 44123 images
            12: Blacklist Phase1 run for outlier detection megarun -> 11648 images
            13: Whitelist Phase2 run for outlier detection megarun -> 2000 images
            14: Whitelist Phase2 run for outlier detection megarun -> 165104 images
            15: partition 0 for outlier detection phase 3 -> 41211 images
            16: partition 1 for outlier detection phase 3 -> 41211 images
            17: partition 2 for outlier detection phase 3 -> 41211 images
            18: partition 3 for outlier detection phase 3 -> 41211 images
            19: Whitelist after phase 3 -> 163000 images
            20: partition 0 for outlier detection phase 4 -> 40750 images
            21: partition 1 for outlier detection phase 4 -> 40750 images
            22: partition 2 for outlier detection phase 4 -> 40750 images
            23: partition 3 for outlier detection phase 4 -> 40750 images
            24: phase3b mixed list => 1844 images
            25: Test Dataset D for outlier detection -> 100 images with 24% good images
            26: Dataset A* for ODLite -> 1000 Random images
            27: Dataset B* for ODLite -> 1000 Random images
            28. 2D Dataset after removal of bad images from no ML run -> 168368 images
            29. partition 0 for OD phase 1* -> 42092 images
            30. partition 1 for OD phase 1* -> 42092 images
            31. partition 2 for OD phase 1* -> 42092 images
            32. partition 3 for OD phase 1* -> 42092 images
            33. Dataset C* for ODLite -> 1000 Random images from filtered dataset
            34. Removed 4800 bad images from VAE OD run -> 4800 images
            35. Final 2D dataset after removal of bad images from VAE run -> 163568 images
            

        """
        if data_paths is None:
            data_paths = [r'/raid/data01/deephealth/dh_dh2/',
                          r'/raid/data01/deephealth/dh_dh0new/',
                          r'/raid/data01/deephealth/dh_dcm_ast/']
        else:
            data_paths = data_paths

        if csv_paths is None:
            csv_paths = [r'/raid/data01/deephealth/labels/dh_dh2_labels.csv',
                         r'/raid/data01/deephealth/labels/dh_dh0new_labels.csv',
                         r'/raid/data01/deephealth/labels/dh_dcm_ast_labels'
                         r'.csv']
        else:
            csv_paths = csv_paths

        if study_folder_names is None:
            study_folder_names = ['dh_dh2', 'dh_dh0new', 'dh_dcm_ast']
        else:
            study_folder_names = study_folder_names

        if config_num is None:
            config_num = 1
        else:
            config_num = config_num

        if dicom_tags is None:
            dicom_tags = ["SamplesPerPixel", "PhotometricInterpretation",
                          "PlanarConfiguration", "Rows", "Columns",
                          "PixelAspectRatio", "BitsAllocated",
                          "BitsStored",
                          "HighBit", "PixelRepresentation",
                          "SmallestImagePixelValue",
                          "LargestImagePixelValue",
                          "PixelPaddingRangeLimit", "SOPInstanceUID",
                          "RedPaletteColorLookupTableDescriptor",
                          "GreenPaletteColorLookupTableDescriptor",
                          "BluePaletteColorLookupTableDescriptor",
                          "RedPaletteColorLookupTableData",
                          "GreenPaletteColorLookupTableData",
                          "BluePaletteColorLookupTableData", "ICCProfile",
                          "ColorSpace", "PixelDataProviderURL",
                          "ExtendedOffsetTable", "NumberOfFrames",
                          "ExtendedOffsetTableLengths", "PixelData"]
        else:
            dicom_tags = dicom_tags

        if dicom_2d_substring is None:
            dicom_2d_substring = "DXm"
        else:
            dicom_2d_substring = dicom_2d_substring

        if dicom_3d_substring is None:
            dicom_3d_substring = "BT"
        else:
            dicom_3d_substring = dicom_3d_substring

        if patient_identifier is None:
            patient_identifier = "SOPInstanceUID"
        else:
            patient_identifier = patient_identifier

        if cancer_identifier is None:
            cancer_identifier = "StudyInstanceUID"
        else:
            cancer_identifier = cancer_identifier

        if config_loc is None:
            config_loc = "/raid/mpsych/config_files/config_nums.ini"
        else:
            config_loc = config_loc

        if cache_paths is None and caselist_path is None:
            # check if config_loc is a csv or ini file and load accordingly
            if config_loc.endswith(".csv"):
                print("Loading config data from csv file")
                cache_paths, caselist_path = self.load_config_data_from_csv(
                    config_num, config_loc)
            elif config_loc.endswith(".ini"):
                print("Loading config data from ini file")
                cache_paths, caselist_path = self.load_config_data_from_ini(
                    config_num, config_loc)

        super().__init__(
            data_paths,
            csv_paths,
            study_folder_names,
            cache_paths,
            dicom_tags,
            dicom_2d_substring,
            dicom_3d_substring,
            patient_identifier,
            cancer_identifier,
            caselist_path,
            config_num=config_num,
            config_loc=config_loc,
        )
