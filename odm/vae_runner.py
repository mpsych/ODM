import argparse
import pydicom as dicom
from odm import OutlierDetector
from .fivebhist_runner import *

# constants for feature type and normalization type
FEAT = 'hist'
NORM = 'minmax'


def load_data_from_text_file(file_path):
    """
    Loads the DICOM files from a text file of paths.

    Parameters
    ----------
    file_path : str
        The path to the text file containing the paths of the DICOM files.

    Returns
    -------
    list
        A list of DICOM files.
    """
    with open(file_path, 'r') as f:
        paths = f.readlines()

    paths = [path.strip() for path in paths]  # remove the new line characters

    data = [dicom.read_file(path) for path in paths]  # load the DICOM files

    return data


def vae_runner(data_root, contamination=0.015, verbose=False, norm_type=NORM):
    """
    Runs the Variational AutoEncoder (VAE) algorithm on given data.

    Parameters
    ----------
    data_root : str
        The path to the data.
    contamination : float, optional
        The proportion of outliers in the data. Default is 0.015.
    verbose : bool, optional
        Whether to print progress messages to stdout. Default is False.
    norm_type : str, optional
        The type of normalization to be applied. Default is NORM (minmax).
    """
    # load the data after running 5bhist algorithm
    data_dict = load_data_dict(data_root)
    imgs = get_pixel_list(data_dict)

    # creating features from the images
    feats = Features.get_features(imgs,
                                  feature_type=FEAT,
                                  norm_type=norm_type)

    # run the outlier detection algorithm
    OutlierDetector.detect_outliers(features=feats,
                                    imgs=imgs,
                                    pyod_algorithm='VAE',
                                    contamination=contamination,
                                    verbose=verbose)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs the Variational AutoEncoder (VAE) algorithm on given data.')
    parser.add_argument('data_root', type=str, help='The path to the data.')
    parser.add_argument('--contamination', type=float, default=0.015,
                        help='The proportion of outliers in the data.')
    parser.add_argument('--verbose', action='store_true',
                        help='Whether to print progress messages to stdout.')
    parser.add_argument('--norm_type', type=str, default=NORM,
                        help='The type of normalization to be applied.')
    args = parser.parse_args()

    vae_runner(args.data_root, args.contamination, args.verbose, args.norm_type)
