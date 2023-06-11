import numpy as np

from odm import OutlierDetector
import logging
import argparse
import configparser
from tqdm import tqdm
from PIL import Image
import pydicom as dicom
from feature_extractor import *

logger = logging.getLogger(__name__)

# read the config file
config = configparser.ConfigParser()
config.read("config.ini")

TIMING = config["VAE"]["timing"]
LOG_DIR = config["5BHIST"]["log_dir"]


def load_data_batch(files):
    """
    Load a batch of DICOM files into a dictionary.

    Parameters:
    files (list): A list of file paths to be loaded.

    Returns:
    dict: A dictionary where the keys are indices and the values are tuples of DICOM data and the file path.
    """
    data_dict = {}
    for index, file in tqdm(
            enumerate(files), desc="Loading files", total=len(files)
    ):
        try:
            if file.endswith(".dcm") or file.endswith(".DCM") or file.endswith(""):
                data_dict[index] = [dicom.dcmread(file).pixel_array, file]  # DICOM
            else:
                with Image.open(file) as img:
                    data_dict[index] = [np.array(img), file]  # Non-DICOM
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    return data_dict


def get_pixel_list(data):
    imgs = []
    for key in tqdm(data, desc="Generating pixel arrays"):
        try:
            imgs.append(data[key][0].pixel_array)
        except Exception as e:
            print(f"Error reading file {data[key][1]}: {e}")
    return imgs


def vae_runner(caselist, contamination, verbose, batch_size):
    FEAT = "hist"
    NORM = "minmax"

    master_decision_scores = {}
    master_labels = {}
    master_paths = {}

    good_img_paths = []

    # Read the list of file paths
    with open(caselist, "r") as f_:
        all_files = [path_.strip() for path_ in f_.readlines()]

    # Process the files in batches
    for i in range(0, len(all_files), batch_size):
        file_batch = all_files[i: i + batch_size]

        # Load the data batch after running 5bhist algorithm
        data_dict = load_data_batch(file_batch)
        imgs = get_pixel_list(data_dict)

        # Create features from the images
        feats = Features.get_features(imgs, feature_type=FEAT, norm_type=NORM)

        # Run the outlier detection algorithm
        decision_scores, labels = OutlierDetector.detect_outliers(
            features=feats,
            paths=file_batch,
            contamination=contamination,
            verbose=verbose
        )

        # Add the decision scores, labels, and paths to the master dictionaries using the index as the key
        # so index i in each dictionary corresponds to the path, decision score, and label for the same file
        for index, path_ in enumerate(file_batch):
            master_decision_scores[i + index] = decision_scores[index]
            master_labels[i + index] = labels[index]
            master_paths[i + index] = path_

        # construct a master list of good paths
        good_img_paths = []
        for key in master_labels:
            if master_labels[key] == 0:
                good_img_paths.append(master_paths[key])

    return good_img_paths


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")

    parser = argparse.ArgumentParser(
        description="Runs the Variational AutoEncoder (VAE) algorithm on given data."
    )
    parser.add_argument(
        "caselist",
        type=str,
        default=config["VAE"]["caselist"],
        help="The path to the text file containing the paths of the DICOM files.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=config["VAE"]["contamination"],
        help="The proportion of outliers in the data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print progress messages to stdout.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["VAE"]["batch_size"],
        help="The number of files to process in each batch.",
    )
    parser.add_argument(
        "--final_file",
        type=str,
        default=config["VAE"]["final_file"],
        help="The name of the final file where good paths are saved.",
    )
    args = parser.parse_args()

    good_paths = vae_runner(
        args.caselist,
        args.contamination,
        args.verbose,
        args.batch_size,
    )

    with open(args.final_file, "w") as f:
        for path in good_paths:
            f.write(f"{path}\n")
