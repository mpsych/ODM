import os
import numpy as np
import datetime
import time
from outlier_detector import OutlierDetector
import logging
import argparse
from configparser import ConfigParser
from tqdm import tqdm
from PIL import Image
from feature_extractor import *
from utils import *


logger = logging.getLogger(__name__)


def load_data_batch(files, timing):
    """
    Load a batch of DICOM files into a dictionary.

    Parameters:
    files (list): A list of file paths to be loaded.

    Returns:
    dict: A dictionary where the keys are indices and the values are tuples of DICOM data and the file path.
    """
    import pydicom as dicom
    t0 = time.time()
    img_formats = [
        ".png",
        ".jpg",
        ".jpeg",
        ".PNG",
        ".JPG",
        ".JPEG",
        ".tif",
        ".tiff",
        ".TIF",
        ".TIFF",
    ]
    data_dict = {}
    for index, file in tqdm(enumerate(files), desc="Loading files", total=len(files)):
        try:
            if file.endswith(".dcm") or file.endswith(".DCM") or file.endswith(""):
                data_dict[index] = [dicom.dcmread(
                    file).pixel_array, file]  # DICOM
            elif file.endswith(tuple(img_formats)):
                with Image.open(file) as img:
                    data_dict[index] = [np.array(img), file]  # Non-DICOM
            else:
                print(f"File {file} is not a valid image file.")
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    if timing:
        print(
            f"Time to load {len(files)} files: {datetime.timedelta(seconds=time.time() - t0)}"
        )
    return data_dict


def get_pixel_list(data, timing):
    """
    Generate a list of pixel arrays from a dictionary of DICOM data.

    Parameters:
    data (dict): A dictionary of DICOM data.
    """
    t0 = time.time()
    imgs = []
    for key in tqdm(data, desc="Generating pixel arrays"):
        try:
            imgs.append(data[key][0])
        except Exception as e:
            print(f"Error reading file {data[key][1]}: {e}")

    if timing:
        print(
            f"Time to generate pixel arrays: {datetime.timedelta(seconds=time.time() - t0)}"
        )
    return imgs


def vae_runner(log_dir, caselist, contamination, batch_size, verbose, timing):
    """
    Run the VAE algorithm on a list of files.

    Parameters:
    log_dir (str): The path to the directory where the log file will be written.
    caselist (str): The path to a file containing a list of file paths to be processed.
    contamination (float): The proportion of outliers in the data set.
    batch_size (int): The number of files to process at a time.
    verbose (bool): Whether to print verbose output.
    timing (bool): Whether to print timing information.
    """
    t0 = time.time()
    FEAT = "hist"
    NORM = "minmax"

    master_decision_scores = {}
    master_labels = {}
    master_paths = {}

    good_img_paths = []
    bad_img_paths = []

    # check if caselist is just a file name or a full path
    if not os.path.isfile(caselist):
        caselist = os.path.join(log_dir, caselist)

    # Read the list of file paths
    with open(caselist, "r") as f_:
        all_files = [path_.strip() for path_ in f_.readlines()]

    # Process the files in batches
    for i in range(0, len(all_files), batch_size):
        file_batch = all_files[i: i + batch_size]

        # Load the data batch after running 5bhist algorithm
        data_dict = load_data_batch(file_batch, timing)
        imgs = get_pixel_list(data_dict, timing)

        # Create features from the images
        feats = Features.get_features(
            imgs, feature_type=FEAT, norm_type=NORM, timing=timing
        )

        # Run the outlier detection algorithm
        decision_scores, labels = OutlierDetector.detect_outliers(
            features=feats,
            contamination=contamination,
            verbose=verbose,
            timing=timing,
        )

        # Add the decision scores, labels, and paths to the master dictionaries using the index as the key
        # so index i in each dictionary corresponds to the path, decision score, and label for the same file
        for index, path_ in enumerate(file_batch):
            master_decision_scores[i + index] = decision_scores[index]
            master_labels[i + index] = labels[index]
            master_paths[i + index] = path_

    # construct a master list of good paths
    for key in master_labels:
        if master_labels[key] == 0:
            good_img_paths.append(master_paths[key])
        else:
            bad_img_paths.append(master_paths[key])

    if timing:
        print(
            f"Time to run VAE on {len(all_files)} files: {datetime.timedelta(seconds=time.time() - t0)}"
        )
    return good_img_paths, bad_img_paths


if __name__ == "__main__":
    """
    Main entry point of the program. Parses command-line arguments, reads the config file,
    overwrites config values if command line arguments are provided, and then runs the VAE algorithm.

    Supports the following command-line arguments:
        log_dir (str): The path to the directory where the log file will be written.
        caselist (str): Path to the text file containing the paths of the DICOM files.
        verbose (bool, optional): Whether to print progress messages to stdout. Defaults to False.
        batch_size (int, optional): The number of files to process in each batch. Defaults to 100.
        good_output (str, optional): The path to the text file to write the final list of good files to.
        bad_output (str, optional): The path to the text file to write the final list of bad files to.
    """
    # read the config file
    config = ConfigParser()
    config.read("config.ini")

    parser = argparse.ArgumentParser(
        description="Runs the Variational AutoEncoder (VAE) algorithm on given data."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=config["DEFAULT"]["log_dir"],
        help="The path to the directory where log files will be saved.",
    )
    parser.add_argument(
        "--caselist",
        type=str,
        default=config["VAE"]["caselist"],
        help="The path to the text file containing the paths of the DICOM files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["VAE"]["batch_size"],
        help="The number of files to process in each batch.",
    )
    parser.add_argument(
        "--good_output",
        type=str,
        default=config["VAE"]["good_output"],
        help="The name of the final file where good paths are saved.",
    )
    parser.add_argument(
        "--bad_output",
        type=str,
        default=config["VAE"]["bad_output"],
        help="The name of the final file where bad paths are saved.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        default=config.getboolean("VAE", "timing"),
        help="Whether to time the execution of the algorithm.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=config.getboolean("VAE", "verbose"),
        help="Whether to print progress messages to stdout.",
    )

    args = parser.parse_args()

    validate_inputs(**vars(args))

    print_properties("VAE Runner", **vars(args))

    good_paths, bad_paths = vae_runner(
        args.log_dir,
        args.caselist,
        args.contamination,
        args.batch_size,
        args.verbose,
        args.timing
    )

    # check if output files are just file names or full paths
    if not os.path.isfile(args.good_output):
        args.good_output = os.path.join(args.log_dir, args.good_output)

    if not os.path.isfile(args.bad_output):
        args.bad_output = os.path.join(args.log_dir, args.bad_output)

    try:
        with open(args.good_output, "w") as f:
            for path in good_paths:
                f.write(f"{path}\n")
    except Exception as e:
        print(f"Error writing to file {args.good_output}: {e}")

    try:
        with open(args.bad_output, "w") as f:
            for path in bad_paths:
                f.write(f"{path}\n")
    except Exception as e:
        print(f"Error writing to file {args.bad_output}: {e}")
