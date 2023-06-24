from .__configloc__ import CONFIG_LOC
from configparser import ConfigParser
from .feature_extractor import Features
from .outlier_detector import OutlierDetector
from .utils import validate_inputs, print_properties
from keras.losses import get
from keras.losses import mse
from PIL import Image
from tqdm import tqdm

import argparse
import ast
import datetime
import logging
import numpy as np
import os
import time

logger = logging.getLogger(__name__)


def load_data_batch(files, timing):
    """Load a batch of DICOM files into a dictionary.

    Parameters
    ----------
    files : list
        A list of file paths.
    timing : bool
        Whether to time the loading process.

    Returns
    -------
    data_dict : dict
        A dictionary containing the loaded images and their file paths.
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
                data_dict[index] = [dicom.dcmread(file).pixel_array, file]  # DICOM
            elif file.endswith(tuple(img_formats)):
                with Image.open(file) as img:
                    data_dict[index] = [np.array(img), file]  # Non-DICOM
            else:
                logging.info(f"File {file} is not a valid image file.")
        except Exception as e:
            logging.info(f"Error reading file {file}: {e}")

    if timing:
        end = time.time()
        logging.info(
            f"Time to load {len(files)} files: {datetime.timedelta(seconds=end - t0)}"
        )
    return data_dict


def get_pixel_list(data, timing) -> list:
    """Generate a list of pixel arrays from a dictionary of DICOM data.

    Parameters
    ----------
    data : dict
        A dictionary of DICOM data.
    timing : bool
        Whether to time the process.

    Returns
    -------
    images : list
        A list of pixel arrays.
    """
    t0 = time.time()
    images = []
    for key in tqdm(data, desc="Generating pixel arrays"):
        try:
            images.append(data[key][0])
        except Exception as e:
            logging.info(f"Error reading file {data[key][1]}: {e}")

    if timing:
        logging.info(
            f"Time to generate pixel arrays: "
            f"{datetime.timedelta(seconds=time.time() - t0)}"
        )
    return images


def get_hyperparameters(timing=False) -> dict:
    """Fetches the hyperparameters from the configuration file.

    Parameters
    ----------
    timing : bool
        Whether to time the process.

    Returns
    -------
    hyperparameters : dict
        A dictionary of hyperparameters.
    """
    t0 = time.time()

    # read the config file
    config_ = ConfigParser()
    config_.read(CONFIG_LOC)

    # Fetch hyperparameters as strings
    raw_values = {
        param: config_.get("HYPERPARAMS", param, fallback=None)
        for param in [
            "latent_dim",
            "hidden_activation",
            "output_activation",
            "loss",
            "optimizer",
            "epochs",
            "batch_size",
            "dropout_rate",
            "l2_regularizer",
            "validation_size",
            "preprocessing",
            "verbose",
            "contamination",
            "gamma",
            "capacity",
            "random_state",
            "encoder_neurons",
            "decoder_neurons",
        ]
    }

    # Prepare default values
    default_values = {
        "latent_dim": 2,
        "hidden_activation": "relu",
        "output_activation": "sigmoid",
        "loss": mse,
        "optimizer": "adam",
        "epochs": 100,
        "batch_size": 32,
        "dropout_rate": 0.2,
        "l2_regularizer": 0.1,
        "validation_size": 0.1,
        "preprocessing": True,
        "verbose": 1,
        "contamination": 0.1,
        "gamma": 1.0,
        "capacity": 0.0,
        "random_state": None,
        "encoder_neurons": None,
        "decoder_neurons": None,
    }

    # Convert string values to correct types, with fallbacks for empty fields
    values = {}
    for param, raw_val in raw_values.items():
        if raw_val == "":
            # Use default value if field is empty
            values[param] = default_values[param]
        else:
            try:
                if param in ["latent_dim", "epochs", "batch_size", "verbose"]:
                    # These parameters should be integers
                    values[param] = int(raw_val)
                elif param in [
                    "dropout_rate",
                    "l2_regularizer",
                    "validation_size",
                    "contamination",
                    "gamma",
                    "capacity",
                ]:
                    # These parameters should be floats
                    values[param] = float(raw_val)
                elif param in ["random_state", "encoder_neurons", "decoder_neurons"]:
                    # These parameters should be evaluated as Python expressions
                    values[param] = ast.literal_eval(raw_val)
                elif param in ["loss"]:
                    # These parameters should be evaluated as Keras loss
                    # functions
                    values[param] = get(raw_val)
                else:
                    # All other parameters are kept as strings
                    values[param] = raw_val
            except Exception as e:
                logging.error(f"Error processing parameter {param}: {e}")
                return None
    logging.info("Hyperparameters fetched.")

    print_properties("Hyperparameters", **values)

    if timing:
        logging.info(
            f"Time to fetch hyperparameters: "
            f"{datetime.timedelta(seconds=time.time() - t0)}"
        )
    return values


def vae_runner(
    log_dir, caselist, batch_size, log_to_terminal, timing
) -> tuple[list[str], list[str]]:
    """Run the VAE algorithm on a list of files.

    Parameters:
    log_dir (str): The path to the directory where the log file will be written.
    caselist (str): The path to a file containing a list of file paths to be
        processed.
    batch_size (int): The number of files to process at a time.
    log_to_terminal (bool): Whether to logging.info verbose output.
    timing (bool): Whether to logging.info timing information.
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

    logging.info(f"Number of files to process: {len(all_files)}")

    # Load the hyperparameters
    values = get_hyperparameters(timing)

    # Process the files in batches
    for i in range(0, len(all_files), batch_size):
        file_batch = all_files[i : i + batch_size]

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
            log_to_terminal=log_to_terminal,
            timing=timing,
            **values,
        )

        # Add the decision scores, labels, and paths to the master dictionaries
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
        logging.info(
            f"Time to run VAE on {len(all_files)} "
            f"files: {datetime.timedelta(seconds=time.time() - t0)}"
        )
    return good_img_paths, bad_img_paths


if __name__ == "__main__":
    """Main entry point of the program. Parses command-line arguments,
    reads the config file, overwrites config values if command line arguments
    are provided, and then runs the VAE algorithm.

    Supports the following command-line arguments:
        --log_dir (str): The path to the directory where the log file will be
        written.
        --caselist (str): Path to the text file containing the paths of the
        DICOM files.
        --batch_size (int, optional): The number of files to process in each
        batch.
        --good_output (str, optional): The path to the text file to write the
        final list of good files to.
        --bad_output (str, optional): The path to the text file to write the
        final list of bad files to.
    """
    # read the config file
    config = ConfigParser()
    config.read(CONFIG_LOC)

    parser = argparse.ArgumentParser(
        description="Runs the Variational AutoEncoder (VAE) algorithm on " "given data."
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
        help="The path to the text file containing the paths of the DICOM " "files.",
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
        help="Whether to logging.info progress messages to stdout.",
    )

    args = parser.parse_args()

    validate_inputs(**vars(args))

    print_properties("VAE Runner", **vars(args))

    good_paths, bad_paths = vae_runner(
        log_dir=args.log_dir,
        caselist=args.caselist,
        batch_size=args.batch_size,
        log_to_terminal=args.verbose,
        timing=args.timing,
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
        logging.info(f"Error writing to file {args.good_output}: {e}")

    try:
        with open(args.bad_output, "w") as f:
            for path in bad_paths:
                f.write(f"{path}\n")
    except Exception as e:
        logging.info(f"Error writing to file {args.bad_output}: {e}")
