import os
import datetime
import time
import pydicom as dicom
import argparse
import logging
from tqdm import tqdm
import configparser
from feature_extractor import *

logger = logging.getLogger(__name__)

# read the config file
config = configparser.ConfigParser()
config.read("config.ini")

FEAT = config["5BHIST"]["feat"]
NORM = config["5BHIST"]["norm"]
LOG_DIR = config["5BHIST"]["log_dir"]
EXT = config["5BHIST"]["ext"]
BATCH_SIZE = int(config["5BHIST"]["batch_size"])
TIMING = config["5BHIST"]["timing"]


def file_batches_generator(directory, ext, batch_size):
    """
    Generator that yields batches of files from a directory and the total file count.

    Parameters:
    directory (str): Path of the directory to be searched.
    ext (str): File extension to be searched.
    batch_size (int): Size of the file batches to be returned.

    Yields:
    list: A list of file paths of size 'batch_size'.
    int: Total file count.
    """
    all_files = []
    for root, dirs, files in tqdm(
        os.walk(directory), desc="Walking through directories"
    ):
        all_files.extend(
            [os.path.join(root, file) for file in files if file.endswith(ext)]
        )

    total_files = len(all_files)
    for i in tqdm(range(0, total_files, batch_size), desc="Generating file batches"):
        yield all_files[i : i + batch_size], total_files


def load_data_batch(files):
    """
    Load a batch of DICOM files into a dictionary.

    Parameters:
    files (list): A list of file paths to be loaded.

    Returns:
    dict: A dictionary where the keys are indices and the values are tuples of DICOM data and the file path.
    """
    t0 = time.time()
    data_dict = {}
    for index, file in tqdm(
        enumerate(files), desc="Loading DICOM files", total=len(files)
    ):
        try:
            data_dict[index] = [dicom.dcmread(file), file]
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    if TIMING:
        logger.info(
            f"Time to load {len(files)} files: {datetime.timedelta(seconds=time.time() - t0)}"
        )

    return data_dict


def get_pixel_list(data):
    """
    Generate a list of pixel arrays from a dictionary of DICOM data.

    Parameters:
    data (dict): A dictionary of DICOM data.

    Returns:
    list: A list of pixel arrays.
    """
    t0 = time.time()
    imgs = []
    for key in tqdm(data, desc="Generating pixel arrays"):
        try:
            imgs.append(data[key][0].pixel_array)
        except Exception as e:
            print(f"Error reading file {data[key][1]}: {e}")

    if TIMING:
        logger.info(
            f"Time to generate pixel arrays: {datetime.timedelta(seconds=time.time() - t0)}"
        )

    return imgs


def fivebhist_runner(data_root, final_file_name):
    """
    Run the 5-bin histogram feature extraction and bad image identification.

    Parameters:
    data_root (str): The root directory of the DICOM files.
    final_file_name (str): The name of the final text file containing paths to good images.
    """
    t0 = time.time()
    if not os.path.isdir(data_root):
        print("Provided data root directory does not exist.")
        return

    bad_indexes_found = []
    paths = []

    print("Loading DICOM files...")
    file_batches_gen = file_batches_generator(data_root, EXT, BATCH_SIZE)

    total_files = 0
    data_dict = {}
    for file_batch, file_count in file_batches_gen:
        total_files += file_count
        data_dict = load_data_batch(file_batch)
        data_imgs = get_pixel_list(data_dict)
        five_b_hist = Features.get_features(
            data_imgs, feature_type=FEAT, norm_type=NORM, bins=5
        )

        print("Finding bad images...")
        for i, binary in enumerate(five_b_hist):
            if binary[4] > 15000 or binary[1] < 1000:
                print(i, binary)
                paths.append(data_dict[i][1])
                bad_indexes_found.append(i)

    print(f"Total files: {total_files}")

    date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{date_and_time}_{FEAT}_{NORM}.txt"

    os.makedirs(LOG_DIR, exist_ok=True)

    with open(os.path.join(LOG_DIR, file_name), "w") as f:
        f.write("\n".join(paths))

        index_file_name = f"{date_and_time}_{FEAT}_{NORM}_indexes.txt"
        with open(os.path.join(LOG_DIR, index_file_name), "w") as f:
            f.write("\n".join(map(str, bad_indexes_found)))

        with open(os.path.join(LOG_DIR, final_file_name), "w") as f:
            good_paths = [
                data[1]
                for i, data in enumerate(data_dict.values())
                if i not in bad_indexes_found
            ]
            f.write("\n".join(good_paths))

        print(f"number of bad images found: {len(bad_indexes_found)}")
        print(f"number of good images found: {len(good_paths)}")

    if TIMING:
        logger.info(
            f"Total time to run feature extraction: {datetime.timedelta(seconds=time.time() - t0)}"
        )


def print_properties():
    """
    Print the properties used in the 5-bin histogram feature extraction process.
    """
    print(f"Feature type: {FEAT}")
    print(f"Norm type: {NORM}")
    print(f"Log directory: {LOG_DIR}")
    print(f"File extension: {EXT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Data root: {args.data_root}")
    print(f"Final file: {args.final_file}")


# In the main section of your script, you can call this method at the beginning:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image feature extraction task.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=config["5BHIST"]["data_root"],
        help="Root directory of the data.",
    )
    parser.add_argument(
        "--final_file",
        type=str,
        default=config["5BHIST"]["final_file"],
        help="Name of the final file of good images.",
    )
    args = parser.parse_args()

    print_properties()

    try:
        fivebhist_runner(args.data_root, args.final_file)
    except Exception as e:
        print(f"An error occurred: {e}")
