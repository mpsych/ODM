from odm import OutlierDetector
from .fivebhist_runner import *


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
            enumerate(files), desc="Loading DICOM files", total=len(files)
    ):
        try:
            data_dict[index] = [dicom.dcmread(file), file]
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

    # Read the list of file paths
    with open(caselist, "r") as f:
        all_files = [path.strip() for path in f.readlines()]

    # Process the files in batches
    for i in range(0, len(all_files), batch_size):
        file_batch = all_files[i: i + batch_size]

        # Load the data batch after running 5bhist algorithm
        data_dict = load_data_batch(file_batch)
        imgs = get_pixel_list(data_dict)

        # Create features from the images
        feats = Features.get_features(imgs, feature_type=FEAT, norm_type=NORM)

        # Run the outlier detection algorithm
        OutlierDetector.detect_outliers(
            features=feats,
            imgs=imgs,
            pyod_algorithm="VAE",
            contamination=contamination,
            verbose=verbose,
            caselist=caselist,
        )


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")

    parser = argparse.ArgumentParser(
        description="Runs the Variational AutoEncoder (VAE) algorithm on given data."
    )
    parser.add_argument(
        "caselist",
        type=str,
        help="The path to the text file containing the paths of the DICOM files.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=None,
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
        default=100,
        help="The number of files to process in each batch.",
    )
    args = parser.parse_args()

    if args.caselist:
        config["VAE"]["caselist"] = args.caselist
    if args.contamination is not None:
        config["VAE"]["contamination"] = str(args.contamination)
    if args.verbose:
        config["VAE"]["verbose"] = str(args.verbose)
    if args.batch_size:
        config["VAE"]["batch_size"] = str(args.batch_size)

    vae_runner(
        config["VAE"]["caselist"],
        float(config["VAE"]["contamination"]),
        config["VAE"]["verbose"],
        int(config["VAE"]["batch_size"]),
    )