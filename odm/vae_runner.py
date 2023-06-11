from odm import OutlierDetector
from .fivebhist_runner import *


def load_data_from_text_file(file_path):
    """
    Loads DICOM data from a text file containing a list of file paths.

    Parameters:
        file_path (str): Path to the text file containing the paths of the DICOM files.

    Returns:
        list: List of DICOM data.
    """
    with open(file_path, "r") as f:
        paths = f.readlines()

    paths = [path.strip() for path in paths]  # remove the new line characters

    data = [dicom.read_file(path) for path in paths]  # load the DICOM files

    return data


def vae_runner(caselist, contamination=0.015, verbose=False):
    """
    Runs the Variational AutoEncoder (VAE) outlier detection algorithm on DICOM data.

    Parameters:
        caselist (str): Path to the text file containing the paths of the DICOM files.
        contamination (float, optional): The proportion of outliers in the data. Defaults to 0.015.
        verbose (bool, optional): Whether to print progress messages to stdout. Defaults to False.

    Returns:
        None
    """
    FEAT = "hist"
    NORM = "minmax"
    # load the data after running 5bhist algorithm
    data = [
        dicom.read_file(path.strip())
        for path in tqdm(open(caselist, "r"), desc="Loading data")
    ]
    imgs = get_pixel_list(data)

    # creating features from the images
    feats = Features.get_features(imgs, feature_type=FEAT, norm_type=NORM)

    # run the outlier detection algorithm
    OutlierDetector.detect_outliers(
        features=feats,
        imgs=imgs,
        pyod_algorithm="VAE",
        contamination=contamination,
        verbose=verbose,
        caselist=caselist,
    )
    return


if __name__ == "__main__":
    """
    Main entry point of the program. Parses command-line arguments, reads the config file,
    overwrites config values if command line arguments are provided, and then runs the VAE algorithm.

    Supports the following command-line arguments:
        caselist (str): Path to the text file containing the paths of the DICOM files.
        contamination (float, optional): The proportion of outliers in the data. Defaults to 0.015.
        verbose (bool, optional): Whether to print progress messages to stdout. Defaults to False.
    """
    # Create a config parser
    config = configparser.ConfigParser()
    # Read the configuration file
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
    args = parser.parse_args()

    # Overwrite config.ini values if command line arguments are provided
    if args.caselist:
        config["VAE"]["caselist"] = args.caselist
    if args.contamination is not None:
        config["VAE"]["contamination"] = str(args.contamination)
    if args.verbose:
        config["VAE"]["verbose"] = str(args.verbose)

    vae_runner(
        config["VAE"]["caselist"],
        float(config["VAE"]["contamination"]),
        config["VAE"]["verbose"],
    )
