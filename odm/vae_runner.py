from odm import OutlierDetector
from .fivebhist_runner import *


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
    with open(file_path, "r") as f:
        paths = f.readlines()

    paths = [path.strip() for path in paths]  # remove the new line characters

    data = [dicom.read_file(path) for path in paths]  # load the DICOM files

    return data


def vae_runner(caselist, contamination=0.015, verbose=False):
    """
    Runs the Variational AutoEncoder (VAE) algorithm on given data.

    Parameters
    ----------
    caselist : str
        The path to the text file containing the paths of the DICOM files.
    contamination : float, optional
        The proportion of outliers in the data. Default is 0.015.
    verbose : bool, optional
        Whether to print progress messages to stdout. Default is False.
    norm_type : str, optional
        The type of normalization to be applied. Default is NORM (minmax).
    """
    FEAT = "hist"
    NORM = "minmax"
    # load the data after running 5bhist algorithm
    data = [dicom.read_file(path.strip()) for path in tqdm(open(caselist, "r"), desc="Loading data")]
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
    # Create a config parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config.ini')

    # Get the valus from the config file
    caselist = config['VAE']['caselist']
    contamination = float(config['VAE']['contamination'])
    verbose = config['VAE']['verbose']

    parser = argparse.ArgumentParser(
        description="Runs the Variational AutoEncoder (VAE) algorithm on given data."
    )
    parser.add_argument("caselist", type=str, help="The path to the text file containing the paths of the DICOM files.")
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.015,
        help="The proportion of outliers in the data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print progress messages to stdout.",
    )
    args = parser.parse_args()

    vae_runner(args.caselist, args.contamination, args.verbose)
