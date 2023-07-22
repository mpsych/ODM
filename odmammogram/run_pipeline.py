import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from odmammogram.core.__configloc__ import CONFIG_LOC
from odmammogram.core.fivebhist_runner import fivebhist_runner
from odmammogram.core.utils import print_properties, validate_inputs
from odmammogram.core.vae_runner import vae_runner
import argparse
import configparser
import logging
import os
import psutil
import sys
import time


def setup_logging(logfile_, level="INFO", verbose_=False):
    """Setup logging to stdout and file.

    Parameters
    ----------
    logfile_ : str
        The path to the log file.
    level : str, optional
        The logging level.
    verbose_ : bool, optional
        Whether to log to stdout.
    """
    loglevel_ = getattr(logging, level.upper(), None)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=loglevel_,
        handlers=[
            logging.FileHandler(logfile_),
            logging.StreamHandler(sys.stdout) if verbose_ else logging.NullHandler(),
        ],
    )
    logging.info("Logging initialized.")


def get_5bhist_args(config_):
    """Get the arguments for the 5BHIST runner.

    Parameters
    ----------
    config_ : configparser.ConfigParser
        The configuration file parser.

    Returns
    -------
    args_ : argparse.Namespace
        The arguments for the 5BHIST runner.
    """
    parser_ = argparse.ArgumentParser(
        description="Stage 1 mammogram Outlier detection task."
    )
    parser_.add_argument(
        "--data_root",
        type=str,
        default=config_["5BHIST"]["data_root"],
        help="Root directory of the data.",
    )
    parser_.add_argument(
        "--log_dir",
        type=str,
        default=config_["DEFAULT"]["log_dir"],
        help="Directory to save the log files.",
    )
    parser_.add_argument(
        "--final_file",
        type=str,
        default=config_["5BHIST"]["final_file"],
        help="Name of the final text file containing paths to good images.",
    )
    parser_.add_argument(
        "--batch_size",
        type=int,
        default=config_["5BHIST"]["batch_size"],
        help="Number of files to load at a time.",
    )
    parser_.add_argument(
        "--ext",
        type=str,
        default=config_["5BHIST"]["ext"],
        help="File extension of the DICOM files.",
    )
    parser_.add_argument(
        "--max_workers",
        type=int,
        default=config_["5BHIST"]["max_workers"],
        help="Number of processes to use.",
    )
    parser_.add_argument(
        "--timing",
        action="store_true",
        default=config_.getboolean("5BHIST", "timing"),
        help="Whether to time the feature extraction process.",
    )
    args_ = parser_.parse_args()
    validate_inputs(**vars(args_))
    print_properties("5BHIST Runner", **vars(args_))
    return args_


def get_vae_args(config_):
    """Gets the arguments for the VAE runner.

    Parameters
    ----------
    config_ : configparser.ConfigParser
        The configuration file parser.

    Returns
    -------
    args_ : argparse.Namespace
        The arguments for the VAE runner.
    """
    parser_ = argparse.ArgumentParser(
        description="Stage 2 mammogram Outlier detection task."
    )
    parser_.add_argument(
        "--log_dir",
        type=str,
        default=config_["DEFAULT"]["log_dir"],
        help="Directory to save the log files.",
    )
    parser_.add_argument(
        "--caselist",
        type=str,
        default=config_["VAE"]["caselist"],
        help="The path to the text file containing the paths of the DICOM files.",
    )
    parser_.add_argument(
        "--batch_size",
        type=int,
        default=config_["VAE"]["batch_size"],
        help="The number of files to process in each batch.",
    )
    parser_.add_argument(
        "--good_output",
        type=str,
        default=config_["VAE"]["good_output"],
        help="The name of the final file where good paths are saved.",
    )
    parser_.add_argument(
        "--bad_output",
        type=str,
        default=config_["VAE"]["bad_output"],
        help="The name of the final file where bad paths are saved.",
    )
    parser_.add_argument(
        "--timing",
        action="store_true",
        default=config_.getboolean("VAE", "timing"),
        help="Whether to time the execution of the algorithm.",
    )
    parser_.add_argument(
        "--verbose",
        action="store_true",
        default=config_.getboolean("VAE", "verbose"),
        help="Whether to print progress messages to stdout.",
    )
    args_ = parser_.parse_args()
    validate_inputs(**vars(args_))
    print_properties("VAE Runner", **vars(args_))
    return args_


def run_stage1(args_):
    """Runs the 5BHIST runner.

    Parameters
    ----------
    args_ : argparse.Namespace
        the arguments for the 5BHIST runner.
    """
    try:
        fivebhist_runner(
            data_root=args_.data_root,
            final_file=args_.final_file,
            log_dir=args_.log_dir,
            ext=args_.ext,
            batch_size=args_.batch_size,
            max_workers=args_.max_workers,
            timing=args_.timing,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        logging.error(f"Error in 5BHIST runner: {e}")
        sys.exit(1)


def run_stage2(args_):
    """Runs the VAE runner.

    Parameters
    ----------
    args_ : argparse.Namespace
        The arguments for the VAE runner.
    """
    try:
        gp, bp = vae_runner(
            log_dir=args_.log_dir,
            caselist=args_.caselist,
            batch_size=args_.batch_size,
            log_to_terminal=args_.verbose,
            timing=args_.timing,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        logging.error(f"Error in VAE runner: {e}")
        sys.exit(1)

    # check if good_output and bad_output are just file names or full paths,
    # if just file names, add log_dir
    if not os.path.dirname(args_.good_output):
        args_.good_output = os.path.join(args_.log_dir, args_.good_output)
    if not os.path.dirname(args_.bad_output):
        args_.bad_output = os.path.join(args_.log_dir, args_.bad_output)

    write_to_file(args_.good_output, gp)
    write_to_file(args_.bad_output, bp)
    logging.info(f"Good paths written to {args_.good_output}")
    logging.info(f"Bad paths written to {args_.bad_output}")
    logging.info(f"number of good paths: {len(gp)}")
    logging.info("****** Outlier detection complete. ******")


def write_to_file(file_path, paths):
    """Writes the paths to a text file.

    Parameters
    ----------
    file_path : str
        The path to the text file.
    paths : list
        The list of paths to write to the file.
    """
    try:
        with open(file_path, "w") as f_:
            for path_ in paths:
                f_.write(f"{path_}\n")
    except Exception as e_:
        import traceback

        traceback.print_exc()
        logging.error(f"Error writing to file {file_path}: {e_}")
        sys.exit(1)


if __name__ == "__main__":
    """The main function for the automated ODM pipeline."""
    t0 = time.time()
    config = configparser.ConfigParser()
    config.read(CONFIG_LOC)

    log_dir = config["DEFAULT"]["log_dir"]
    logfile = config["DEFAULT"]["logfile"]

    # if logfile is only a file name and not a path, prepend log_dir
    if not os.path.dirname(logfile):
        logfile = os.path.join(log_dir, logfile)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    loglevel = config["DEFAULT"]["loglevel"]
    verbose = config.getboolean("DEFAULT", "verbose")
    setup_logging(logfile, loglevel, verbose)

    logging.info("Starting 5BHIST runner.")
    args1 = get_5bhist_args(config)
    run_stage1(args1)
    logging.info("5BHIST runner completed.")

    logging.info("Starting VAE runner.")
    args2 = get_vae_args(config)
    run_stage2(args2)
    logging.info("VAE runner completed.")

    # print total time and memory usage
    t1 = time.time()
    total_time = t1 - t0
    logging.info(f"Total time: {total_time} seconds.")
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024**2
    logging.info(f"Memory usage: {memory} MB.")
