from .__configloc__ import CONFIG_LOC

import configparser
import logging
import os


config = configparser.ConfigParser()
config.read(CONFIG_LOC)


def print_properties(tite, **properties) -> None:
    """Print the properties of the VAE Runner.

    Parameters
    ----------
    tite : str
        The title of the runner.
    **properties
        The properties of the runner.
    """
    max_key_length = max(len(key) for key in properties)
    max_val_length = max(len(str(val)) for val in properties.values())
    max_line_length = max_key_length + max_val_length + 3
    header_length = (
        len("Running ") + len(tite) + len("with the following " "properties:")
    )
    logging_prefix_length = 33

    max_length = max(max_line_length, header_length) + logging_prefix_length

    print("-" * max_length)
    logging.info(f"Running {tite} with the following properties:")
    print("-" * max_length)
    logging.info(f"{'Property':<{max_key_length}} {'Value':<{max_val_length}}")
    print("-" * max_length)
    for key, value in properties.items():
        logging.info(f"{key:<{max_key_length}} {str(value):<{max_val_length}}")
    print("-" * max_length)


def validate_inputs(**kwargs) -> None:
    """Validate the inputs.

    Parameters
    ----------
    **kwargs : dict, Any
        The keyword arguments to validate.
    """
    log_dir = config["DEFAULT"]["log_dir"]
    # check if the caselist is in kwargs and if so if it is a valid file
    caselist = kwargs.get("caselist", None)
    if caselist is not None:
        # join caselist with log_dir if it is not a full path
        if not os.path.isfile(caselist):
            caselist = os.path.join(log_dir, caselist)
        if not os.path.isfile(caselist):
            raise FileNotFoundError(f"File {caselist} does not exist.")

    # check if the contamination is in kwargs and if so if it is a valid float
    contamination = kwargs.get("contamination", None)
    if contamination is not None:
        if not 0 <= contamination <= 1:
            raise ValueError("Contamination must be a float between 0 and 1.")

    # check if the batch size is a valid positive integer
    batch_size = kwargs.get("batch_size", None)
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")

    # check if the good output has valid parent directories
    good_output = kwargs.get("good_output", None)
    if good_output is not None:
        # join good_output with log_dir if it is not a full path
        if not os.path.isfile(good_output):
            good_output = os.path.join(log_dir, good_output)
        parent_dir = os.path.dirname(good_output)
        if not parent_dir or not os.path.isdir(parent_dir):
            raise NotADirectoryError(f"Directory {parent_dir} does not exist.")

    # check if the bad output has valid parent directories
    bad_output = kwargs.get("bad_output", None)
    if bad_output is not None:
        # join bad_output with log_dir if it is not a full path
        if not os.path.isfile(bad_output):
            bad_output = os.path.join(log_dir, bad_output)
        parent_dir = os.path.dirname(bad_output)
        if not parent_dir or not os.path.isdir(parent_dir):
            raise NotADirectoryError(f"Directory {parent_dir} does not exist.")

    # check if the log directory has valid parent directories if not make it
    log_dir = kwargs.get("log_dir", None)
    if log_dir is not None:
        parent_dir = os.path.dirname(log_dir)
        if not parent_dir or not os.path.isdir(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except Exception as e:
                raise NotADirectoryError(
                    f"Directory {parent_dir} does not exist and could not be "
                    f"created."
                )

    # check if the data root has valid parent directories
    data_root = kwargs.get("data_root", None)
    if data_root is not None:
        parent_dir = os.path.dirname(data_root)
        if not parent_dir or not os.path.isdir(parent_dir):
            raise NotADirectoryError(f"Directory {parent_dir} does not exist.")

    # check if the final file is a valid string name
    final_file = kwargs.get("final_file", None)
    if final_file is not None:
        if not isinstance(final_file, str):
            raise ValueError("Final file must be a string.")

    # check if the ext is a valid string or empty
    ext = kwargs.get("ext", None)
    if ext is not None:
        if not isinstance(ext, str):
            raise ValueError("File extension must be a string.")

    # check if the n_proc is a valid positive integer and not above the
    # number of cores
    n_proc = kwargs.get("max_workers", None)
    if n_proc is not None:
        if not isinstance(n_proc, int) or n_proc <= 0:
            raise ValueError("Number of workers must be a positive integer.")
        import multiprocessing

        if n_proc > multiprocessing.cpu_count():
            raise ValueError(
                "Number of workers must not exceed the number of " "cores."
            )

    # check if time is a valid boolean
    time = kwargs.get("time", None)
    if time is not None:
        if not isinstance(time, bool):
            raise ValueError("Time must be a boolean.")

    logging.info("All inputs are valid.")
