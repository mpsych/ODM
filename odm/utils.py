import os


def print_properties(tite, **kwargs) -> None:
    """
    Print the properties of the VAE Runner.
    """
    print("-" * 80)
    print(f"Running {tite} with the following properties:")
    print("-" * 80)
    print(f"{'Property':<25} {'Value':<10}")
    print("-" * 80)
    for key, value in kwargs.items():
        print(f"{key:<25} {str(value):<10}")
    print("-" * 80)


def validate_inputs(**kwargs) -> None:
    """
    Validate the inputs.
    """
    # check if the caselist is in kwargs and if so if it is a valid file
    caselist = kwargs.get("caselist", None)
    if caselist is not None:
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
        parent_dir = os.path.dirname(good_output)
        if not parent_dir or not os.path.isdir(parent_dir):
            raise NotADirectoryError(f"Directory {parent_dir} does not exist.")

    # check if the bad output has valid parent directories
    bad_output = kwargs.get("bad_output", None)
    if bad_output is not None:
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
                    f"Directory {parent_dir} does not exist and could not be created."
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

    # check if the n_proc is a valid positive integer and not above the number of cores
    n_proc = kwargs.get("max_workers", None)
    if n_proc is not None:
        if not isinstance(n_proc, int) or n_proc <= 0:
            raise ValueError("Number of workers must be a positive integer.")
        import multiprocessing

        if n_proc > multiprocessing.cpu_count():
            raise ValueError("Number of workers must not exceed the number of cores.")

    # check if time is a valid boolean
    time = kwargs.get("time", None)
    if time is not None:
        if not isinstance(time, bool):
            raise ValueError("Time must be a boolean.")

    print("All inputs are valid.")
