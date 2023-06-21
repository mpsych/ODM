# Outlier Detection for Mammograms (ODM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ODM is an open-source project that utilizes a two-stage hybrid algorithm, where the second stage employs unsupervised learning for outlier detection. Primarily designed for handling large mammogram collections, ODM's primary aim is to streamline mammogram datasets by filtering out low-quality and undesired scans. This facilitates more efficient and accurate downstream analysis. The first stage of the pipeline employs a threshold-based 5-bin histogram filtering (5BHIST) method, designed to exclude poor-quality images. The second stage leverages a Variational Autoencoder (VAE), an unsupervised machine learning model, to further identify and eliminate outlier scans. By combining conventional filtering techniques with advanced machine learning, ODM ensures robust and comprehensive outlier detection in mammogram datasets.

The algorithms and strategies employed by ODM are based on our published research paper. In the additional documentation, we've included our replies to reviewers, a more detailed explanation of our methods, an analysis of our chosen parameters, and a thorough list of tested algorithms.

## Table of Contents

- [Outlier Detection for Mammograms (ODM)](#outlier-detection-for-mammograms-odm)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Standard Installation](#standard-installation)
    - [Docker Installation](#docker-installation)
  - [Usage](#usage)
    - [Full Outlier Detection Pipeline](#full-outlier-detection-pipeline)
    - [Separate Pipeline Stages](#separate-pipeline-stages)
    - [1. Five-Bin Histogram Based Thresholding (5BHIST)](#1-five-bin-histogram-based-thresholding-5bhist)
    - [2. Variational Autoencoder (VAE)](#2-variational-autoencoder-vae)
    - [Configuration](#configuration)
    - [Hyperparameters](#hyperparameters)
    - [Additional Documentation](#additional-documentation)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
  - [Contact](#contact)

## Installation

### Standard Installation

Ensure you have Python 3.8 or newer installed. You can check your Python version by running `python --version` in your command line.

Clone the repository:

```bash
git clone https://github.com/mpsych/ODM.git
cd ODM
```

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Or create a new conda environment and install the required packages:

```bash
conda env create -f environment.yml
conda activate ODM
```

### Docker Installation

Ensure you have Docker installed. You can check your Docker version by running `docker --version` in your command line. We have also provided a Dockerfile which can be used to build a Docker image and run the software in a container. To do this, first, make sure you have Docker installed on your system. Then, follow the steps below:

1. Clone the repository:

```bash
git clone https://github.com/mpsych/ODM.git
cd ODM
```

2. Build the Docker image:

```bash
docker build -t odm .
```

3. Run the Docker container:

```bash
docker run -it odm
```

Note: if your project requires access to local files or directories (e.g., for data input or output), you can mount a local directory inside the Docker container using the `-v` flag:

```bash
docker run -it -v /path/to/local/directory:/path/to/container/directory odm
```

Replace `/path/to/local/directory` with the path to the directory on your host system that you want to access from within the Docker container, and replace `/path/in/container` with the path where you want the directory to be mounted in the Docker container.

In the command above, the `-it` flag starts the container in interactive mode, so you can interact with the command line of the running container, and the -v` flag mounts a local directory inside the container.

## Usage

### Full Outlier Detection Pipeline

Please ensure you're situated in the `ODM/odm` directory before proceeding with the execution of the commands discussed below.

The comprehensive outlier detection pipeline can be initiated by invoking the `run_pipeline.py` script. This script accepts the root directory that houses the mammograms. The mammograms may reside directly within this directory, or within nested subdirectories, in structures such as subject folders or group folders..

The script navigates recursively through all the subdirectories, identifying and processing the mammograms. It refers to the parameters from the `config.ini` file by default. However, these parameters can be modified directly via command-line arguments for a more customizable execution.

To run the full outlier detection pipeline using config.ini settings run the following:

```bash
python run_pipeline.py
```

To override the config.ini settings, run the script with one or more of the following command-line arguments:

```bash
python run_pipeline.py --data_root PATH_TO_YOUR_DATA_ROOT --log_dir LOG_DIRECTORY --final_file PATH_TO_YOUR_FINAL_FILE --batch_size BATCH_SIZE --ext EXT --max_workers MAX_WORKERS --caselist PATH_TO_YOUR_FINAL_FILE --contamination CONTAMINATION --good_output GOOD_OUTPUT --bad_output BAD_OUTPUT --timing  --verbose
```

In this command:

- `--data_root` is the path to the root directory of the data. (Must be a valid path containing the mammograms.)
- `--log_dir` is the path to the directory where the log files will be created. (If the directory does not exist, it will be created.)
- `--final_file` is the path to the final file where the paths of the determined good images will be created. (Should only be a file name. Will be saved in the log_dir` directory.)
- `--batch_size` is the batch size for the VAE. (Must be a positive integer. The larger the batch size, the faster the VAE will run, but the more memory it will equire. If you run out of memory, try reducing the batch size.)
- `--ext` is the extension of the mammogram files. (Leave blank if the mammograms do not have an extension. Otherwise, include the period in the extension, e.g., .png`.)
- `--max_workers` is the number of workers for the VAE. (Must be a positive integer. The larger the number of workers, the faster the VAE will run, but the more memory it will require. If you run out of memory, try reducing the number of workers.)
- `--caselist` is the path/filename to the file containing the list of cases to be processed. (If only a filename is specified, the file will be saved in the log_dir` directory, otherwise the full path must be specified.)
- `--contamination` is the contamination rate for the VAE. (Must be a float between 0 and 1. The higher the contamination rate, the more images will be classified as outliers.)
- `--good_output` is the path/filename to the file where the paths of the determined good images will be created. (If only a filename is specified, the file will be saved in the log_dir` directory, otherwise the full path must be specified.)
- `--bad_output` is the path/filename to the file where the paths of the determined bad images will be created. (If only a filename is specified, the file will be saved in the log_dir` directory, otherwise the full path must be specified.)
- `--timing` is a flag that indicates whether to time the execution of the pipeline. (If specified, the execution time will be printed to the console.)
- `--verbose` is a flag that indicates whether to print verbose output to the console. (If specified, the script will print verbose output to the console.)

### Separate Pipeline Stages

You can also run the two stages of the pipeline separately using the `fivebhist_runner.py`
and `vae_runner.py` scripts:

### 1. Five-Bin Histogram Based Thresholding (5BHIST)

The `fivebhist_runner.py` script runs the first stage of the pipeline. It accepts the
root directory of the data and the name of the final file where the paths of the determined
good images will be created. These parameters will be loaded by default from the `[5BHIST]`
section of the `config.ini` file, but they can be overridden via command-line arguments.

To run the 5BHIST stage using config.ini settings run the following:

```shell
python fivebhist_runner.py
```

To override the config.ini settings, run the script with one or more of the following command-line arguments:

```shell
python fivebhist_runner.py --data_root PATH_TO_YOUR_DATA_ROOT --log_file LOG_FILE --final_file PATH_TO_YOUR_FINAL_FILE --batch_size BATCH_SIZE --ext EXT --max_workers MAX_WORKERS --timing
```

See the description of the `run_pipeline.py` script above for a description of these parameters.

### 2. Variational Autoencoder (VAE)

The `vae_runner.py` script runs the second stage of the pipeline. It requires the path to the data and optionally accepts the proportion of outliers in the data(contamination) and a verbosity flag. These parameters will be loaded by default from the `[VAE]` section of the `config.ini` file, but they can be overridden via command-line arguments.

To run the VAE stage using config.ini settings run the following:

```shell
python vae_runner.py
```

To override the config.ini settings, run the script with one or more of the following command-line arguments:

```shell
python vae_runner.py --caselist PATH_TO_YOUR_FINAL_FILE --contamination CONTAMINATION --batch_size BATCH_SIZE --good_output GOOD_OUTPUT --bad_output BAD_OUTPUT --timing  --verbose
```

See the description of the `run_pipeline.py` script above for a description of these parameters.

For additional help/usage instructions for any of the scripts, run `python SCRIPT_NAME.py -h`.

### Configuration

The `config.ini` file contains the default parameters for the pipeline and should look like this:

```ini
[DEFAULT]
log_dir = /tmp/odm/logs/

[5BHIST]
batch_size = 500
timing = false
ext =
data_root = /hpcstor6/scratch01/r/ryan.zurrin001/test_data/raid/data01/deephealth
final_file = good_paths_from_step_1a.txt
max_workers = 4

[VAE]
caselist = good_paths_from_step_1a.txt
contamination = 0.15
batch_size = 500
good_output = good_paths3.txt
bad_output = bad_paths3.txt
timing = false
verbose = false

; Set the following parameters to the desired values to override the defaults, else leave them empty
[HYPERPARAMS]
latent_dim =
hidden_activation =
output_activation =
loss =
optimizer =
epochs =
batch_size =
dropout_rate =
l2_regularizer =
validation_size =
preprocessing =
verbose =
contamination = 0.15
gamma =
capacity =
random_state =
encoder_neurons =
decoder_neurons =
```

You can change these parameters by editing the file or by overriding the [5BHIST] and [VAE] sections via command-line arguments. The parameters are divided into meaningful sections, each corresponding to a stage of the pipeline. The parameters for each stage were described above. The parameters in the [DEFAULT] section are used by both stages of the pipeline.

### Hyperparameters

The [HYPERPARAMS] section provides configuration options for the training process of the Variational Autoencoder (VAE). The only way to set these parameters is in the config.ini file and cannot be overridden via command-line. Here's a brief description of each parameter:

- `latent_dim`: The dimension of the latent space in the VAE.
- `hidden_activation`: The activation function used in the hidden layers of the VAE.
- `output_activation`: The activation function used in the output layer of the VAE.
- `loss`: The loss function used for the optimization process.
- `optimizer`: The optimization algorithm used for training.
- `epochs`: The number of complete passes through the training dataset.
- `batch_size`: The number of training examples utilized in one iteration.
- `dropout_rate`: The probability at which output of each neuron is set to zero; can be used to improve generalization.
- `l2_regularizer`: The coefficient for L2 regularization; can be used to avoid overfitting.
- `validation_size`: The proportion of the dataset to be used as validation data.
- `preprocessing`: The preprocessing steps to apply to the data before training.
- `verbose`: If true, the training process will output detailed information.
- `contamination`: The proportion of outliers in the data.
- `gamma`: The coefficient for the capacity constraint term in the loss function of the VAE.
- `capacity`: The capacity parameter in the loss function of the VAE, which controls the complexity of the learned latent distribution.
- `random_state`: The seed of the pseudo-random number generator to use when shuffling the data.
- `encoder_neurons`: The number of neurons in the encoder part of the VAE.
- `decoder_neurons`: The number of neurons in the decoder part of the VAE.

Please note that not all of these parameters need to be specified; only specify those parameters that are relevant to your specific use case. The values for these parameters will be dependent on the specific dataset you're using, the architecture of your model, and the specific problem you're trying to solve.

### Additional Documentation

For more detailed information, please see the following additional documentation:

- [Replies to Reviewers](docs/additional_info/review_replies.md): Our responses to the insightful questions and suggestions from the reviewers of our research paper. The information here can provide some clarity regarding specific decisions and improvements in our approach.
- [Methods](docs/additional_info/methods.md): Detailed explanations of the methods employed in this project, from pre-processing of the mammogram scans to outlier detection.
- [Parameter analysis](docs/additional_info/parameter_analysis.md): An analysis of the different parameters we have used in the project, offering insights into our choices and their impact on the results.
- [Algorithms Tested](docs/additional_info/algorithms_tested.md): A list and brief description of the different algorithms we tested during our research.
- [limitations](docs/additional_info/limitations.md): A discussion on the potential limitations of our current approach and possible directions for future work.
- [References](docs/additional_info/references.md): A collection of all the references that were instrumental in guiding our research.

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{
zurrin2023outlier,
title={Outlier Detection for Mammograms},
author={Ryan Zurrin and Neha Goyal and Pablo Bendiksen and Muskaan Manocha and Dan Simovici and Nurit Haspel and Marc Pomplun and Daniel Haehn},
booktitle={Medical Imaging with Deep Learning, short paper track},
year={2023},
url={https://openreview.net/forum?id=4E93Xdg98u}
}
```

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for more details.

## Acknowledgements

This research project is a result of collaborative efforts and intellectual contributions that extend beyond our team. We would like to express our heartfelt gratitude to the University of Massachusetts Boston for fostering an environment that encourages academic exploration.

We are particularly thankful for the financial support provided by the Massachusetts Life Sciences Center, which greatly facilitated the realization of this project. Their commitment to nurturing scientific research played a pivotal role in our endeavor.

We also wish to extend our appreciation to DeepHealth for their invaluable technical support. Their expertise significantly enriched the quality of our work and the practicality of our solutions.

Finally, we acknowledge all the fellow researchers, students, and mentors who provided insightful comments, constructive criticisms, and encouragement during the course of this work. Your inputs have played a vital role in shaping our research.

Thank you for supporting us in pushing the boundaries of machine learning in medical imaging.

## Contact

For any questions or comments, feel free to reach out to:

- Ryan Zurrin at [ryan.zurrin001@umb.edu](mailto:ryan.zurrin001@umb.edu)
- Daniel Haehn at [daniel.haehn@umb.edu](mailto:daniel.haehn@umb.edu)
