# Outlier Detection for Mammograms (ODM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Table of Contents
1. [Installation](#installation)
    - [Standard Installation](#standard-installation)
    - [Docker Installation](#docker-installation)
2. [Usage](#usage)
    - [Full Outlier Detection Pipeline](#full-outlier-detection-pipeline)
    - [Separate Pipeline Stages](#separate-pipeline-stages)
3. [Configuration](#configuration)
    - [Five-Bin Histogram Based Thresholding (5BHIST)](#five-bin-histogram-based-thresholding-5bhist)
    - [Variational Autoencoder (VAE)](#variational-autoencoder-vae)
4. [Citation](#citation)
5. [License](#license)
6. [Acknowledgements](#acknowledgements)
7. [Contact](#contact)

ODM is a two-stage unsupervised learning-based outlier detection pipeline for large 
mammogram collections. It uses a threshold-based 5-bin histogram filtering (5-BHIST) 
and a variational autoencoder (VAE) to remove low-quality and undesired scans.

## Installation

### Standard Installation

Ensure you have Python 3.8 or newer installed. You can check your Python version by 
running `python --version` in your command line.

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

Ensure you have Docker installed. You can check your Docker version by running 
`docker --version` in your command line.

We have also provided a Dockerfile which can be used to build a Docker image and run 
the software in a container. To do this, first, make sure you have Docker installed on 
your system. Then, follow the steps below:

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

Note: if your project requires access to local files or directories (e.g., for data input or output), 
you can mount a local directory inside the Docker container using the `-v` flag:
```bash
docker run -it -v /path/to/local/directory:/path/to/container/directory odm
```
Replace `/path/to/local/directory` with the path to the directory on your host system 
that you want to access from within the Docker container, and replace `/path/in/container` 
with the path where you want the directory to be mounted in the Docker container.


In the command above, the `-it` flag starts the container in interactive mode, 
so you can interact with the command line of the running container, and the `-v` 
flag mounts a local directory inside the container.


## Usage

### Full Outlier Detection Pipeline
Be sure to navigate to the `ODM/odm` directory before running any of the commands below.

You can run the full outlier detection pipeline using the `run_pipeline.py` script. 
This script accepts a text file with paths to the mammograms and outputs a final caselist 
of paths to the filtered, high-quality images. The paths and other parameters will be 
loaded by default from the `config.ini` file, but they can be overridden via command-line arguments.
```bash
python run_pipeline.py --data_root PATH_TO_YOUR_DATA_ROOT --log_file LOG_FILE --final_file PATH_TO_YOUR_FINAL_FILE --batch_size BATCH_SIZE --ext EXT --max_workers MAX_WORKERS --caselist PATH_TO_YOUR_FINAL_FILE --contamination CONTAMINATION --batch_size BATCH_SIZE --good_output GOOD_OUTPUT --bad_output BAD_OUTPUT --timing  --verbose
```

### Separate Pipeline Stages
You can also run the two stages of the pipeline separately using the `fivebhist_runner.py` 
and `vae_runner.py` scripts:

### 1. Five-Bin Histogram Based Thresholding (5BHIST)

The `fivebhist_runner.py` script runs the first stage of the pipeline. It accepts the 
root directory of the data and the name of the final file where the paths of the determined 
good images will be created. These parameters will be loaded by default from the `[5BHIST]` 
section of the `config.ini` file, but they can be overridden via command-line arguments.
```shell
python fivebhist_runner.py --data_root PATH_TO_YOUR_DATA_ROOT --log_file LOG_FILE --final_file PATH_TO_YOUR_FINAL_FILE --batch_size BATCH_SIZE --ext EXT --max_workers MAX_WORKERS --timing
```

### 2. Variational Autoencoder (VAE)
The `vae_runner.py` script runs the second stage of the pipeline. It requires the path 
to the data and optionally accepts the proportion of outliers in the data (contamination) 
and a verbosity flag. These parameters will be loaded by default from the `[VAE]` section 
of the `config.ini` file, but they can be overridden via command-line arguments.
```shell
python vae_runner.py --caselist PATH_TO_YOUR_FINAL_FILE --contamination CONTAMINATION --batch_size BATCH_SIZE --good_output GOOD_OUTPUT --bad_output BAD_OUTPUT --timing  --verbose
```
In these commands:
- `<data_path>` is the path to your data.
- `<log_path>` is the path to your log file.
- `<final_path>` is the path to your final file for the high-quality images from stage 1.
- `<batch_size>` is the batch size you want to use for processing large datasets.
- `<ext>` is the file extension of your images.
- `<worker_count>` is the maximum number of workers to use for parallel processing.
- `<caselist_path>` is your final file path for the processed images.
- `<contamination_rate>` is the proportion of outliers in your data (between 0 and 1).
- `<good_output_path>` and `<bad_output_path>` are the paths to the output files for the high-quality and low-quality images, respectively.
- `--timing` will time the process.
- `--verbose` will enable or disable console progress outputs.
For additional help for any of the scripts, run `python SCRIPT_NAME.py -h`.

## Configuration
The `config.ini` file contains the default parameters for the pipeline. You can change 
these parameters by editing the file or by overriding them via command-line arguments. 
The parameters are divided into sections, each corresponding to a stage of the pipeline. 
The parameters for each stage are described below.

### 1. Five-Bin Histogram Based Thresholding (5BHIST)
The parameters for the 5BHIST stage are in the `[5BHIST]` section of the `config.ini` file. They are described below:
- `data_root`: The root directory of the data. This is the directory that contains the `mammograms` and `tomosynthesis` directories.
- `log_dir`: The directory where the log files will be saved. If you do not want to save the log files, set this to `None`.- 
- `final_file`: The path to the final file of good images. This is the file that will be created by the 5BHIST stage and used as input for the VAE stage.
- `ext`: The file extension of the images. This should be `.png` for PNG images or `.jpg` for JPEG images. 
- `batch_size`: The batch size to use for loading the images. This is the number of images that will be loaded into memory at a time. If you have a large amount of memory, you can increase this number to speed up the process. If you have a small amount of memory, you may need to decrease this number to avoid running out of memory.
- `max_workers`: The number of workers to use for loading the images. This is the number of parallel processes that will be used to load the images. If you have a large number of CPU cores, you can increase this number to speed up the process. If you have a small number of CPU cores, you may need to decrease this number to avoid running out of memory.
- `timing`: Whether to time the 5BHIST stage. If set to `True`, the time taken to run the 5BHIST stage will be printed to stdout.

### 2. Variational Autoencoder (VAE)
- `caselist`: The path to the final file of good images. This is the file that will be created by the 5BHIST stage and used as input for the VAE stage.
- `contamination`: The proportion of outliers in the data. This is a number between 0 and 1. If you do not know the proportion of outliers in your data, you can set this to 0.15.
- `batch_size`: The batch size to use for loading the images. This is the number of images that will be loaded into memory at a time. If you have a large amount of memory, you can increase this number to speed up the process. If you have a small amount of memory, you may need to decrease this number to avoid running out of memory.
- `good_output`: The path to the final output file. This is the file that will be created by the VAE stage and is the final output of the pipeline (All the good images).
- `bad_output`: The path to the final output file. This is the file that will be created by the VAE stage and is the final output of the pipeline (All the bad images).
- `timing`: Whether to time the VAE stage. If set to `True`, the time taken to run the VAE stage will be printed to stdout.
- `verbose`: Whether to print progress messages to stdout. If set to `True`, progress messages will be printed to stdout. If set to `False`, the script will run silently.


## Citation
If you use this code in your research, please cite the following paper:
```
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