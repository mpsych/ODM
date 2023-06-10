# Outlier Detection for Mammograms (ODM)

ODM is a two-stage unsupervised learning-based outlier detection pipeline for large mammogram collections. It uses a threshold-based 5-bin histogram filtering (5-BHIST) and a variational autoencoder (VAE) to remove low-quality and undesired scans.

## Installation

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

## Usage

### Full Outlier Detection Pipeline
Be sure to navigate to the `ODM/odm` directory before running any of the commands below.

You can run the full outlier detection pipeline using the `odm_runner.py` script. This script accepts a text file with paths to the mammograms and outputs a final caselist of paths to the filtered, high-quality images. The paths and other parameters will be loaded by default from the `config.ini` file, but they can be overridden via command-line arguments.
```bash
python odm_runner.py --data_root PATH_TO_YOUR_DATA_ROOT --final_file PATH_TO_YOUR_FINAL_FILE --contamination CONTAMINATION --verbose VERBOSE
```

### Separate Pipeline Stages
You can also run the two stages of the pipeline separately using the `fivebhist_runner.py` and `vae_runner.py` scripts:

### 1. Five-Bin Histogram Based Thresholding (5BHIST)

The `fivebhist_runner.py` script runs the first stage of the pipeline. It accepts the root directory of the data and the name of the final file where the paths of the determined good images will be created. These parameters will be loaded by default from the `[5BHIST]` section of the `config.ini` file, but they can be overridden via command-line arguments.
```shell
python fivebhist_runner.py --data_root PATH_TO_YOUR_DATA_ROOT --final_file PATH_TO_YOUR_FINAL_FILE
```

### 2. Variational Autoencoder (VAE)
The `vae_runner.py` script runs the second stage of the pipeline. It requires the path to the data and optionally accepts the proportion of outliers in the data (contamination) and a verbosity flag. These parameters will be loaded by default from the `[VAE]` section of the `config.ini` file, but they can be overridden via command-line arguments.
```shell
python vae_runner.py --caselist PATH_TO_YOUR_FINAL_FILE --contamination CONTAMINATION --verbose VERBOSE
```

In the commands above, replace `PATH_TO_YOUR_DATA_ROOT` with the path to your data, `PATH_TO_YOUR_FINAL_FILE` with the path to your final file of good images, and `CONTAMINATION` with the proportion of outliers in your data (a number between 0 and 1). If you include `--verbose` in the `vae_runner.py` command, progress messages will be printed to stdout. If not, the script will run silently.

For additional help for any of the scripts, run `python SCRIPT_NAME.py -h`.

## Configuration
The `config.ini` file contains the default parameters for the pipeline. You can change these parameters by editing the file or by overriding them via command-line arguments. The parameters are divided into sections, each corresponding to a stage of the pipeline. The parameters for each stage are described below.

### 1. Five-Bin Histogram Based Thresholding (5BHIST)
The parameters for the 5BHIST stage are in the `[5BHIST]` section of the `config.ini` file. They are described below:
- `data_root`: The root directory of the data. This is the directory that contains the `mammograms` and `tomosynthesis` directories.
- `final_file`: The path to the final file of good images. This is the file that will be created by the 5BHIST stage and used as input for the VAE stage.
- `batch_size`: The batch size to use for loading the images. This is the number of images that will be loaded into memory at a time. If you have a large amount of memory, you can increase this number to speed up the process. If you have a small amount of memory, you may need to decrease this number to avoid running out of memory.
- `timing`: Whether to time the 5BHIST stage. If set to `True`, the time taken to run the 5BHIST stage will be printed to stdout.
- `log_dir`: The directory where the log files will be saved. If you do not want to save the log files, set this to `None`.
- `ext`: The file extension of the images. This should be `.png` for PNG images or `.jpg` for JPEG images.
- `data_root`: The root directory of the data. This is the directory that contains the `mammograms` and `tomosynthesis` directories.

### 2. Variational Autoencoder (VAE)
- `caselist`: The path to the final file of good images. This is the file that will be created by the 5BHIST stage and used as input for the VAE stage.
- `contamination`: The proportion of outliers in the data. This is a number between 0 and 1. If you do not know the proportion of outliers in your data, you can set this to 0.15.
- `verbose`: Whether to print progress messages to stdout. If set to `True`, progress messages will be printed to stdout. If set to `False`, the script will run silently.
- `batch_size`: The batch size to use for loading the images. This is the number of images that will be loaded into memory at a time. If you have a large amount of memory, you can increase this number to speed up the process. If you have a small amount of memory, you may need to decrease this number to avoid running out of memory.

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
- Daniel Haehn at [daniel.heahn@umb.edu](mailto:daniel.heahn@umb.edu)