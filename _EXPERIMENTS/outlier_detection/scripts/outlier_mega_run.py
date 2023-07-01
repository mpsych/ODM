import sys

sys.path.insert(0, "/home/ryan.zurrin001/Projects/omama/")

from ..outlier_detector import *

# each one of the config numbers is a different partition of images
# containing 44123 images randomly selected from the 176492 images.
config_numbers = [29, 30, 31, 32]
# sets the feature type and normalization type
feature = "hist"
norm = "minmax"


# plan to run 4 different jobs. each has 44123 images and each image is at
# maximum 14 MB in size giving us a total of about 618 GB for each job.
def per_task_execution(idx):
    # arguments is for the pyod algorithm to fine tune it's parameters
    # arguments = {
    #     'contamination': 0.05,
    #     'n_neighbors': 30,
    #     'radius': 1.0,
    #     'leaf_size': 50,
    #     'metric': 'euclidean',
    #     'p': 3,
    #     'n_jobs': 1,
    #     'verbose': False,
    #     'norm_type': 'minmax'
    # }
    # detector_list = [
    #     'HBOS',
    #     'KNN',
    #     'DeepSVDD',
    # ]

    arguments = {
        "contamination": 0.015,
        "verbose": False,
        "norm_type": norm,
        # 'metric': 'euclidean',
        # 'n_jobs': None,
        # 'n_bins': 2,
        # 'alpha': 0.2,
        # 'n_clusters': 2,
        # "tol": 0.2,
        # "beta": 3,
        # "n_neighbors": 10,
        # "base_estimators": detector_list,
        # "cluster_estimator": 'KMeans',
        # "n_estimators": 6,
        # "subset_size": 0.5,
        # "combination": "maximum",
        # 'leaf_size': 50,
        # "p": 3,
        # "novelty": False,
        # "kernel": "rbf",
        # "degree": 2,
        # "gamma": 0.1,
        # "coef0": 1,
        # "nu": 0.5,
        # "shrinking": True,
        # "cache_size": 50,
        # "max_iter": -1,
        # "max_samples": 0.5,
        # "bandwidth": 1.0,
        # "radius": 1.0,
        # "hidden_neurons": [24, 12, 12, 24]
    }

    # set the config number by the task index
    config = config_numbers[idx % 4]
    # config = config_numbers[0]

    # pulling in the data from where it is stored on the cluster
    imgs = DataHelper.get2D(N=42092, config_num=config, randomize=True)

    # creating features from the images
    feat = Features.get_features(imgs, feature_type=feature, norm_type=norm)

    # run the outlier detection algorithm
    OutlierDetector.detect_outliers(
        features=feat, imgs=imgs, pyod_algorithm="VAE", id_=idx, **arguments
    )

    print(f"task number is: {idx}")


if __name__ == "__main__":
    TASK_NUM = sys.argv[1]
    TASK_NUM = int(TASK_NUM)
    per_task_execution(TASK_NUM)
