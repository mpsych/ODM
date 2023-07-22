import os
import sys
import datetime

from ..feature_extractor import *
from ..utils import *

# each one of the config numbers is a different partition of images
# containing 44123 images randomly selected from the 176492 images.
config_numbers = [8, 9, 10, 11]
# sets the feature type and normalization type
feature = "hist"
norm = "minmax"
LOG_DIR = r"/tmp/odmammogram/logs/"


# plan to run 4 different jobs. each has 44123 images and each image is at
# maximum 14 MB in size giving us a total of about 618 GB for each job.
def per_task_execution(idx):
    # set the config number by the task index
    config = config_numbers[idx % 4]

    # pulling in the data from where it is stored on the cluster
    data_imgs = DataHelper.get2D(N=44123, config_num=config, randomize=True)

    # creating features from the images
    binary_bin_feats = Features.get_features(
        data_imgs, feature_type=feature, norm_type=norm, bins=5
    )

    # make a list of all the img pahts with images that have higher then 1 in
    # second bin
    bad_indexes_found = []
    paths = []
    for i, binary in enumerate(binary_bin_feats):
        if binary[4] > 15000 or binary[1] < 1000:
            print(i, binary)
            paths.append(data_imgs[i].filePath)
            bad_indexes_found.append(i)

    # create a file name using the datetime + feat + norm
    date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{date_and_time}_{feature}_{norm}.txt"
    # if the log directory does not exist, create it
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    # write the paths to the file in the log directory
    with open(os.path.join(LOG_DIR, file_name), "w") as f:
        for path in paths:
            f.write(f"{path}\n")
    # write the indexes to the file in the log directory as well
    index_file_name = f"{date_and_time}_{feature}_{norm}_indexes.txt"
    with open(os.path.join(LOG_DIR, index_file_name), "w") as f:
        for index in bad_indexes_found:
            f.write(f"{index}\n")

    # close the file
    f.close()

    # print the number of bad images found
    print(f"number of bad images found: {len(bad_indexes_found)}")

    print(f"task number is: {idx}")


if __name__ == "__main__":
    TASK_NUM = sys.argv[1]
    TASK_NUM = int(TASK_NUM)
    per_task_execution(TASK_NUM)
