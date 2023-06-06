import os
import datetime
import pydicom as dicom
import argparse
from feature_extractor import *

# histogram and minmax has been found to be the best combination for 5BHIST algorithm
FEAT = 'hist'
NORM = 'minmax'

LOG_DIR = r'/tmp/odm/logs/'


def load_data_dict(path, ext='.dcm'):
    data_dict = {}
    for index, (root, dirs, files) in enumerate(os.walk(path)):
        dicom_files = [file for file in files if file.endswith(ext)]
        for file in dicom_files:
            try:
                data_dict[index] = [dicom.dcmread(os.path.join(root, file)),
                                    os.path.join(root, file)]
            except Exception as e:
                print(f"Error reading file {file}: {e}")
    return data_dict


def get_pixel_list(data_dict):
    return [data[0].pixel_array for data in data_dict.values() if data[0].pixel_array.size > 0]


def fivebhist_runner(data_root, final_file_name):
    if not os.path.isdir(data_root):
        print("Provided data root directory does not exist.")
        return

    # pulling in the data from where it is stored
    data_dict = load_data_dict(data_root)
    data_imgs = get_pixel_list(data_dict)

    # creating features from the images
    five_b_hist = Features.get_features(data_imgs, feature_type=FEAT, norm_type=NORM, bins=5)

    # make a list of all the image paths with images that have higher than 1 in second bin
    bad_indexes_found = []
    paths = []
    for i, binary in enumerate(five_b_hist):
        if binary[4] > 15000 or binary[1] < 1000:
            print(i, binary)
            paths.append(data_dict[i][1])
            bad_indexes_found.append(i)

    # create a file name using the datetime + feat + norm
    date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"{date_and_time}_{FEAT}_{NORM}.txt"

    # if the log directory does not exist, create it
    os.makedirs(LOG_DIR, exist_ok=True)

    # write the paths to the file in the log directory
    with open(os.path.join(LOG_DIR, file_name), 'w') as f:
        f.write("\n".join(paths))

    # write the indexes to the file in the log directory as well
    index_file_name = f"{date_and_time}_{FEAT}_{NORM}_indexes.txt"
    with open(os.path.join(LOG_DIR, index_file_name), 'w') as f:
        f.write("\n".join(map(str, bad_indexes_found)))

    # create a final text file of paths to good images after removing the bad images
    with open(os.path.join(LOG_DIR, final_file_name), 'w') as f:
        good_paths = [data[1] for i, data in enumerate(data_dict.values()) if i not in bad_indexes_found]
        f.write("\n".join(good_paths))

    # print the number of bad images found
    print(f"number of bad images found: {len(bad_indexes_found)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image feature extraction task.")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the data.')
    parser.add_argument('--final_file', type=str, required=True, help='Name of the final file of good images.')
    args = parser.parse_args()

    fivebhist_runner(args.data_root, args.final_file)
