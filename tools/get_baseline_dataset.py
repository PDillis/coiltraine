import os

import tarfile

from download_tools import download_file_from_google_drive


if __name__ == "__main__":
    try:
        path = os.environ["COIL_DATASET_PATH"]
    except KeyError as e:
        print("")
        print("COIL_DATASET_PATH env variable must be defined.")
        print("")
        raise e

    # Download the datasets
    file_id = '1MAWC14skEafud2UQK0tncM6yTGa_s4Km'
    destination_pack = 'coil_baseline_data_l0.tar.gz'

    print("Downloading on training   datasets (12GB total)")
    download_file_from_google_drive(file_id, destination_pack)
    destination_final = os.path.join("~/", os.environ["COIL_DATASET_PATH"])
    if not os.path.exists(destination_final):
        os.makedirs(destination_final)

    print("Unpacking the dataset")

    tf = tarfile.open("coil_baseline_data_l0.tar.gz")
    tf.extractall(destination_final)

    os.remove("coil_baseline_data_l0.tar.gz")
