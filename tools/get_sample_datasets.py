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
    file_id = '1cXX7Pdxfkz5MD6oMjbmNlXDkUIAxPd6m'
    destination_pack = 'COiLTRAiNESampleDatasets.tar.gz'

    print("Downloading on training an two validations datasets (7GB total)")
    download_file_from_google_drive(file_id, destination_pack)
    destination_final = os.path.join("~/", os.environ["COIL_DATASET_PATH"])
    if not os.path.exists(destination_final):
        os.makedirs(destination_final)

    print("Unpacking the dataset")

    tf = tarfile.open("COiLTRAiNESampleDatasets.tar.gz")
    tf.extractall(destination_final)

    os.remove("COiLTRAiNESampleDatasets.tar.gz")
