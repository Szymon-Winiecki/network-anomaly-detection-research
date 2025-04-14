from pathlib import Path
import sys
import zipfile

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


this_directory = Path(__file__).parent.resolve()

utils_directory = this_directory / Path("../utils/")
sys.path.append(str(utils_directory))
from dataset_utils import save_preprocessed_data

# path do the archive with the raw data
raw_data_zip = this_directory / Path("../../data/UNSW-NB15/raw/CSV Files.zip")

# paths to the files inside the archive
train_dataset_filename = "CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv"
test_dataset_filename = "CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv"
# features_description_filename = "CSV Files/NUSW-NB15_features.csv"

processed_data_root_directory = this_directory / Path("../../data/UNSW-NB15/preprocessed/")

# check if the zip file exists
if not raw_data_zip.exists():
    print(f"Raw data zip file {raw_data_zip} does not exist.")
    exit(1)

# check if the files inside the zip file exist
with zipfile.ZipFile(raw_data_zip, 'r') as z:
    zip_files = z.namelist()
    if train_dataset_filename not in zip_files:
        print(f"File {train_dataset_filename} not found in the zip file.")
        exit(1)
    if test_dataset_filename not in zip_files:
        print(f"File {test_dataset_filename} not found in the zip file.")
        exit(1)


# load raw data
with zipfile.ZipFile(raw_data_zip, 'r') as zf:
    train_path = zipfile.Path(zf, train_dataset_filename)
    test_path = zipfile.Path(zf, test_dataset_filename)

    with train_path.open() as f:
        train_data = pd.read_csv(f, engine='pyarrow')

    with test_path.open() as f:
        test_data = pd.read_csv(f, engine='pyarrow')

# remove id column
train_data = train_data.drop(columns=["ď»żid"])
test_data = test_data.drop(columns=["ď»żid"])


# one-hot encode categorical features
columns_to_one_hot_encode = train_data.select_dtypes(include=["object"]).columns.to_list()
columns_to_one_hot_encode.remove("attack_cat")

one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").set_output(transform="pandas")
train_encoded = one_hot_encoder.fit_transform(train_data[columns_to_one_hot_encode])
test_encoded = one_hot_encoder.transform(test_data[columns_to_one_hot_encode])

train_data = pd.concat([train_data, train_encoded], axis=1).drop(columns=columns_to_one_hot_encode)
test_data = pd.concat([test_data, test_encoded], axis=1).drop(columns=columns_to_one_hot_encode)

# label encode the attack category
label_encoder = LabelEncoder()
concatenated_data = pd.concat([train_data["attack_cat"], test_data["attack_cat"]], axis=0, ignore_index=True)
label_encoder.fit(concatenated_data)
train_data["attack_cat"] = label_encoder.transform(train_data["attack_cat"])
test_data["attack_cat"] = label_encoder.transform(test_data["attack_cat"])

# save the processed data
save_preprocessed_data(
    train_data, 
    dir = processed_data_root_directory / "train", 
    filename_prefix = "UNSW-NB15_train", 
    label_encoder = label_encoder, 
    num_files = 1
)

save_preprocessed_data(
    test_data, 
    dir = processed_data_root_directory / "test", 
    filename_prefix = "UNSW-NB15_test", 
    label_encoder = label_encoder, 
    num_files = 1
)