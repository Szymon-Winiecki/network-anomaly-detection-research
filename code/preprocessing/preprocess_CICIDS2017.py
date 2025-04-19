from pathlib import Path
import sys
import zipfile

import pandas as pd

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


this_directory = Path(__file__).parent.resolve()

utils_directory = this_directory / Path("../utils/")
sys.path.append(str(utils_directory))
from dataset_utils import save_preprocessed_data

# path do the archive with the raw data
raw_data_zip = this_directory / Path("../../data/CIC-IDS2017/raw/MachineLearningCSV.zip")

processed_data_root_directory = this_directory / Path("../../data/CIC-IDS2017/preprocessed/")

# check if the zip file exists
if not raw_data_zip.exists():
    print(f"Raw data zip file {raw_data_zip} does not exist.")
    exit(1)


with zipfile.ZipFile(raw_data_zip, 'r') as zf:

    # check if the files inside the zip file exist
    zip_files = zf.namelist()
    if len(zip_files) != 9: # 8 files + 1 directory
        print(f"Expected 8 files in the zip file, but found {len(zip_files) - 1}.")
        exit(1)

    # remove the directory from the list
    for file in zip_files:
        if file.endswith("/"):
            zip_files.remove(file)
            break

    # read header from the first file
    with zipfile.Path(zf, zip_files[0]).open() as f:
        header = pd.read_csv(f, nrows=0).columns.to_list()

    # find index of unique columns
    not_duplicated_columns = []
    for i in range(len(header)):
        if header[i] not in header[:i]:
            not_duplicated_columns.append(i)

    # load raw data
    paths = [zipfile.Path(zf, file) for file in zip_files]
    dfs = []
    for path in paths:
        with path.open() as f:
            dfs.append(pd.read_csv(f, engine='c', usecols=not_duplicated_columns))

    data = pd.concat(dfs, ignore_index=True, axis=0)


# trim whitespaces from column names
data.columns = data.columns.str.strip()

# rename Label to label to unify with other datasets
data.rename(columns={"Label": "label"}, inplace=True)

# replace -inf and inf with NaN
data.replace([-np.inf, np.inf], np.nan, inplace=True)

# one-hot encode categorical features
columns_to_one_hot_encode = data.select_dtypes(include=["object"]).columns.to_list()
columns_to_one_hot_encode.remove("label")

one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary").set_output(transform="pandas")
encoded = one_hot_encoder.fit_transform(data[columns_to_one_hot_encode])
data = pd.concat([data, encoded], axis=1).drop(columns=columns_to_one_hot_encode)

# label encode the attack category (specified in label column)
label_encoder = LabelEncoder()
data["attack_cat"] = label_encoder.fit_transform(data["label"])

# binarize the label column
data["label"] = data["label"].apply(lambda x: 0 if x == "BENIGN" else 1)

# save the processed data
save_preprocessed_data(
    data, 
    dir = processed_data_root_directory / "full", 
    filename_prefix = "CICIDS2017", 
    label_encoder = label_encoder, 
    num_files = 10
)