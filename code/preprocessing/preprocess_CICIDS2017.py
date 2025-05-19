from pathlib import Path
import sys
import zipfile

import pandas as pd

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

this_directory = Path(__file__).parent.resolve()

utils_directory = this_directory / Path("../utils/")
sys.path.append(str(utils_directory))
from dataset_utils import save_preprocessed_data, train_test_split_01

# path do the archive with the raw data
raw_data_zip = this_directory / Path("../../data/CIC-IDS2017/raw/MachineLearningCSV.zip")

processed_data_root_directory = this_directory / Path("../../data/CIC-IDS2017/preprocessed/")

RANDOM_STATE = 42
TEST_NUM_ANOMALIES = 278823
TEST_NUM_NORMAL = 278823
VAL_NUM_ANOMALIES = 278823

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

# label encode the attack category (specified in label column)
label_encoder = LabelEncoder()
data["attack_cat"] = label_encoder.fit_transform(data["label"])

# binarize the label column
data["label"] = data["label"].apply(lambda x: 0 if x == "BENIGN" else 1)

# split data into train and test

num_normal = data[data["label"] == 0].shape[0]
num_attack = data[data["label"] == 1].shape[0]

test_normal_anomlay_ratio = TEST_NUM_NORMAL / TEST_NUM_ANOMALIES
val_num_normal = int(VAL_NUM_ANOMALIES * test_normal_anomlay_ratio)


train_normal_df, test_normal_df = train_test_split_01(data[data["label"] == 0], test_size=TEST_NUM_NORMAL, random_state=RANDOM_STATE)
train_normal_df, val_normal_df = train_test_split_01(train_normal_df, test_size=val_num_normal, random_state=RANDOM_STATE)

train_attack_df, test_attack_df = train_test_split_01(data[data["label"] == 1], test_size=TEST_NUM_ANOMALIES, stratify=data[data["label"] == 1]["attack_cat"], random_state=RANDOM_STATE)
train_attack_df, val_attack_df = train_test_split_01(train_attack_df, test_size=VAL_NUM_ANOMALIES, stratify=data[data["label"] == 1]["attack_cat"], random_state=RANDOM_STATE)

# concatenate normal and anomaly records
train_df = train_normal_df
val_df = pd.concat([val_normal_df, val_attack_df], axis=0, ignore_index=True)
test_df = pd.concat([test_normal_df, test_attack_df], axis=0, ignore_index=True)

# encode featres

columns_to_one_hot_encode = data.select_dtypes(include=["object"]).columns.to_list()

one_hot_encoder = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary'), columns_to_one_hot_encode),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

train_df = one_hot_encoder.fit_transform(train_df)
val_df = one_hot_encoder.transform(val_df)
test_df = one_hot_encoder.transform(test_df)

# save the processed data
save_preprocessed_data(
    train_df, 
    dir = processed_data_root_directory / "train", 
    filename_prefix = "CICIDS2017_train", 
    label_encoder = label_encoder, 
    num_files = 20
)

save_preprocessed_data(
    val_df, 
    dir = processed_data_root_directory / "val", 
    filename_prefix = "CICIDS2017_val", 
    label_encoder = label_encoder, 
    num_files = 10
)

save_preprocessed_data(
    test_df, 
    dir = processed_data_root_directory / "test", 
    filename_prefix = "CICIDS2017_test", 
    label_encoder = label_encoder, 
    num_files = 10
)