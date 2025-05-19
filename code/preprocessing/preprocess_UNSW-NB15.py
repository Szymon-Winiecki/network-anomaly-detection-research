from pathlib import Path
import sys
import zipfile

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

this_directory = Path(__file__).parent.resolve()

utils_directory = this_directory / Path("../utils/")
sys.path.append(str(utils_directory))
from dataset_utils import save_preprocessed_data, train_test_split_01

RANDOM_STATE = 42

# number of anomaly records to be placed in the validation set
# number of normal val records is calculated to keep proportion from test set
VAL_NUM_ANOMALIES = 12252

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
        train_df = pd.read_csv(f, engine='pyarrow')

    with test_path.open() as f:
        test_df = pd.read_csv(f, engine='pyarrow')

# remove id column
train_df = train_df.drop(columns=["ď»żid"])
test_df = test_df.drop(columns=["ď»żid"])

# label encode the attack category
label_encoder = LabelEncoder()
concatenated_data = pd.concat([train_df["attack_cat"], test_df["attack_cat"]], axis=0, ignore_index=True)
label_encoder.fit(concatenated_data)
train_df["attack_cat"] = label_encoder.transform(train_df["attack_cat"])
test_df["attack_cat"] = label_encoder.transform(test_df["attack_cat"])

# split train data into train and validation sets
test_normal_anomaly_ratio = test_df[test_df["label"] == 0].shape[0] / test_df[test_df["label"] == 1].shape[0]
val_num_normal = int(VAL_NUM_ANOMALIES * test_normal_anomaly_ratio)

_, val_anomaly_df = train_test_split_01(
    train_df[train_df["label"] == 1], 
    test_size=VAL_NUM_ANOMALIES, 
    random_state=RANDOM_STATE,
    stratify=train_df[train_df["label"] == 1]["attack_cat"]
)

train_df, val_normal_df = train_test_split_01(
    train_df[train_df["label"] == 0], 
    test_size=val_num_normal, 
    random_state=RANDOM_STATE
)

val_df = pd.concat([val_normal_df, val_anomaly_df], axis=0, ignore_index=True)

# one-hot encode categorical features
columns_to_one_hot_encode = train_df.select_dtypes(include=["object"]).columns.to_list()

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
    filename_prefix = "UNSW-NB15_train", 
    label_encoder = label_encoder, 
    num_files = 1
)

save_preprocessed_data(
    val_df, 
    dir = processed_data_root_directory / "val", 
    filename_prefix = "UNSW-NB15_val", 
    label_encoder = label_encoder, 
    num_files = 1
)

save_preprocessed_data(
    test_df, 
    dir = processed_data_root_directory / "test", 
    filename_prefix = "UNSW-NB15_test", 
    label_encoder = label_encoder, 
    num_files = 2
)