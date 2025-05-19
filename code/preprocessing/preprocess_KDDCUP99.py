import gzip
import pandas as pd
from pathlib import Path
import re
import sys

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

this_directory = Path(__file__).parent.resolve()

utils_directory = this_directory / Path("../utils/")
sys.path.append(str(utils_directory))
from dataset_utils import save_preprocessed_data, train_test_split_01

RANDOM_STATE = 42

LOAD_FULL_DATASET = False # if True, load the full dataset, if False, load 10% version of the dataset

# number of anomaly records to be placed in the validation set
# number of normal val records is calculated to keep proportion from test set
VAL_NUM_ANOMALIES = 125218

raw_data_directory = this_directory / Path("../../data/KDDCUP99/raw/")

if LOAD_FULL_DATASET:
    train_dataset_file = raw_data_directory / Path("kddcup.data.gz")
else:
    train_dataset_file = raw_data_directory / Path("kddcup.data_10_percent.gz")

test_dataset_file = raw_data_directory / Path("corrected.gz")

features_file = raw_data_directory / Path("kddcup.names")

processed_data_root = this_directory / Path("../../data/KDDCUP99/preprocessed/")

with open(features_file, "r") as f:
    features_names = f.readlines()

# drop first line - it is a list of possible attacks
features_names = features_names[1:]

# extract features names
# line format: "feature_name: feature_type."
m = re.compile(r"(\w+): (\w+)\.")
features_names = [m.match(line).groups()[0] for line in features_names]

# add label - attack_cat
features_names.append("attack_cat")

# load data from files
with gzip.open(train_dataset_file, "r") as f:
    train_df = pd.read_csv(f, names=features_names, engine='pyarrow')

with gzip.open(test_dataset_file, "r") as f:
    test_df = pd.read_csv(f, names=features_names, engine='pyarrow')

# remove dot from the attack_cat
train_df["attack_cat"] = train_df["attack_cat"].str[:-1]
test_df["attack_cat"] = test_df["attack_cat"].str[:-1]

# create target column - binarize attack_cat to normal (0) and attack (1)
train_df["label"] = train_df["attack_cat"].apply(lambda x: 0 if x == "normal" else 1)
test_df["label"] = test_df["attack_cat"].apply(lambda x: 0 if x == "normal" else 1)

# concatenated data for fitting (because there are additional attack types in test data)
train_test_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# encode attack_cat
label_encoder = LabelEncoder()
label_encoder.fit(train_test_df["attack_cat"])
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
one_hot_encoder = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary'), ["protocol_type", "service", "flag"]),
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
    dir = processed_data_root / "train", 
    filename_prefix = "KDDCUP99_train", 
    label_encoder = label_encoder, 
    num_files = 1
)

save_preprocessed_data(
    val_df, 
    dir = processed_data_root / "val", 
    filename_prefix = "KDDCUP99_val", 
    label_encoder = label_encoder, 
    num_files = 3
)

save_preprocessed_data(
    test_df, 
    dir = processed_data_root / "test", 
    filename_prefix = "KDDCUP99_test", 
    label_encoder = label_encoder, 
    num_files = 5
)

