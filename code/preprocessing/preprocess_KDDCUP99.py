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
from dataset_utils import save_preprocessed_data

raw_data_directory = this_directory / Path("../../data/KDDCUP99/raw/")

train_dataset_file = raw_data_directory / Path("kddcup.data.gz")
test_dataset_file = raw_data_directory / Path("corrected.gz")
features_file = raw_data_directory / Path("kddcup.names")

processed_data_root = this_directory / Path("../../data/KDDCUP99/preprocessed/")
processed_train_data_directory = processed_data_root / Path("train/")
processed_test_data_directory = processed_data_root / Path("test/")

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

# encode featres

one_hot_encoder = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary'), ["protocol_type", "service", "flag"]),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

# concatenated data fro fitting (because there are additional attack types in test data)
train_test_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# one-hot encode categorical features
one_hot_encoder.fit(train_test_df)
train_df = one_hot_encoder.transform(train_df)
test_df = one_hot_encoder.transform(test_df)

# encode attack_cat
label_encoder = LabelEncoder()
label_encoder.fit(train_test_df["attack_cat"])
train_df["attack_cat"] = label_encoder.transform(train_df["attack_cat"])
test_df["attack_cat"] = label_encoder.transform(test_df["attack_cat"])

# save the processed data
save_preprocessed_data(
    train_df, 
    feature_dtypes = None, 
    label_encoder = label_encoder, 
    dir = processed_train_data_directory, 
    filename_prefix = "KDDCUP99_train", 
    num_files = 10
)

save_preprocessed_data(
    test_df, 
    feature_dtypes = None, 
    label_encoder = label_encoder, 
    dir = processed_test_data_directory, 
    filename_prefix = "KDDCUP99_test", 
    num_files = 1
)

