from pathlib import Path
import json
import sys

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

this_directory = Path(__file__).parent.resolve()

utils_directory = this_directory / Path("../utils/")
sys.path.append(str(utils_directory))
from dataset_utils import save_preprocessed_data

raw_data_directory = this_directory / Path("../../data/UNSW-NB15/raw/CSV Files")
raw_data_filenames = ["UNSW-NB15_1.csv", "UNSW-NB15_2.csv", "UNSW-NB15_3.csv", "UNSW-NB15_4.csv"]
features_description_filename = "NUSW-NB15_features.csv"

processed_data_directory = this_directory / Path("../../data/UNSW-NB15/preprocessed")
num_processed_files = 10

# Load features description
features_description = pd.read_csv(raw_data_directory / features_description_filename, header=0, encoding='latin1')

feature_names = features_description["Name"].to_list()
feature_types = features_description["Type "].to_list()

# map feature types to pandas dtypes
type_map = {
    "integer": "Int64",
    "Integer": "Int64",
    "binary": "Int64",
    "Binary": "Int64",
    "Timestamp": "float64",
    "Float": "float64",
    "nominal": "object"
}

feature_dtypes = {feature: type_map[type] for feature, type in zip(feature_names, feature_types)}

# some port numbers are stored as a hexadecimal number, so we need to treat it as a string 
# (pandas can't handle hex numbers as Int64 when reading from csv). 
# Moreover, port is reather a categorical feature than a numerical one.
feature_dtypes["sport"] = "object"
feature_dtypes["dsport"] = "object"

# load raw data from all files and concatenate them
raw_data = pd.concat((pd.read_csv(raw_data_directory / f, names=feature_names, dtype=feature_dtypes, na_values=[" "], engine='pyarrow') for f in raw_data_filenames), ignore_index=True, axis=0)

# remove source and destination IP addresses and ports as they are not useful for the analysis
columns_to_drop = ["srcip", "sport", "dstip", "dsport"]
processed_data = raw_data.drop(columns=columns_to_drop)

# one-hot encode categorical features
columns_to_one_hot_encode = ["proto", "state", "service"]
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").set_output(transform="pandas")
encoded = one_hot_encoder.fit_transform(processed_data[columns_to_one_hot_encode])
processed_data = pd.concat([processed_data, encoded], axis=1).drop(columns=columns_to_one_hot_encode)

# label encode the attack category
label_encoder = LabelEncoder()
processed_data["attack_cat"] = label_encoder.fit_transform(processed_data["attack_cat"])


# save the processed data

save_preprocessed_data(
    processed_data, 
    feature_dtypes = feature_dtypes, 
    label_encoder = label_encoder, 
    dir = processed_data_directory, 
    filename_prefix = "UNSW-NB15", 
    num_files = num_processed_files
)