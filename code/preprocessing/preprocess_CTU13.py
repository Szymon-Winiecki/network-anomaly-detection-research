import tarfile
import pandas as pd
from pathlib import Path
import sys

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42
VERBOSE = False

this_directory = Path(__file__).parent.resolve()

utils_directory = this_directory / Path("../utils/")
sys.path.append(str(utils_directory))
from dataset_utils import save_preprocessed_data, train_test_split_01

raw_data_directory = this_directory / Path("../../data/CTU-13/raw/")

subfolders_to_process = ["08", "09", "10", "13"]

scenarios_balance = {
    "08": {
        "test_num_anomalies": 3677,
        "test_num_normal": 43694,
        "train_num_normal": 29128,
    },
    "09": {
        "test_num_anomalies": 110993,
        "test_num_normal": 17981,
        "train_num_normal": 11986,
    },
    "10": {
        "test_num_anomalies": 63812,
        "test_num_normal": 9509,
        "train_num_normal": 6338,
    },
    "13": {
        "test_num_anomalies": 24002,
        "test_num_normal": 19164,
        "train_num_normal": 12775,
    },
}

train_dataset_file = raw_data_directory / Path("kddcup.data.gz")
test_dataset_file = raw_data_directory / Path("corrected.gz")

processed_data_root = this_directory / Path("../../data/CTU-13/preprocessed/")

for subfolder in subfolders_to_process:
    current_directory = raw_data_directory / Path(subfolder)

    tar_files = list(current_directory.glob("*.tar.gz"))
    csv_files = list(current_directory.glob("*.csv"))

    if len(tar_files) > 0:
        if VERBOSE:
            print(f"Loading from .tar.gz archive in {subfolder}")
        with tarfile.open(tar_files[0], "r") as tar:
            for member in tar:
                f = tar.extractfile(member)
                if f is not None:
                    data = pd.read_csv(f, engine='pyarrow')
                    break

    elif len(csv_files) > 0:
        if VERBOSE:
            print(f"Loading from .csv file in {subfolder}")
        with open(csv_files[0], "r") as f:
            data = pd.read_csv(f, engine='pyarrow')
    
    else:
        print(f"No data files found in {current_directory}, skipping.")
        continue

    # create label column - binarize Label to normal (0) and attack (1)
    data["label"] = data["Label"].apply(lambda x: 1 if "botnet" in x.lower() else 0)

    if VERBOSE:
        print("Class distribution:")
        print(data["label"].value_counts())

    # create attack_cat column - same as label for now
    data["attack_cat"] = data["Label"].apply(lambda x: 1 if "botnet" in x.lower() else 0)

    # remove Label (note capital L) column
    data = data.drop(columns=["Label"])

    # drop identyfying or unnecessary columns
    data = data.drop(columns=["StartTime", "SrcAddr", "DstAddr"])

    # encode featres

    one_hot_encoder = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary'), ["Proto", "Dir", "State", "sTos", "dTos"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    data = one_hot_encoder.fit_transform(data)

    # encode attack_cat

    label_encoder = LabelEncoder()
    label_encoder.fit(data["attack_cat"])
    data["attack_cat"] = label_encoder.transform(data["attack_cat"])


    # split data into train and test

    num_normal = data[data["label"] == 0].shape[0]
    num_attack = data[data["label"] == 1].shape[0]

    num_test_attack = scenarios_balance[subfolder]["test_num_anomalies"]
    num_train_attack = 0
    num_val_attack = min(num_attack - num_test_attack, num_test_attack)

    val_test_ratio = num_val_attack / num_test_attack

    num_test_normal = scenarios_balance[subfolder]["test_num_normal"]
    num_train_normal = scenarios_balance[subfolder]["train_num_normal"]
    num_val_normal = int(num_test_normal * val_test_ratio)
    num_val_normal = min(num_val_normal, num_normal - num_test_normal - num_train_normal)
    

    train_normal_df, test_normal_df = train_test_split_01(data[data["label"] == 0], test_size=num_test_normal, random_state=RANDOM_STATE)
    train_normal_df, val_normal_df = train_test_split_01(train_normal_df, test_size=num_val_normal, random_state=RANDOM_STATE)
    train_normal_df, _ = train_test_split_01(train_normal_df, train_size=num_train_normal, random_state=RANDOM_STATE)

    train_attack_df, test_attack_df = train_test_split_01(data[data["label"] == 1], test_size=num_test_attack, random_state=RANDOM_STATE)
    train_attack_df, val_attack_df = train_test_split_01(train_attack_df, test_size=num_val_attack, random_state=RANDOM_STATE)
    train_attack_df, _ = train_test_split_01(train_attack_df, train_size=num_train_attack, random_state=RANDOM_STATE)

    # concatenate normal and attack records
    train_df = pd.concat([train_normal_df, train_attack_df], axis=0)
    val_df = pd.concat([val_normal_df, val_attack_df], axis=0)
    test_df = pd.concat([test_normal_df, test_attack_df], axis=0)

    # save the processed data
    save_preprocessed_data(
        train_df, 
        dir = processed_data_root / subfolder / "train", 
        filename_prefix = f"CTU13-{subfolder}_train", 
        label_encoder = label_encoder, 
        num_files = 10
    )

    save_preprocessed_data(
        val_df, 
        dir = processed_data_root / subfolder / "val", 
        filename_prefix = f"CTU13-{subfolder}_val", 
        label_encoder = label_encoder, 
        num_files = 10
    )

    save_preprocessed_data(
        test_df, 
        dir = processed_data_root / subfolder / "test", 
        filename_prefix = f"CTU13-{subfolder}_test", 
        label_encoder = label_encoder, 
        num_files = 10
    )

    


