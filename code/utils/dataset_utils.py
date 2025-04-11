import json
from pathlib import Path

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_features_dtypes(dir : str | Path, filename_regex : str = "*dtypes.json"):
    """ Loads features types of a preprocessed dataset in the given directory.

    Args:
        dir (str | Path): Path to the directory with the processed dataset.
        filename_regex (str): Regex pattern to match the feature types file.

    Returns:
        dict: Dictionary with feature names as keys and their types as values.

    Raises:
        FileNotFoundError: If feature types file is not found in the data directory.
    """

    if not isinstance(dir, Path):
            dir = Path(dir)

    dtype_file = list(dir.glob("*dtypes.json"))
    if len(dtype_file) == 0:
        raise FileNotFoundError("Feature types file not found in the directory.")
    
    with open(dtype_file[0], "r") as f:
        feature_dtypes = json.load(f)

    return feature_dtypes

def load_label_encoder(dir : str | Path, filename_regex : str = "*label_encoder.json"):
    """ Loads classes labels and recreates label encoder of a preprocessed dataset in the given directory. 
    
    Args:
        dir (str | Path): Path to the directory with the processed dataset.
        filename_regex (str): Regex pattern to match the label encoder file.

    Returns:
        LabelEncoder: Label encoder with the classes labels loaded
    """
    if not isinstance(dir, Path):
            dir = Path(dir)

    le_file = list(dir.glob("*label_encoder.json"))
    if len(le_file) == 0:
        raise FileNotFoundError("Label encoder file not found in the directory.")
    
    with open(le_file[0], "r") as f:
        le_classes = json.load(f)
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(le_classes)

    # to make sure everyting is ok
    label_encoder.inverse_transform([0])

    return label_encoder

def load_preprocessed_data(dir : str | Path, feature_dtypes : dict | None = None, na_values : list = []):
    """ Loads preprocessed data from gzipped csv files in the given directory."

    Args:
        dir (str | Path): Path to the directory with the preprocessed dataset.
        feature_dtypes (dict | None): Dictionary with feature names as keys and their types as values. If not provided, the dtypes will be loaded from the dir.
        na_values (list): List of values to consider as NA.
    """
    if not isinstance(dir, Path):
            dir = Path(dir)

    if feature_dtypes is None:
       feature_dtypes = load_features_dtypes(dir)

    data_filenames = dir.glob("*.csv.gzip")
    files = (pd.read_csv(f, 
                            header=0, 
                            dtype=feature_dtypes, 
                            na_values=na_values, 
                            engine='pyarrow',
                            compression='gzip'
                        ) for f in data_filenames)
    
    data = pd.concat(files, ignore_index=True, axis=0)

    return data

def save_preprocessed_data(data : pd.DataFrame, feature_dtypes : dict | None, label_encoder : LabelEncoder | None, dir : str | Path, filename_prefix : str, num_files : int = 1):
    """ Saves preprocessed datset in the given directory.
    
    Data is splitted into num_files files.
    The data files will be named as <filename_prefix>_<i>.csv.gzip, where i is the file number.
    The feature types will be saved to a json file named <filename_prefix>_dtypes.json.
    The label encoder (if provided) will be saved to a json file named <filename_prefix>_label_encoder.json.

    Args:
        data (pd.DataFrame): Data to save.
        feature_dtypes (dict | None): Dictionary with feature names as keys and their types as values. If None, the dtypes will be inferred from the data.
        label_encoder (LabelEncoder | None): Label encoder to save. If None, the label encoder will not be saved.
        dir (str | Path): Path to the directory where to save the data.
        filename_prefix (str): Prefix for the filenames. Data files will be named as <filename_prefix>_<i>.csv.gzip, where i is the file number.
        num_files (int): Number of files to split the data into.
    """
    if not isinstance(dir, Path):
            dir = Path(dir)

    # create the directory if it does not exist
    dir.mkdir(parents=True, exist_ok=True)

    # if feature dtypes are not provided, infer them from the data
    if feature_dtypes is None:
        feature_dtypes = {col: data[col].dtype.name for col in data.columns}

    # save feature dtypes to a json file
    with open(dir / f"{filename_prefix}_dtypes.json", "w") as f:
        json.dump(feature_dtypes, f)

    # save label encoder to a json file if provided
    if label_encoder is not None:
        with open(dir / f"{filename_prefix}_label_encoder.json", "w") as f:
            json.dump(label_encoder.classes_.tolist(), f)

    # save the data to num_files files

    rows_per_file = len(data) // num_files
    for i in range(num_files - 1):
        processed_data_path = dir / f"{filename_prefix}_{i}.csv.gzip"

        start = i * rows_per_file
        end = (i + 1) * rows_per_file
        data.iloc[start : end].to_csv(processed_data_path, index=False, compression='gzip')

    # save the last file with the remaining rows
    processed_data_path = dir / f"{filename_prefix}_{num_files - 1}.csv.gzip"
    data.iloc[(num_files - 1) * rows_per_file :].to_csv(processed_data_path, index=False, compression='gzip')

def train_test_split_01(data : pd.DataFrame,
                         test_size = None,
                         train_size = None,
                         random_state=None,
                         shuffle=True,
                         stratify=None):
    """ train_test_split wrapper that allows 0% / 100% split """

    if train_size == 1 or test_size == 0 or train_size == len(data):
        return data, data[0:0]
    
    if train_size == 0 or test_size == 1 or test_size == len(data):
        return data[0:0], data

    return train_test_split(data, 
                            test_size=test_size, 
                            train_size=train_size, 
                            random_state=random_state, 
                            shuffle=shuffle,
                            stratify=stratify)
     