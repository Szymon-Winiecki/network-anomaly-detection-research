import json
import sys
from pathlib import Path
import random

import pandas as pd

from torch import tensor
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

sys.path.append('./utils')
from dataset_utils import load_features_dtypes, load_preprocessed_data, load_label_encoder, train_test_split_01

def load_UNSWNB15Dataset(data_dir):
    dtypes  = load_features_dtypes(data_dir)
    data = load_preprocessed_data(data_dir, feature_dtypes = dtypes, na_values = [" "])

    return data


def split_UNSWNB15Dataset(data_dir : str | Path, 
                          data : pd.DataFrame | None = None, 
                          max_num_records : int | None = None,
                          records_num : dict = {},
                          normal_records_num : dict = {},
                          transformer : Pipeline | None = None, 
                          random_state : int = None):
    
    """ Splits the UNSW-NB15 dataset into train, validation and test sets. 
        Args meaning the same as in UNSWNB15Dataset class constructor.
    """

    random_state = random_state if random_state is not None else random.randint(0, 2**32 - 1)

    if data is None:
        data = load_UNSWNB15Dataset(data_dir)

    train_dataset = UNSWNB15Dataset(data_dir = data_dir,
                                   data = data,
                                   type = "train",
                                   max_num_records = max_num_records,
                                   records_num = records_num,
                                   normal_records_num = normal_records_num,
                                   transformer = transformer,
                                   random_state = random_state)
    
    val_dataset = UNSWNB15Dataset(data_dir = data_dir,
                                    data = data,
                                    type = "val",
                                    max_num_records = max_num_records,
                                    records_num = records_num,
                                    normal_records_num = normal_records_num,
                                    transformer = transformer,
                                    random_state = random_state)
    
    test_dataset = UNSWNB15Dataset(data_dir = data_dir,
                                    data = data,
                                    type = "test",
                                    max_num_records = max_num_records,
                                    records_num = records_num,
                                    normal_records_num = normal_records_num,
                                    transformer = transformer,
                                    random_state = random_state)
    
    return train_dataset, val_dataset, test_dataset


# class pre-declaration just to enable UNSWNB15Dataset for type hinting
class UNSWNB15Dataset(Dataset):
    pass

class UNSWNB15Dataset(Dataset):
    """ UNSW-NB15 Dataset """

    def __init__(self, 
                 data_dir : str | Path, 
                 data : pd.DataFrame | None = None, 
                 type : str = "train",
                 max_num_records : int | None = None,
                 records_num : dict = {},
                 normal_records_num : dict = {},
                 transformer : Pipeline | None = None, 
                 random_state : int = 42):
        """
        Parameters:
            data_dir (str | path): Path to the directory containing the dataset (directory with preprocesed csv files).
            data (pd.DataFrame | None): Dataframe containing the dataset. If None, the data will be loaded from the data_dir.
            type (str): Type of the dataset to create. Possible values: "train", "val", "test"
            max_num_records (int): Maximum number of records in all train, val and test datasets. If None, the size of the dataset is the limit. It is used as a base to calcuate number of reocrds if it is given as a proportion.
            num_records (dict): Dictionary containing number (or proportion) of records to return for each dataset. 
                If provided, it should contain only keys "train", "val" and "test" with values inidcating number of records to return 
                (int for actual number, float for proportion relative to max_num_records, the rest will be splitted evenly to None positions).
            normal (dict): Dictionary containing number (or proportion) of normal records to return for each dataset (structure as num_records). None keeps orignal proportion of normal records.
            transformer (Pipeline): Data transformation pipeline to apply to the data. If None, no transformation will be applied. If test_for is not None, transformer should be the same as for the train dataset (transformer is fitted only on the train dataset).
            random_state (int): Random seed for reproducibility, should be set to the same value for all datasets types (train, val, test) in one split to avoid duplicates.

        Raises:
            FileNotFoundError: If feature types file is not found in the data directory
            ValueError: If number of records or number of normal records to return is invalid or there is not enaugh records to create dataset
            NotADirectoryError: If data directory does not exist
        """

        # basic params validation

        if data_dir is None and data is None:
            raise ValueError("Data or data directory is required.")

        if data_dir is not None:
            if not isinstance(data_dir, Path):
                data_dir = Path(data_dir)
            if not data_dir.is_dir():
                raise NotADirectoryError("Data directory does not exist.")


        self.random_state = random_state

        # load data from data_dir if not provided directly
        if data is None:
            feature_dtypes = load_features_dtypes(data_dir)
            data = load_preprocessed_data(data_dir, feature_dtypes, na_values = [" "])

        # load label encoder
        self.label_encoder = load_label_encoder(data_dir)
        
        # get actual max number of records to return
        if max_num_records is None:
            max_num_records = len(data)
        elif isinstance(max_num_records, float) and max_num_records > 0 and max_num_records <= 1:
            max_num_records = int(max_num_records * len(data))
        elif isinstance(max_num_records, int) and max_num_records > 0 and max_num_records <= len(data):
            pass
        else:
            raise ValueError("Invalid max_num_records value.")
        
        normal_df = data[data["Label"] == 0]
        attack_df = data[data["Label"] == 1]
        
        # get actual number of differnt types of records to return in each datset
        records_num = self.calc_num_records(records_num, max_num_records)
        normal_records_num = self.calc_normal_records(normal_records_num, records_num, default_normal_proportion = len(normal_df) / len(data))
        attack_records_num = self.calc_attack_records(records_num, normal_records_num)

        normal_total = sum(normal_records_num.values())
        attack_total = sum(attack_records_num.values())

        # check if there are enough records to create the dataset
        if normal_total > len(normal_df):
            raise ValueError("Not enough normal records to create dataset.")
        if attack_total > len(attack_df):
            raise ValueError("Not enough attack records to create dataset.")

        normal_df = normal_df.sample(normal_total, random_state=self.random_state)
        attack_df = attack_df.sample(attack_total, random_state=self.random_state)

        # split data into train, validation and test sets and get the required number of records of each type

        normal_train_val, normal_test = train_test_split_01(normal_df, test_size=normal_records_num["test"], random_state=self.random_state)
        normal_train, normal_val = train_test_split_01(normal_train_val, test_size=normal_records_num["val"], random_state=self.random_state)

        attack_train_val, attack_test = train_test_split_01(attack_df, test_size=attack_records_num["test"], stratify=attack_df["attack_cat"], random_state=self.random_state)
        attack_train, attack_val = train_test_split_01(attack_train_val, test_size=attack_records_num["val"], stratify=attack_train_val["attack_cat"], random_state=self.random_state)

        if type == "train":
            data = pd.concat([normal_train, attack_train], ignore_index=True, axis=0)
        elif type == "val":
            data = pd.concat([normal_val, attack_val], ignore_index=True, axis=0)
        elif type == "test":
            data = pd.concat([normal_test, attack_test], ignore_index=True, axis=0)

        # shuffle the data
        data = data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # separate data, target (0: normal, 1: anomaly) and attack categories
        self.x = data.drop(columns=["Label", "attack_cat"]).astype("float32")
        self.y = data["Label"].astype("int64")
        self.attack_cat = data["attack_cat"].astype("int64")

        # apply data transformation pipeline
        if transformer is not None and type == "train":
            self.x = transformer.fit_transform(self.x)
        elif transformer is not None:
            self.x = transformer.transform(self.x)

        # convert data to torch.tensor
        self.x = tensor(self.x.values)
        self.y = tensor(self.y.values)
        self.attack_cat = tensor(self.attack_cat.values)



    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.attack_cat[idx]
    

    def normalize_train_val_test_dict(self, dict):
        """ Normalize the given dictionary to contain these and only these keys: "train", "val" and "test".
            If a key is not present, it is set to None.

        Args:
            dict (dict): Dictionary to normalize.

        Returns:
            dict: Normalized dictionary containing keys "train", "val" and "test"
        """
        return {
            "train": dict.get("train", None),
            "val": dict.get("val", None),
            "test": dict.get("test", None)
        }
    
    def calc_num_records(self, num_records, max_num_records):
        """ Calculate number of records to return for each dataset (train, val, test) based on the given proportions, quantities or None values. 
            If a proportion is given, it is converted to a quantity based on the max_num_records.
            Remaining records are split evenly to None values.

        Args:
            num_records (dict): Dictionary containing number of records to return for each dataset (train, val, test).
            max_num_records (int): Maximum number of records to return.

        Returns:
            dict: Dictionary containing number of records to return for each dataset (train, val, test).
        """

        num_records = self.normalize_train_val_test_dict(num_records)

        # convert proportions to quantity
        for k in num_records:
            if num_records[k] is None:
                continue
            elif isinstance(num_records[k], int) and num_records[k] >= 0:
                continue
            elif isinstance(num_records[k], float) and num_records[k] > 0.0 and num_records[k] <= 1.0:
                num_records[k] = int(num_records[k] * max_num_records)
            elif num_records[k] >= 0:
                num_records[k] = int(num_records[k])
            else:
                raise ValueError("Invalid number of records to return.")
        
        none_count = sum((1 for k in num_records if num_records[k] is None))
        current_records_used = sum(filter(None, num_records.values()))

        if current_records_used > max_num_records:
                raise ValueError("Number of records to return exceeds maximum.")

        for k in num_records:
            if num_records[k] is None:
                num_records[k] = (max_num_records - current_records_used) // none_count
                none_count -= 1
                current_records_used += num_records[k]

        return num_records
    
    def calc_normal_records(self, normal, num_records, default_normal_proportion):
        """ Calculate number of normal records to return for each dataset (train, val, test) based on the given proportions, quantities or None values. 
            If a proportion is given, it is converted to a quantity based on the number of records to return for this dataset.
            If None, the default_normal_proportion is used to calculate the number of normal records to return.

        Args:
            normal (dict): Dictionary containing number of normal records to return for each dataset (train, val, test).
            num_records (dict): Dictionary containing total number of records to return for each dataset (train, val, test).
            default_normal_proportion (float): Default proportion of normal records to use if None is given.
        """

        normal = self.normalize_train_val_test_dict(normal)
        
        for k in normal:
            if normal[k] is None:
                normal[k] = default_normal_proportion * num_records[k]
            elif isinstance(normal[k], float) and normal[k] > 0 and normal[k] <= 1:
                normal[k] = int(normal[k] * num_records[k])
            elif isinstance(normal[k], int) and normal[k] > 0 and normal[k] <= num_records[k]:
                continue
            else:
                raise ValueError("Invalid number of normal records to return.")
        
        return normal
    
    def calc_attack_records(self, num_records, normal_records):
        """ Calculate number of attack records to return for each dataset (train, val, test) based on the given number of all records and normal records to return.
            The number of attack records is calculated as the difference between the total number of records and the number of normal records.
        Args:
            num_records (dict): Dictionary containing number of records to return for each dataset (train, val, test).
            normal_records (dict): Dictionary containing number of normal records to return for each dataset (train, val, test).

        Returns:
            dict: Dictionary containing number of attack records to return for each dataset (train, val, test).
        """
        return {k: num_records[k] - normal_records[k] for k in num_records}