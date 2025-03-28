import pandas as pd
from torch.utils.data import Dataset
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import tensor

# class pre-declaration just to enable UNSWNB15Dataset for type hinting
class UNSWNB15Dataset(Dataset):
    pass

class UNSWNB15Dataset(Dataset):
    """ UNSW-NB15 Dataset """

    def __init__(self, data_dir, num_records : int | float | None = None, normal : int | float | None = None, test_for : UNSWNB15Dataset | None = None, random_state : int = 42):
        """
        Parameters:
            data_dir (str): Path to the directory containing the dataset (directory with preprocesed csv files)
            num_records (int | float): Number (int) or proportion (float) of records to return. If None, return all records.
            normal (int | float): If float, it indicates the propotion of normal records to return. If int, it indicates number of them. If None, original proportion will be kept.
            test_for (UNSWNB15Dataset): If None, this will create a train dataset. Otherwise, previously created train dataset should be provided and it will create a test dataset for it.
            random_state (int): Random seed for reproducibility

        Raises:
            FileNotFoundError: If feature types file is not found in the data directory
            ValueError: If number of records or number of normal records to return is invalid or there is not enaugh records to create dataset
            NotADirectoryError: If data directory does not exist
        """

        # basic params validation

        if data_dir is None:
            raise ValueError("Data directory path is required.")
        
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)

        if not data_dir.is_dir():
            raise NotADirectoryError("Data directory does not exist.")


        self.data_dir = data_dir

        # for train dataset use provided or default random state, for test dataset use random state of the train dataset to ensure train and test data separation
        self.random_state = random_state if test_for is None else test_for.random_state

        # load feature types
        dtype_file = list(data_dir.glob("*dtypes.json"))
        if len(dtype_file) == 0:
            raise FileNotFoundError("Feature types file not found in the data directory.")
        
        with open(dtype_file[0], "r") as f:
            feature_dtypes = json.load(f)


        # load data from csv files and concatenate them
        data_files = data_dir.glob("*.csv")
        data = pd.concat((pd.read_csv(f, header=0, dtype=feature_dtypes, na_values=[" "]) for f in data_files), ignore_index=True, axis=0)
        
        # get actual number of records to return
        if num_records is None:
            self.num_records = len(data)
        elif isinstance(num_records, float) and num_records > 0 and num_records <= 1:
            self.num_records = int(num_records * len(data))
        elif isinstance(num_records, int) and num_records > 0 and num_records <= len(data):
            self.num_records = num_records
        else:
            raise ValueError("Invalid number of records to return.")
        
        if test_for == None and self.num_records == len(data):
            # if all records are to be loaded no need to do anything else
            return
        
        normal_df = data[data["Label"] == 0]
        attack_df = data[data["Label"] == 1]
        
        # get number of normal records to return
        if normal is None:
            normal_frac = data["Label"].value_counts(normalize=True).loc[0]
            self.normal_records = int(normal_frac * self.num_records)
        elif isinstance(normal, float) and normal > 0 and normal <= 1:
            self.normal_records = int(normal * self.num_records)
        elif isinstance(normal, int) and normal > 0 and normal <= len(normal_df):
            self.normal_records = normal
        else:
            raise ValueError("Invalid number of normal records to return.")
        
        # get number of attack records to return
        self.attack_records = self.num_records - self.normal_records

        # split data into train and test sets and get the required number of records of each type

        normal_train_size = self.normal_records if test_for is None else test_for.normal_records 

        normal_train, normal_test = train_test_split(normal_df, train_size=normal_train_size, random_state=self.random_state)

        if test_for is not None:
            if self.normal_records > len(normal_test):
                raise ValueError("Not enough normal records to create test dataset.")
            normal_test = normal_test.sample(self.normal_records, random_state=self.random_state)

        attack_train_size = self.attack_records if test_for is None else test_for.attack_records

        attack_train, attack_test = train_test_split(attack_df, train_size=attack_train_size, stratify=attack_df["attack_cat"], random_state=self.random_state)

        if test_for is not None:
            if self.attack_records > len(attack_test):
                raise ValueError("Not enough attack records to create test dataset.")
            attack_test, _ = train_test_split(attack_test, train_size=self.attack_records, stratify=attack_test["attack_cat"], random_state=self.random_state)

        if test_for is None:
            data = pd.concat([normal_train, attack_train], ignore_index=True, axis=0)  
        else:
            data = pd.concat([normal_test, attack_test], ignore_index=True, axis=0)

        # drop attack_cat column as it was used only for stratification
        data.drop(columns=["attack_cat"], inplace=True)

        # shuffle the data
        data = data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # separate data and labels

        self.x = data.drop(columns=["Label"]).astype("float32")
        self.y = data["Label"].astype("int64")

        # convert data to torch.tensor
        self.x = tensor(self.x.values)
        self.y = tensor(self.y.values)



    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        