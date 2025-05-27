import sys
from pathlib import Path

from torch import tensor
from torch.utils.data import Dataset

from sklearn.pipeline import Pipeline

import pandas as pd

this_directory = Path(__file__).parent.resolve()
sys.path.append(str(this_directory / '../utils'))
from utils.dataset_utils import load_preprocessed_data, load_label_encoder, train_test_split_01

class IDS_Dataset(Dataset):

    # debug flag to print dataset statistics
    DEBUG = True

    def __init__(self, 
                 root_dir: str | Path, 
                 type: str = "train", 
                 strategy: str = "train_val_test",
                 fold_idx: int = 0,
                 name: str = "undefined",
                 transformer : Pipeline | None = None,
                 random_state: int = 42):
        """IDS (intrusion detection system) dataset class.

        Args:
            root_dir (str | Path): Path to the directory with the processed datasets (root dir of test, train and val sets).
            type (str): Type of the dataset. Can be "train", "val" or "test".
            strategy (str): Strategy for splitting the dataset. Possible options:
                - "train_val_test": split of data the same as from preprocessing: train goes to train, val to val and test to test; (default)
                - "train_test": all normal data from train and val data goes to train dataset; test remains unchanged;
                - "cross_validation": uses a modified cross-validation strategy on train and val data to create train and val sets, while test remains unchanged.
            fold_idx (int): Index of the fold to use for modified cross-validation. Only used if strategy is "cross_validation". Default is 0.
            name (str): Name of the dataset.
            transformer (Pipeline, optional): Optional transformer to be applied on a sample.
            random_state (int, optional): Random state for reproducibility. Only used if strategy is "cross_validation". Should be the same for all folds. Default is 42.
        """
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)

        if type not in ["train", "val", "test"]:
            raise ValueError(f"Invalid type: {type}. Expected 'train', 'val' or 'test'.")
        
        if strategy not in ["train_val_test", "train_test", "cross_validation"]:
            raise ValueError(f"Invalid strategy: {strategy}. Expected 'train_val_test', 'train_test' or 'cross_validation'.")

        self.root_dir = root_dir
        self.type = type

        self.name = name

        self.random_state = random_state

        if strategy == "train_val_test":
            dataset_dir = self.root_dir / self.type
            self.label_encoder = load_label_encoder(dataset_dir)
            data = load_preprocessed_data(dataset_dir)

        elif strategy == "train_test":
            if self.type == "train":
                train_dir = self.root_dir / "train"
                val_dir = self.root_dir / "val"

                self.label_encoder = load_label_encoder(train_dir)

                train_data = load_preprocessed_data(train_dir)
                val_data = load_preprocessed_data(val_dir)
                val_data = val_data[val_data["label"] == 0]  # keep only normal data from validation
                data = pd.concat([train_data, val_data], ignore_index=True)

            elif self.type == "test":
                test_dir = self.root_dir / "test"
                self.label_encoder = load_label_encoder(test_dir)
                data = load_preprocessed_data(test_dir)

            else:
                raise ValueError(f"Invalid type: {type}. Expected 'train' or 'test'.")
            
        elif strategy == "cross_validation":
            if self.type == "test":
                # For test type, we always load the test set
                test_dir = self.root_dir / "test"
                self.label_encoder = load_label_encoder(test_dir)
                data = load_preprocessed_data(test_dir)

            else:
                train_dir = self.root_dir / "train"
                val_dir = self.root_dir / "val"

                self.label_encoder = load_label_encoder(train_dir)

                val_data = load_preprocessed_data(val_dir)
                num_val_anomalies = val_data[val_data["label"] == 1].shape[0]
                num_val_normal = val_data[val_data["label"] == 0].shape[0]

                train_data = load_preprocessed_data(train_dir)
                num_train_normal = train_data[train_data["label"] == 0].shape[0]

                normal_data = pd.concat([train_data[train_data["label"] == 0], val_data[val_data["label"] == 0]], ignore_index=True)
                anomaly_data = val_data[val_data["label"] == 1]

                train_normal_data, val_normal_data = train_test_split_01(
                    normal_data, 
                    test_size=num_val_normal, 
                    random_state=self.random_state + fold_idx
                )

                if self.type == "train":
                    data = train_normal_data
                elif self.type == "val":
                    data = pd.concat([val_normal_data, anomaly_data], ignore_index=True)

        if IDS_Dataset.DEBUG:
            print(f"\n{self.name}")
            print(f"Loaded {type} data with {len(data)} samples.")
            print(f"Data shape: {data.shape}")
            print(f"num_anomalies: {data['label'].sum()}")
            print(f"num_normal: {data[data['label'] == 0].shape[0]}")
            print(f"anomaly ratio: {data['label'].sum() / len(data)}")

        # separate data, target (0: normal, 1: anomaly) and attack categories
        self.x = data.drop(columns=["label", "attack_cat"]).astype("float32")
        self.y = data["label"].astype("int64")
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
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.attack_cat[idx]