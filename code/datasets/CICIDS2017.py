import sys
from pathlib import Path

import pandas as pd

from torch import tensor
from torch.utils.data import Dataset

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

this_directory = Path(__file__).parent.resolve()
sys.path.append(str(this_directory / '../utils'))
from dataset_utils import load_preprocessed_data, load_label_encoder

class CICIDS2017Dataset(Dataset):

    # debug flag to print dataset statistics
    DEBUG = True

    # number of anomaly an normal records to be placed in the validation set
    # identical proportions to the ones in the test set
    VAL_NUM_ANOMALIES = 278823
    VAL_NUM_NORMAL = 278823

    def __init__(self, 
                 root_dir: str | Path, 
                 type: str = "train", 
                 transformer : Pipeline | None = None,
                 random_state: int = 42):
        """CICIDS2017 dataset class.

        Args:
            root_dir (str | Path): Path to the directory with the processed datasets (root dir of test and train sets).
            type (str): Type of the dataset. Can be "train", "val" or "test".
            transformer (Pipeline, optional): Optional transformer to be applied on a sample.
            random_state (int, optional): Random state for reproducibility.
        """
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)

        self.dir = root_dir / ("test" if type == "test" else "train")
        self.type = type

        # Load label encoder
        self.label_encoder = load_label_encoder(self.dir)

        # Load preprocessed data
        data = load_preprocessed_data(self.dir)


        if type == "train":
            # remove attack records, as we need to train on normal data only
            data = data[data["label"] == 0]

            # separete validation records from training records
            data, val = train_test_split(data, test_size=CICIDS2017Dataset.VAL_NUM_NORMAL, random_state=random_state)
        if type == "val":
            
            anomalies = data[data["label"] == 1]
            normal = data[data["label"] == 0]

            # select appriopriate number of anomaly records
            anomalies = anomalies.sample(n=CICIDS2017Dataset.VAL_NUM_ANOMALIES, random_state=random_state)

            # separate normal validation records from training records
            train, normal = train_test_split(normal, test_size=CICIDS2017Dataset.VAL_NUM_NORMAL, random_state=random_state)

            # concatenate normal and anomaly records
            data = pd.concat([normal, anomalies], axis=0)
            
        elif type == "test":
            # there is no need to split test data
            pass

        if CICIDS2017Dataset.DEBUG:
            print(f"\nLoaded {type} data with {len(data)} samples.")
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