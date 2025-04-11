import sys
from pathlib import Path
import random

import pandas as pd

from torch import tensor
from torch.utils.data import Dataset

from sklearn.pipeline import Pipeline

this_directory = Path(__file__).parent.resolve()
sys.path.append(str(this_directory / '../utils'))
from dataset_utils import load_preprocessed_data, load_label_encoder

class KDDCUP99Dataset(Dataset):
    def __init__(self, root_dir: str | Path, train: bool = True, transformer : Pipeline | None = None):
        """KDDCUP99 dataset class.

        Args:
            root_dir (str | Path): Path to the directory with the processed datasets (root dir of test and train sets).
            train (bool): If True, load training data. If False, load test data.
            transformer (Pipeline, optional): Optional transformer to be applied on a sample.
        """
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)

        self.dir = root_dir / ("train" if train else "test")
        self.train = train

        # Load label encoder
        self.label_encoder = load_label_encoder(self.dir)

        # Load preprocessed data
        data = load_preprocessed_data(self.dir)
        

        # separate data, target (0: normal, 1: anomaly) and attack categories
        self.x = data.drop(columns=["label", "attack_cat"]).astype("float32")
        self.y = data["label"].astype("int64")
        self.attack_cat = data["attack_cat"].astype("int64")

        # apply data transformation pipeline
        if transformer is not None and train:
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