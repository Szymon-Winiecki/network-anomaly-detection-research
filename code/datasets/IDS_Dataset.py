import sys
from pathlib import Path

from torch import tensor
from torch.utils.data import Dataset

from sklearn.pipeline import Pipeline

this_directory = Path(__file__).parent.resolve()
sys.path.append(str(this_directory / '../utils'))
from dataset_utils import load_preprocessed_data, load_label_encoder

class IDS_Dataset(Dataset):

    # debug flag to print dataset statistics
    DEBUG = True

    def __init__(self, 
                 root_dir: str | Path, 
                 type: str = "train", 
                 name: str = "undefined",
                 transformer : Pipeline | None = None,
                 random_state: int = 42):
        """IDS (intrusion detection system) dataset class.

        Args:
            root_dir (str | Path): Path to the directory with the processed datasets (root dir of test, train and val sets).
            type (str): Type of the dataset. Can be "train", "val" or "test".
            name (str): Name of the dataset.
            transformer (Pipeline, optional): Optional transformer to be applied on a sample.
            random_state (int, optional): Random state for reproducibility.
        """
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)

        if type not in ["train", "val", "test"]:
            raise ValueError(f"Invalid type: {type}. Expected 'train', 'val' or 'test'.")

        self.dir = root_dir / type
        self.type = type

        self.name = name

        self.random_state = random_state

        # Load label encoder
        self.label_encoder = load_label_encoder(self.dir)

        # Load preprocessed data
        data = load_preprocessed_data(self.dir)

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