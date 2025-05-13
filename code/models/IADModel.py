from abc import ABC, abstractmethod
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
import torch

from pathlib import Path

class IADModel(ABC):
    """
    Abstract base class for all anomaly detection models.
    """

    default_logger_params = {
        "tracking_uri" : "http://127.0.0.1:8080",
        "experiment_name" : "undefined",
        "run_name" : "undefined",
        "log_model" : False,
        "tags" : {},
    }

    redundant_checkpoints_dir = "bin_for_redundant_checkpoints"

    default_model_save_dir = Path("saved_models")

    def __init__(self):
        self.set_tech_params()

    @abstractmethod
    def fit(self, 
            train_dataset: Dataset, 
            val_dataset: Dataset = None, 
            max_epochs = 10, 
            log = False, 
            logger_params = {},
            random_state = None) -> None:
        """
        Train the model with the given data.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def evaluate(self, test_dataset: Dataset, logger_params = {}) -> dict:
        """
        Evaluate the model with the given data.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def predict(self, dataset: Dataset) -> torch.Tensor:
        """
        Predict labels using the model with for given dataset.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def predict_raw(self, dataset: Dataset) -> torch.Tensor:
        """
        Predict raw values (probabilities, anomaly scores, etc) using the model for the given dataset.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    @abstractmethod
    def save(self, path: str | Path | None) -> Path:
        """
        Save the model to the given path.

        Args:
            path (str | Path | None): The path to save the model. If None, use automaticly generated path.

        Returns:
            path (Path): The path where the model is saved.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @staticmethod
    @abstractmethod
    def load(path: str | Path):
        """
        Load the model from the given path.

        Args:
            path (str | Path): Path to saved model.

        Returns:
            model (IADModel): The loaded model.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def set_tech_params(self, 
                        accelerator = "cpu", 
                        batch_size : int = 32, 
                        num_workers : int = 1, 
                        persistent_workers : bool = False) -> None:
        """
        Set the technical parameters that best suits your machine, data and model.
        """
        self.tech_params = {
            "accelerator": accelerator,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "persistent_workers": persistent_workers
        }
    
    def _get_loader(self, 
                    dataset : torch.utils.data.Dataset, 
                    shuffle : bool = False) -> DataLoader:
        """ 
        Create dataloader that complies with the technical parameters.

        Args:
            dataset (torch.utils.data.Dataset): The input dataset.
            shuffle (bool): DataLoader shuffle flag.

        Returns:
            DataLoader: DataLoader containing only the filtered samples.
        """
        dataloader = DataLoader(dataset, 
                                batch_size = self.tech_params["batch_size"], 
                                shuffle = shuffle, 
                                num_workers = self.tech_params["num_workers"], 
                                persistent_workers = self.tech_params["persistent_workers"])

        return dataloader
    
    def _gen_default_checkpoint_path(self) -> Path:
        """
        Generate default and unique path to save the model checkpoint.
        """
        return Path(self.default_model_save_dir) / f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.pt"
    

def _init_weights_xavier_uniform(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)