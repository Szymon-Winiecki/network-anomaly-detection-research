from sklearn.cluster import Birch

from torch.utils.data import DataLoader, TensorDataset
import torch

import torcheval.metrics.functional as tmf

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

import numpy as np

class BAE:
    """
    BAE (BIRCH Auto Encoder) implementation.

    Based on:
        Wang, Dongqi & Nie, Mingshuo & Chen, Dongming. (2023). BAE: Anomaly
        Detection Algorithm Based on Clustering and Autoencoder. Mathematics. 
        11. 3398. 10.3390/math11153398.
    """

    def __init__(self, 
                 birch_threshold : float, 
                 birch_branching_factor : int, 
                 birch_n_clusters : int, 
                 base_model : L.LightningModule, 
                 *base_model_args, 
                 **base_model_kwargs):
        """
        Initialize the BAE model.

        Args:
            birch_threshold (float): The threshold for the Birch clustering.
            birch_branching_factor (int): The branching factor for the Birch clustering.
            birch_n_clusters (int): The number of clusters for the Birch clustering.
            base_model (LightningModule): Autoencoder model class to be trained on each cluster.
            *base_model_args: Positional arguments for the base model.
            **base_model_kwargs: Keyword arguments for the base model.
        """

        self.birch = Birch(threshold=birch_threshold, branching_factor=birch_branching_factor, n_clusters=birch_n_clusters)

        self.base_model = base_model
        self.base_model_args = base_model_args
        self.base_model_kwargs = base_model_kwargs

        self.set_tech_params()


    def set_tech_params(self, 
                        accelerator = "cpu", 
                        batch_size : int = 1024, 
                        num_workers : int = 1, 
                        persistent_workers : bool = False):
        """
        Set the technical parameters for using the model.

        Args:
            accelerator (str): Accelerator to be used (see Lightning docs for availible options).
            batch_size (int): Batch size for the DataLoaders.
            num_workers (int): Number of workers for the DataLoader.
            persistent_workers (bool): Persistent workers flag for the DataLoader.
        """

        self.tech_params = {
            "accelerator": accelerator,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "persistent_workers": persistent_workers
        }

    def fit(self, 
            data : torch.utils.data.Dataset, 
            birch_fit_sample_size : int,
            experiment_name : str, 
            run_name : str, 
            tracking_uri : str = "http://127.0.0.1:8080", 
            max_epochs : int = 10):
        """
        Fit the BAE model to the data.

        Args:
            data (torch.utils.data.Dataset): Input data to train the model.
            birch_fit_sample_size (int): Number of samples to fit the BIRCH model. Samples are randomly selected from the data.
            experiment_name (str): The name of the experiment.
            run_name (str): The name of the run.
            tracking_uri (str): Tracking URI for MLFlow.
        """

        # select a sample of the data to fit the birch model
        # birch is trained on a sample of the data to speed up the process
        # and to avoid memory issues
        birch_fit_sample_size = min(birch_fit_sample_size, len(data))
        rng = np.random.default_rng()
        birch_fit_sample_indices = rng.choice(len(data), size=birch_fit_sample_size, replace=False)
        birch_fit_samples = data.x[birch_fit_sample_indices].cpu().numpy()

        # fir birch model with a sample of the data to speed up the clustering
        self.birch.fit(birch_fit_samples)

        # get the cluster label for each sample
        clusters = self.birch.predict(data.x.cpu().numpy())

        # create separate datasets for each cluster
        cluster_dataloaders = []
        for i in range(self.birch.n_clusters):
            cluster_dataloader = self._get_filtered_loader(data,
                                                           filter = clusters == i,
                                                           shuffle = True)

            cluster_dataloaders.append(cluster_dataloader)

        # fit autoencoder model on each cluster
        self.autoencoders = []
        for i, dataloader in enumerate(cluster_dataloaders):
            model = self.base_model(*self.base_model_args, **self.base_model_kwargs)

            logger = MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                run_name=f"{run_name}_AE{i}",
                log_model=False,
            )

            trainer = L.Trainer(accelerator=self.tech_params["accelerator"], max_epochs=max_epochs, logger=logger)
            trainer.fit(model, dataloader)
            self.autoencoders.append(model)

    def predict(self, data : torch.utils.data.Dataset):
        """
        Predict the labels for the input data.

        Args:
            data (torch.utils.data.Dataset): dataset to predict.

        Returns:
            torch.Tensor: The predicted labels.
        """

        # get the cluster labels for each sample
        clusters = self.birch.predict(data.x.cpu().numpy())

        dataloader = self._get_loader(data, shuffle=False)

        # make predictions for all samples with each autoencoder
        independent_predictions = []
        for model in self.autoencoders:
            trainer = L.Trainer(accelerator=self.tech_params["accelerator"], logger=False)
            predictions = trainer.predict(model, dataloader)
            predictions = torch.cat(predictions, dim=0)

            independent_predictions.append(predictions)

        # for each sample select the prediction of the AE of corresponding cluster
        predictions = torch.stack(independent_predictions, dim=1)
        clusters = torch.tensor(clusters, device=predictions.device)
        predictions = predictions[torch.arange(predictions.size(0)), clusters]


        return predictions
    
    def evaluate(self, data : torch.utils.data.Dataset):
        """
        Evaluate the model on the input data.

        Metrics:
            - accuracy
            - precision
            - recall
            - f1 score

        Args:
            data (torch.utils.data.Dataset): Dataset to evaluate.

        Returns:
            dict: The evaluation metrics.
        """

        preds = self.predict(data)
        labels = data.y.clone().detach()

        accuracy = tmf.binary_accuracy(preds, labels)
        precision = tmf.binary_precision(preds, labels)
        recall = tmf.binary_recall(preds, labels)
        f1 = tmf.binary_f1_score(preds, labels)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        return metrics


    
    def _get_filtered_loader(self, 
                             dataset : torch.utils.data.Dataset, 
                             filter : torch.Tensor, 
                             shuffle : bool = False):
        """ 
        Create dataloader with samples that satisfy the filter and complies with the technical parameters.

        Args:
            dataset (torch.utils.data.Dataset): The input dataset.
            filter (torch.Tensor): A boolean tensor to filter the dataset.
            shuffle (bool): DataLoader shuffle flag.

        Returns:
            DataLoader: DataLoader containing only the filtered samples.
        """
        x, y, attack_cat = dataset[filter]
        filtered_dataset = TensorDataset(x, y, attack_cat)
        filtered_dataloader = self._get_loader(filtered_dataset, shuffle=shuffle)

        return filtered_dataloader
    
    def _get_loader(self, 
                    dataset : torch.utils.data.Dataset, 
                    shuffle : bool = False):
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
