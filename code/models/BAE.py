from sklearn.cluster import Birch

from torch.utils.data import DataLoader, TensorDataset
import torch

import torcheval.metrics.functional as tmf

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

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



    def fit(self, 
            data : torch.utils.data.Dataset, 
            experiment_name : str, 
            run_name : str, 
            tracking_uri : str = "http://127.0.0.1:8080", 
            accelerator : str = "cpu", 
            max_epochs : int = 10, 
            batch_size : int = 1024):
        """
        Fit the BAE model to the data.

        Args:
            data (torch.utils.data.Dataset): Input data to train the model.
            experiment_name (str): The name of the experiment.
            run_name (str): The name of the run.
            tracking_uri (str): Tracking URI for MLFlow.
            accelerator (str): Accelerator to be used (see Lightning docs for availible options).
            max_epochs (int): Maximum number of epochs for training.
            batch_size (int): Batch size for training dataloaders.
        """

        # fit birch model and get the cluster labels for each sample
        clusters = self.birch.fit_predict(data.x.cpu().numpy())

        # create separate datasets for each cluster
        cluster_dataloaders = []
        for i in range(self.birch.n_clusters):
            cluster_dataloader = self._get_filtered_loader(data,
                                                           filter = clusters == i,
                                                           batch_size = batch_size,
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

            trainer = L.Trainer(accelerator=accelerator, max_epochs=max_epochs, logger=logger)
            trainer.fit(model, dataloader)
            self.autoencoders.append(model)

    def predict(self, 
                data : torch.utils.data.Dataset, 
                batch_size : int = 1024, 
                accelerator : str = "cpu"):
        """
        Predict the labels for the input data.

        Args:
            data (torch.utils.data.Dataset): dataset to predict.
            batch_size (int): Batch size for the DataLoader.
            accelerator (str): Accelerator to be used (see Lightning docs for availible options).

        Returns:
            torch.Tensor: The predicted labels.
        """

        # get the cluster labels for each sample
        clusters = self.birch.predict(data.x.cpu().numpy())

        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

        # make predictions for all samples with each autoencoder
        independent_predictions = []
        for model in self.autoencoders:
            trainer = L.Trainer(accelerator=accelerator, logger=False)
            predictions = trainer.predict(model, dataloader)
            predictions = torch.cat(predictions, dim=0)

            independent_predictions.append(predictions)

        # for each sample select the prediction of the AE of corresponding cluster
        predictions = torch.stack(independent_predictions, dim=1)
        clusters = torch.tensor(clusters, device=predictions.device)
        predictions = predictions[torch.arange(predictions.size(0)), clusters]


        return predictions
    
    def evaluate(self, 
                 data : torch.utils.data.Dataset, 
                 batch_size : int = 1024, 
                 accelerator : str = "cpu"):
        """
        Evaluate the model on the input data.

        Metrics:
            - accuracy
            - precision
            - recall
            - f1 score

        Args:
            data (torch.utils.data.Dataset): Dataset to evaluate.
            batch_size (int): Batch size for the DataLoader.
            accelerator (str): Accelerator to be used (see Lightning docs for availible options).

        Returns:
            dict: The evaluation metrics.
        """

        preds = self.predict(data, batch_size=batch_size, accelerator=accelerator)
        labels = torch.tensor(data.y, device=preds.device)

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
                             batch_size : int = 1024, 
                             shuffle : bool = False):
        """ 
        Create dataloader with samples that satisfy the filter.

        Args:
            dataset (torch.utils.data.Dataset): The input dataset.
            filter (torch.Tensor): A boolean tensor to filter the dataset.
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): DataLoader shuffle flag.

        Returns:
            DataLoader: DataLoader containing only the filtered samples.
        """
        x, y, attack_cat = dataset[filter]
        filtered_dataset = TensorDataset(x, y, attack_cat)
        filtered_dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=11, persistent_workers=True)

        return filtered_dataloader
