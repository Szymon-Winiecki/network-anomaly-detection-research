from pathlib import Path

from sklearn.cluster import Birch

from torch.utils.data import TensorDataset
import torch

import torcheval.metrics.functional as tmf

import mlflow

import numpy as np

from IADModel import IADModel

class BAE(IADModel):
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
                 birch_fit_sample_size : int,
                 birch_fit_quantile : float,
                 base_model : IADModel,
                 **base_model_kwargs):
        """
        Initialize the BAE model.

        Args:
            birch_threshold (float): The threshold for the Birch clustering.
            birch_branching_factor (int): The branching factor for the Birch clustering.
            birch_n_clusters (int): The number of clusters for the Birch clustering.
            birch_fit_sample_size (int): Number of samples to fit the BIRCH model. Samples are randomly selected from the data.
            birch_fit_quantile (float): Quantile to determine the outlier threshold for samples. 
                BIRCH is trained on samples with a magnitude lower than the threshold, to prevent outliers from affecting the clustering.
            base_model (IADModel class): Autoencoder model class to be trained on each cluster.
            **base_model_kwargs: Keyword arguments for the base model.
        """

        IADModel.__init__(self)

        # store hyperparameters to allow saving and loading model 
        self.hparams = {
            "birch_threshold": birch_threshold,
            "birch_branching_factor": birch_branching_factor,
            "birch_n_clusters": birch_n_clusters,
            "birch_fit_sample_size": birch_fit_sample_size,
            "birch_fit_quantile": birch_fit_quantile,
            "base_model": base_model,
            "base_model_kwargs": base_model_kwargs
        }

        self.birch = Birch(threshold=birch_threshold, branching_factor=birch_branching_factor, n_clusters=birch_n_clusters)

        self.base_model = base_model
        self.base_model_kwargs = base_model_kwargs

        self.birch_fit_sample_size = birch_fit_sample_size
        self.birch_fit_quantile = birch_fit_quantile


    def fit(self, 
            train_dataset, 
            val_dataset = None,
            max_epochs = 10,
            log = False,
            logger_params = {},
            random_state = None):
        
        logger_params = self.default_logger_params | logger_params

        if log:
            mlflow.set_tracking_uri(logger_params["tracking_uri"])
            mlflow.set_experiment(logger_params["experiment_name"])
            mlflow_run = mlflow.start_run(run_name=logger_params["run_name"])
            mlflow.log_params(self.hparams | {"base_model" : self.base_model.__name__}, run_id=mlflow_run.info.run_id)
            mlflow.log_params(logger_params["tags"], run_id=mlflow_run.info.run_id)
            mlflow.log_params({"model": self.__class__.__name__}, run_id=mlflow_run.info.run_id)
            mlflow.log_params({"max_epochs": max_epochs}, run_id=mlflow_run.info.run_id)

        # select a sample of the data to fit the birch model
        # birch is trained on a sample of the data to speed up the process
        # and to avoid memory issues
        birch_fit_sample_size = min(self.birch_fit_sample_size, len(train_dataset))
        rng = np.random.default_rng(random_state)
        birch_fit_sample_indices = rng.choice(len(train_dataset), size=birch_fit_sample_size, replace=False)
        birch_fit_samples = train_dataset.x[birch_fit_sample_indices].cpu().numpy()

        fit_samples_magnitude = np.linalg.norm(birch_fit_samples, axis=1)
        outlier_threshold = np.quantile(fit_samples_magnitude, self.birch_fit_quantile)

        self.birch.fit(birch_fit_samples[fit_samples_magnitude < outlier_threshold])

        # get the cluster label for each sample
        train_clusters = self.birch.predict(train_dataset.x.cpu().numpy())
        val_clusters = self.birch.predict(val_dataset.x.cpu().numpy()) if val_dataset is not None else None

        # create separate datasets for each cluster
        cluster_train_datasets = []
        cluster_val_datasets = []
        for i in range(self.birch.n_clusters):
            filtered_dataset = self._get_filtered_dataset(train_dataset,
                                                           filter = train_clusters == i)
            cluster_train_datasets.append(filtered_dataset)

            if val_clusters is not None:
                filtered_dataset = self._get_filtered_dataset(val_dataset,
                                                               filter = val_clusters == i)
                cluster_val_datasets.append(filtered_dataset)
            else:
                cluster_val_datasets.append(None)

        # fit autoencoder model on each cluster
        self.autoencoders = []
        for i, (cluster_train_dataset, cluster_val_dataset) in enumerate(zip(cluster_train_datasets, cluster_val_datasets)):
            cluster_model = self.base_model(**self.base_model_kwargs)

            cluster_model.set_tech_params(**self.tech_params)

            cluster_model_logger_params = logger_params.copy()
            cluster_model_logger_params["experiment_name"] = f"{cluster_model_logger_params['experiment_name']} submodels"
            cluster_model_logger_params["run_name"] = f"{cluster_model_logger_params['run_name']} cluster_{i}"

            cluster_model.fit(
                cluster_train_dataset,
                val_dataset = cluster_val_dataset,
                max_epochs = max_epochs,
                log = log,
                logger_params = cluster_model_logger_params,
                random_state = random_state
            )

            self.autoencoders.append(cluster_model)
        

        if log:
            metrics = self.evaluate(val_dataset)

            mlflow.log_metrics(metrics, run_id=mlflow_run.info.run_id)
            mlflow.end_run()
        

    def evaluate(self, dataset, logger_params = {}):

        preds = self.predict(dataset)
        anomaly_scores = self.predict_raw(dataset)
        labels = dataset.y.clone().detach()

        accuracy = tmf.binary_accuracy(preds, labels)
        precision = tmf.binary_precision(preds, labels)
        recall = tmf.binary_recall(preds, labels)
        f1 = tmf.binary_f1_score(preds, labels)
        auroc = tmf.binary_auroc(anomaly_scores, labels)


        metrics = {
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_auroc": auroc
        }

        return metrics

    def predict(self, dataset):

        # get the cluster labels for each sample
        clusters = self.birch.predict(dataset.x.cpu().numpy())

        # make predictions for all samples with each autoencoder
        independent_predictions = []
        for model in self.autoencoders:
            predictions = model.predict(dataset)
            independent_predictions.append(predictions)

        # for each sample select the prediction of the AE of corresponding cluster
        predictions = torch.stack(independent_predictions, dim=1)
        clusters = torch.tensor(clusters, device=predictions.device)
        predictions = predictions[torch.arange(predictions.size(0)), clusters]

        return predictions
    
    def predict_raw(self, dataset):

        # get the cluster labels for each sample
        clusters = self.birch.predict(dataset.x.cpu().numpy())

        # make predictions for all samples with each autoencoder
        independent_predictions = []
        for model in self.autoencoders:
            predictions = model.predict_raw(dataset)
            independent_predictions.append(predictions)

        # for each sample select the prediction of the AE of corresponding cluster
        predictions = torch.stack(independent_predictions, dim=1)
        clusters = torch.tensor(clusters, device=predictions.device)
        predictions = predictions[torch.arange(predictions.size(0)), clusters]

        return predictions
    
    def save(self, path = None):
        
        if path is None:
            path = self._gen_default_checkpoint_path()

        if not isinstance(path, Path):
            path = Path(path)

        cluster_models_paths = []
        model_dir = path.parent
        filename_stem = path.stem
        for i, model in enumerate(self.autoencoders):
            cluster_models_paths.append(model_dir / f"{filename_stem}_cluster_{i}.pt")
            model.save(cluster_models_paths[-1])

        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "birch": self.birch,
            "cluster_models_paths": cluster_models_paths,
            "hyper_parameters": self.hparams,
        }, path)

        return path
    
    @staticmethod
    def load(path):
        
        checkpoint = torch.load(path)

        model =  BAE(**checkpoint["hyper_parameters"])

        model.birch = checkpoint["birch"]

        model.autoencoders = []
        for cluster_model_path in checkpoint["cluster_models_paths"]:
            model.autoencoders.append(model.base_model.load(cluster_model_path))

        return model
    
    def _get_filtered_dataset(self, 
                             dataset : torch.utils.data.Dataset, 
                             filter : torch.Tensor):
        """ 
        Create dataset with samples that match the filter.

        Args:
            dataset (torch.utils.data.Dataset): The input dataset.
            filter (torch.Tensor): A boolean tensor to filter the dataset.

        Returns:
            filtered_datset (Dataset): Dataset containing only the filtered samples.
        """
        x, y, attack_cat = dataset[filter]
        filtered_dataset = TensorDataset(x, y, attack_cat)

        return filtered_dataset
