from pathlib import Path

import copy

from sklearn.cluster import Birch

from torch.utils.data import TensorDataset
import torch

import torchmetrics.functional as tmf

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

        IADModel.__init__(self, name = "BAE")

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

    
    def revert_threshold(self, revert_data):

        for ae in self.autoencoders:
            ae.revert_threshold(revert_data.pop(0))
    
    def set_threshold_quantile(self, train_dataset, quantile):

        revert_data = []

        train_clusters = self.birch.predict(train_dataset.x.cpu().numpy())

        # create separate datasets for each cluster
        cluster_train_datasets = []
        for i in range(self.birch.n_clusters):
            filtered_dataset = self._get_filtered_dataset(train_dataset,
                                                           filter = train_clusters == i)
            cluster_train_datasets.append(filtered_dataset)

        for ae, dataset in zip(self.autoencoders, cluster_train_datasets):
            rd = ae.set_threshold_quantile(dataset, quantile)
            revert_data.append(rd)

        return revert_data

    def test_threshold_quantile(self, train_dataset, val_dataset, quantile):

        revert_data = self.set_threshold_quantile(train_dataset, quantile)

        metrics = self.evaluate(val_dataset, logger_params=None)

        self.revert_threshold(revert_data)

        return metrics


    def fit(self, 
            train_dataset, 
            val_dataset = None,
            max_epochs = 10,
            trainer_callbacks = None,
            log = False,
            logger_params = {},
            random_state = None,
            adjust_epochs = True):
        
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

            adj_base_model_kwargs = copy.deepcopy(self.base_model_kwargs)
            adj_max_epochs = max_epochs
            if adjust_epochs:
                scaling_factor = (len(train_dataset) / self.birch.n_clusters) / len(cluster_train_dataset)
                scaling_factor = max(scaling_factor, 1.0)
                scaling_factor = min(scaling_factor, 3.0)
                adj_max_epochs = int(max_epochs * scaling_factor)

                if self.base_model_kwargs["scheduler"] == "StepLR":
                    adj_base_model_kwargs["scheduler_params"]["step_size"] = int(self.base_model_kwargs["scheduler_params"]["step_size"] * scaling_factor)

            cluster_model = self.base_model(**adj_base_model_kwargs)

            cluster_model.set_tech_params(**self.tech_params)

            cluster_model_logger_params = logger_params.copy()
            cluster_model_logger_params["experiment_name"] = f"{cluster_model_logger_params['experiment_name']} submodels"
            cluster_model_logger_params["run_name"] = f"{cluster_model_logger_params['run_name']} cluster_{i}"

            cluster_model.fit(
                cluster_train_dataset,
                val_dataset = cluster_val_dataset,
                max_epochs = adj_max_epochs,
                trainer_callbacks = trainer_callbacks,
                log = log,
                logger_params = cluster_model_logger_params,
                random_state = random_state
            )

            self.autoencoders.append(cluster_model)
        
        metrics = self.evaluate(val_dataset)

        if log:

            mlflow.log_metrics(metrics, run_id=mlflow_run.info.run_id)
            mlflow.end_run()

        return metrics
        

    def evaluate(self, dataset, logger_params = {}):

        preds = self.predict(dataset)
        anomaly_scores = self.predict_raw(dataset)
        labels = dataset.y.clone().detach()

        accuracy = tmf.accuracy(preds, labels, task="binary")
        precision = tmf.precision(preds, labels, task="binary")
        recall = tmf.recall(preds, labels, task="binary")
        specificity = tmf.specificity(preds, labels, task="binary")
        f1 = tmf.f1_score(preds, labels, task="binary")
        mcc = tmf.matthews_corrcoef(preds, labels, task="binary")
        
        clusters = self.birch.predict(dataset.x.cpu().numpy())

        aurocs = torch.zeros((self.birch.n_clusters), device=anomaly_scores.device)
        average_precisions = torch.zeros((self.birch.n_clusters), device=anomaly_scores.device)
        cluster_sizes = torch.zeros((self.birch.n_clusters), device=anomaly_scores.device)
        for i in range(self.birch.n_clusters):
            cluster_labels = labels[clusters == i]
            cluster_anomaly_scores = anomaly_scores[clusters == i]
            cluster_sizes[i] = cluster_labels.shape[0]
            if len(cluster_labels) > 0:
                aurocs[i] = tmf.auroc(cluster_anomaly_scores, cluster_labels, task="binary")
                average_precisions[i] = tmf.average_precision(cluster_anomaly_scores, cluster_labels, task="binary")
            else:
                aurocs[i] = torch.tensor(0.0, device=anomaly_scores.device)
                average_precisions[i] = torch.tensor(0.0, anomaly_scores=self.device)
        auroc = (aurocs * cluster_sizes / cluster_sizes.sum()).sum()
        average_precision = (average_precisions * cluster_sizes / cluster_sizes.sum()).sum()


        metrics = {
            "test_accuracy": accuracy.item(),
            "test_precision": precision.item(),
            "test_recall": recall.item(),
            "test_specificity": specificity.item(),
            "test_f1": f1.item(),
            "test_mcc": mcc.item(),
            "test_auroc": auroc.item(),
            "test_average_precision": average_precision.item(),
        }

        for i in range(self.birch.n_clusters):
            metrics[f"test_auroc_cluster_{i}"] = aurocs[i].item()
            metrics[f"test_average_precision_cluster_{i}"] = average_precisions[i].item()
            metrics[f"test_cluster_size_{i}"] = cluster_sizes[i].item()

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
    
    def predict_raw_and_clusters(self, dataset):

        # get the cluster labels for each sample
        clusters = self.birch.predict(dataset.x.cpu().numpy())
        labels = dataset.y.cpu().numpy()

        # make predictions for all samples with each autoencoder
        independent_predictions = []
        for model in self.autoencoders:
            predictions = model.predict_raw(dataset)
            independent_predictions.append(predictions)

        # for each sample select the prediction of the AE of corresponding cluster
        predictions = torch.stack(independent_predictions, dim=1)
        clusters = torch.tensor(clusters, device=predictions.device)
        predictions = predictions[torch.arange(predictions.size(0)), clusters]

        return predictions, labels, clusters
    
    def predict_raw(self, dataset):
        return self.predict_raw_and_clusters(dataset)[0]
    
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
    
    def calc_ROC(self, dataset):
        """ Calculate the ROC curve for the model on the given dataset """

        scores, labels, clusters = self.predict_raw_and_clusters(dataset)

        fprs = []
        tprs = []
        aurocs = []
        cluster_sizes = []

        for i in range(self.birch.n_clusters):
            cluster_scores = torch.tensor(scores[clusters == i])
            cluster_labels = torch.tensor(labels[clusters == i])
            if len(cluster_labels) > 0:
                fpr, tpr, thresholds = tmf.roc(cluster_scores, cluster_labels, task="binary")
                auroc = tmf.auroc(cluster_scores, cluster_labels, task="binary")

            cluster_sizes.append(cluster_labels.shape[0])
            fprs.append(fpr.cpu().numpy())
            tprs.append(tpr.cpu().numpy())
            aurocs.append(auroc.item())


        return fprs, tprs, aurocs, cluster_sizes
