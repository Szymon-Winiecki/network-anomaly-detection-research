from pathlib import Path

from sklearn.cluster import KMeans

from torch.utils.data import TensorDataset
import torch

import torchmetrics.functional as tmf

import mlflow

import numpy as np

from IADModel import IADModel

class KSAE(IADModel):
    """
    KSAE (K-means Shrink AutoEncoder) implementation.

    Based on:
        T. C. Bui, V. L. Cao, M. Hoang and Q. U. Nguyen, 
        "A Clustering-based Shrink AutoEncoder for 
        Detecting Anomalies in Intrusion Detection Systems," 
        2019 11th International Conference on Knowledge 
        and Systems Engineering (KSE), Da Nang, Vietnam, 
        2019, pp. 1-5, doi: 10.1109/KSE.2019.8919446.
    """

    def __init__(self, 
                 kmeans_n_clusters : int, 
                 kmeans_fit_sample_size : int,
                 kmeans_fit_quantile : float,
                 base_model : IADModel, 
                 **base_model_kwargs):
        """
        Initialize the KSAE model.

        Args:
            kmeans_n_clusters (int): The number of clusters for the K-Means clustering.
            kmeans_fit_sample_size (int): Number of samples to fit the K-Means model. Samples are randomly selected from the data.
            kmeans_fit_quantile (float): Quantile to determine the outlier threshold for samples. 
                K-Means is trained on samples with a magnitude lower than the threshold, to prevent outliers from affecting the clustering.
            base_model (IADModel class): Autoencoder model class to be trained on each cluster.
            **base_model_kwargs: Keyword arguments for the base model.
        """

        IADModel.__init__(self, name="KSAE")

        # store hyperparameters to allow saving and loading the model 
        self.hparams = {
            "kmeans_n_clusters": kmeans_n_clusters,
            "kmeans_fit_sample_size": kmeans_fit_sample_size,
            "kmeans_fit_quantile": kmeans_fit_quantile,
            "base_model": base_model,
            "base_model_kwargs": base_model_kwargs
        }

        self.kmeans = KMeans(n_clusters = kmeans_n_clusters)

        self.base_model = base_model
        self.base_model_kwargs = base_model_kwargs

        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_fit_sample_size = kmeans_fit_sample_size
        self.kmeans_fit_quantile = kmeans_fit_quantile


    def fit(self, 
            train_dataset, 
            val_dataset = None,
            max_epochs = 10,
            trainer_callbacks = None,
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

        # select a sample of the data to fit the kmeans model
        # kmeans is trained on a sample of the data to speed up the process
        # and to avoid memory issues
        kmeans_fit_sample_size = min(self.kmeans_fit_sample_size, len(train_dataset))
        rng = np.random.default_rng(random_state)
        kmeans_fit_sample_indices = rng.choice(len(train_dataset), size=kmeans_fit_sample_size, replace=False)
        kmeans_fit_samples = train_dataset.x[kmeans_fit_sample_indices].cpu().numpy()

        fit_samples_magnitude = np.linalg.norm(kmeans_fit_samples, axis=1)
        outlier_threshold = np.quantile(fit_samples_magnitude, self.kmeans_fit_quantile)

        self.kmeans.fit(kmeans_fit_samples[fit_samples_magnitude < outlier_threshold])

        # get the cluster label for each sample
        train_clusters = self.kmeans.predict(train_dataset.x.cpu().numpy())
        val_clusters = self.kmeans.predict(val_dataset.x.cpu().numpy()) if val_dataset is not None else None

        # create separate datasets for each cluster
        cluster_train_datasets = []
        cluster_val_datasets = []
        for i in range(self.kmeans_n_clusters):
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
        
        clusters = self.kmeans.predict(dataset.x.cpu().numpy())

        aurocs = torch.zeros((self.kmeans_n_clusters), device=anomaly_scores.device)
        average_precisions = torch.zeros((self.kmeans_n_clusters), device=anomaly_scores.device)
        cluster_sizes = torch.zeros((self.kmeans_n_clusters), device=anomaly_scores.device)
        for i in range(self.kmeans_n_clusters):
            cluster_labels = labels[clusters == i]
            cluster_anomaly_scores = anomaly_scores[clusters == i]
            cluster_sizes[i] = cluster_labels.shape[0]
            if len(cluster_labels) > 0:
                aurocs[i] = tmf.auroc(cluster_anomaly_scores, cluster_labels, task="binary")
                average_precisions[i] = tmf.average_precision(cluster_anomaly_scores, cluster_labels, task="binary")
            else:
                aurocs[i] = torch.tensor(0.0, device=anomaly_scores.device)
                average_precisions[i] = torch.tensor(0.0, device=anomaly_scores.device)
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

        for i in range(self.kmeans_n_clusters):
            metrics[f"test_auroc_cluster_{i}"] = aurocs[i].item()
            metrics[f"test_average_precision_cluster_{i}"] = average_precisions[i].item()
            metrics[f"test_cluster_size_{i}"] = cluster_sizes[i].item()

        return metrics

    def predict(self, dataset):

        # get the cluster labels for each sample
        clusters = self.kmeans.predict(dataset.x.cpu().numpy())

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
        clusters = self.kmeans.predict(dataset.x.cpu().numpy())
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
            "kmeans": self.kmeans,
            "cluster_models_paths": cluster_models_paths,
            "hyper_parameters": self.hparams,
        }, path)

        return path
    
    @staticmethod
    def load(path):
        
        checkpoint = torch.load(path)

        model = KSAE(**checkpoint["hyper_parameters"])

        model.kmeans = checkpoint["kmeans"]

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

        for i in range(self.kmeans_n_clusters):
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
