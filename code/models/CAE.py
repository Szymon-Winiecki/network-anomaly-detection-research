from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

import torch.nn.functional as F
import torch

import torchmetrics.functional as tmf

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from IADModel import IADModel
from AEBase import AEBase

class CAE(AEBase, IADModel):
    """ Clustering-based Deep Autoencoder (CAE)

    Implementation of model proposed in a paper:
    Nguyen, Van & Viet Hung, Nguyen & Le-Khac, Nhien-An & Cao, Van Loi. (2020).
    Clustering-Based Deep Autoencoders for Network Anomaly Detection. 10.1007/978-
    3-030-63924-2_17.
    """

    def __init__(self, 
                 input_size : int, 
                 hidden_sizes : list[int], 
                 dropout : bool | float = False, 
                 batch_norm : bool = False, 
                 initial_lr : float = 1e-3, 
                 optimizer : str = "Adam",
                 optimizer_params : dict = None,
                 scheduler : str | None = None,
                 scheduler_params : dict = None,
                 num_clusters : int = 2,
                 clustering_force : float = 300,
                 centering_force : float = 1500,
                 threshold_quantile : float = 0.9) -> None:
        """
        Args:
            input_size (int): Size of the input layer
            hidden_sizes (list): List of sizes of the hidden layers of encoder, last element is the size of the bottleneck layer. Decoder will have the same hidden sizes in reverse order. 
            dropout (False | float, optional): If float, it will be used as the dropout rate. If False no dropout layers are used. Defaults to False. Float 0.0 is treated as False.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            initial_lr (float, optional): Initial learning rate. Defaults to 1e-3.
            optimizer (str, optional): Optimizer to use. Supported optimizers are: Adam, SGD, Adadelta, Adagrad, AdamW. Defaults to "Adam".
            optimizer_params (dict, optional): Additional parameters for the optimizer. Defaults to None.
            scheduler (str, optional): Learning rate scheduler to use. Supported schedulers are: LinearLR, ExponentialLR, CosineAnnealingLR, StepLR. Defaults to "LinearLR".
            scheduler_params (dict, optional): Additional parameters for the scheduler. Defaults to None.
            num_clusters (int, optional): Number of clusters. Defaults to 2.
            clustering_force (float, optional): The force of the clustering loss. Defaults to 300.
            centering_force (float, optional): The force of the centering loss. Defaults to 1500.
            threshold_quantile (float, optional): The quantile to use for the anomaly detection threshold calculation.

        """

        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            batch_norm=batch_norm,
            initial_lr=initial_lr,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            num_clusters=num_clusters,
            clustering_force=clustering_force,
            centering_force=centering_force,
            threshold_quantile=threshold_quantile
        )
        IADModel.__init__(self, name="CAE(M-CEN)")

        self.clustering_force = clustering_force
        self.centering_force = centering_force

        self.num_clusters = num_clusters
        self.threshold_quantile = threshold_quantile


        self.train_step_latents = []
        self.train_step_re_losss = []
        self.train_step_clusteing_losses = []
        self.train_step_centering_losses = []
        self.train_step_losses = []

        # Store the preds and labels to compute metrics at the end of the epoch
        self.validation_step_scores = []
        self.validation_step_preds = []
        self.validation_step_labels = []
        self.validation_step_clusters = []
        self.validation_step_losses = []
        self.validation_step_latents = []

        self.test_step_socores = []
        self.test_step_preds = []
        self.test_step_labels = []
        self.test_step_clusters = []
        self.test_step_losses = []

    def on_fit_start(self):

        self.cluster_centers = torch.zeros((self.hparams.num_clusters, self.hidden_sizes[-1]), device=self.device)
        self.choose_centers = True # whether to choose random cluster centers in the next (first) epoch

        self.thresholds = torch.zeros((self.hparams.num_clusters), device=self.device)

        self.logger.log_hyperparams({
            "model": self.__class__.__name__,
            "max_epochs": self.trainer.max_epochs,
            "optimizer": self.trainer.optimizers[0].__class__.__name__,
        })

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        x_latent = self.encoder(x)
        x_recon = self.decoder(x_latent)

        if self.choose_centers:
            # in the first batch of the first epoch choose cluseter centers as a random samples from the latent space
            self.cluster_centers = x_latent[torch.randperm(x_latent.shape[0])[:self.num_clusters]].detach()
            self.choose_centers = False

        distances_to_the_closest_center, clusters = self._calc_clusters(x_latent)

        loss, re_loss, clustering_loss, centering_loss = self._calc_loss(x, x_recon, distances_to_the_closest_center)

        self.train_step_latents.append(x_latent.detach().clone())
        self.train_step_losses.append(loss.detach().clone())
        self.train_step_re_losss.append(re_loss.detach().clone())
        self.train_step_clusteing_losses.append(clustering_loss.detach().clone())
        self.train_step_centering_losses.append(centering_loss.detach().clone())

        return loss.mean()
    
    def on_train_epoch_end(self):

        latents = torch.cat(self.train_step_latents)
        losses = torch.tensor(self.train_step_losses)

        re_losses = torch.cat(self.train_step_re_losss)
        clustering_losses = torch.cat(self.train_step_clusteing_losses)
        centering_losses = torch.cat(self.train_step_centering_losses)

        # assign each sample to the closest cluster center
        distances_to_centers = torch.zeros((latents.shape[0], self.num_clusters), device=self.device)
        for i in range(self.num_clusters):
            distances_to_centers[:, i] = torch.linalg.vector_norm(latents - self.cluster_centers[i], ord=2, dim=1)
        clusters = torch.argmin(distances_to_centers, dim=1)

        # update the cluster centers
        for i in range(self.num_clusters):
            cluster_latents = latents[clusters == i]
            if len(cluster_latents) > 0:
                self.cluster_centers[i] = cluster_latents.mean(dim=0)

        # recalculate anomaly detection thresholds
        distances_to_centers = torch.zeros((latents.shape[0], self.num_clusters), device=self.device)
        for i in range(self.num_clusters):
            distances_to_centers[:, i] = torch.linalg.vector_norm(latents - self.cluster_centers[i], ord=2, dim=1)
        clusters = torch.argmin(distances_to_centers, dim=1)
        distances_to_the_closest_center = torch.min(distances_to_centers, dim=1).values
        for i in range(self.num_clusters):
            distances = distances_to_the_closest_center[clusters == i]
            if distances.shape[0] > 0:
                self.thresholds[i] = torch.quantile(distances, self.threshold_quantile)
            else:
                self.thresholds[i] = 0


        self.log("train_loss", losses.mean())
        self.log("re_loss", re_losses.mean())
        self.log("clustering_loss", clustering_losses.mean())
        self.log("centering_loss", centering_losses.mean())

        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])

        self.train_step_latents.clear()
        self.train_step_losses.clear()
        self.train_step_re_losss.clear()
        self.train_step_clusteing_losses.clear()
        self.train_step_centering_losses.clear()

    def validation_step(self, batch, batch_idx):
        
        x, y, attack_cat = batch

        x_latent = self.encoder(x)

        distances_to_the_closest_center, clusters = self._calc_clusters(x_latent)

        anomaly_scores = distances_to_the_closest_center
        preds = self._calc_predictions(anomaly_scores, clusters)

        self.validation_step_scores.append(anomaly_scores)
        self.validation_step_clusters.append(clusters)
        self.validation_step_preds.append(preds)
        self.validation_step_labels.append(y)
        self.validation_step_latents.append(x_latent)
 
        return 0
    
    def on_validation_epoch_end(self):
        
        anomaly_scores = torch.cat(self.validation_step_scores)
        clusters = torch.cat(self.validation_step_clusters)
        preds = torch.cat(self.validation_step_preds)
        labels = torch.cat(self.validation_step_labels)

        accuracy = tmf.accuracy(preds, labels, task="binary")
        precision = tmf.precision(preds, labels, task="binary")
        recall = tmf.recall(preds, labels, task="binary")
        specificity = tmf.specificity(preds, labels, task="binary")
        f1 = tmf.f1_score(preds, labels, task="binary")
        mcc = tmf.matthews_corrcoef(preds, labels, task="binary")

        aurocs = torch.zeros((self.num_clusters), device=self.device)
        average_precisions = torch.zeros((self.num_clusters), device=self.device)
        cluster_sizes = torch.zeros((self.num_clusters), device=self.device)
        for i in range(self.num_clusters):
            cluster_labels = labels[clusters == i]
            cluster_anomaly_scores = anomaly_scores[clusters == i]
            cluster_sizes[i] = cluster_labels.shape[0]
            if len(cluster_labels) > 0:
                aurocs[i] = tmf.auroc(F.tanh(cluster_anomaly_scores), cluster_labels, task="binary")
                average_precisions[i] = tmf.average_precision(F.tanh(cluster_anomaly_scores), cluster_labels, task="binary")
            else:
                aurocs[i] = torch.tensor(0.0, device=self.device)
                average_precisions[i] = torch.tensor(0.0, device=self.device)
        auroc = (aurocs * cluster_sizes / cluster_sizes.sum()).sum()
        average_precision = (average_precisions * cluster_sizes / cluster_sizes.sum()).sum()

        self.log("val_accuracy", accuracy)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_specificity", specificity)
        self.log("val_f1", f1)
        self.log("val_mcc", mcc)
        self.log("val_auroc", auroc)
        self.log("val_average_precision", average_precision)

        for i in range(self.num_clusters):
            self.log(f"val_auroc_cluster_{i}", aurocs[i])
            self.log(f"val_average_precision_cluster_{i}", average_precisions[i])
            self.log(f"val_cluster_size_{i}", cluster_sizes[i])

        self.log("positive_rate", preds.float().mean())


        # Reset the losses and labels for the next epoch
        self.validation_step_scores.clear()
        self.validation_step_preds.clear()
        self.validation_step_labels.clear()
        self.validation_step_clusters.clear()
        self.validation_step_latents.clear()

    def test_step(self, batch, batch_idx):
        x, y, attack_cat = batch

        x_latent = self.encoder(x)

        distances_to_the_closest_center, clusters = self._calc_clusters(x_latent)

        anomaly_scores = distances_to_the_closest_center
        preds = torch.zeros((x_latent.shape[0]), dtype=bool, device=self.device)
        for i in range(self.num_clusters):
            preds[clusters == i] = anomaly_scores[clusters == i] > self.thresholds[i]

        self.test_step_socores.append(anomaly_scores)
        self.test_step_preds.append(preds)
        self.test_step_labels.append(y)
        self.test_step_clusters.append(clusters)

        return 0

    def on_test_epoch_end(self):
        # Concatenate all losses and labels of the samples in the validation step
        anomaly_scores = torch.cat(self.test_step_socores)
        preds = torch.cat(self.test_step_preds)
        labels = torch.cat(self.test_step_labels)
        clusters = torch.cat(self.test_step_clusters)

        accuracy = tmf.accuracy(preds, labels, task="binary")
        precision = tmf.precision(preds, labels, task="binary")
        recall = tmf.recall(preds, labels, task="binary")
        specificity = tmf.specificity(preds, labels, task="binary")
        f1 = tmf.f1_score(preds, labels, task="binary")
        mcc = tmf.matthews_corrcoef(preds, labels, task="binary")

        aurocs = torch.zeros((self.num_clusters), device=self.device)
        average_precisions = torch.zeros((self.num_clusters), device=self.device)
        cluster_sizes = torch.zeros((self.num_clusters), device=self.device)
        for i in range(self.num_clusters):
            cluster_labels = labels[clusters == i]
            cluster_anomaly_scores = anomaly_scores[clusters == i]
            cluster_sizes[i] = cluster_labels.shape[0]
            if len(cluster_labels) > 0:
                aurocs[i] = tmf.auroc(F.tanh(cluster_anomaly_scores), cluster_labels, task="binary")
                average_precisions[i] = tmf.average_precision(F.tanh(cluster_anomaly_scores), cluster_labels, task="binary")
            else:
                aurocs[i] = torch.tensor(0.0, device=self.device)
                average_precisions[i] = torch.tensor(0.0, device=self.device)
        auroc = (aurocs * cluster_sizes / cluster_sizes.sum()).sum()
        average_precision = (average_precisions * cluster_sizes / cluster_sizes.sum()).sum()

        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_specificity", specificity)
        self.log("test_f1", f1)
        self.log("test_mcc", mcc)
        self.log("test_auroc", auroc)
        self.log("test_average_precision", average_precision)

        for i in range(self.num_clusters):
            self.log(f"test_auroc_cluster_{i}", aurocs[i])
            self.log(f"test_average_precision_cluster_{i}", average_precisions[i])
            self.log(f"test_cluster_size_{i}", cluster_sizes[i])

        # Reset the losses and labels for the next epoch
        self.test_step_socores.clear()
        self.test_step_preds.clear()
        self.test_step_labels.clear()
        self.test_step_clusters.clear()
    
    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        x_latent = self.encoder(x)

        distances_to_the_closest_center, clusters = self._calc_clusters(x_latent)

        anomaly_scores = distances_to_the_closest_center
        preds = self._calc_predictions(anomaly_scores, clusters)

        preds = preds.float().cpu().numpy()

        return preds

    def fit(self, 
            train_dataset, 
            val_dataset = None, 
            max_epochs = 10, 
            trainer_callbacks = None,
            log = False, 
            logger_params = {},
            random_state = None):

        if log:
            logger_params = self.default_logger_params | logger_params
            logger = MLFlowLogger(**logger_params)
        else:
            logger = None

        trainer = L.Trainer(accelerator=self.tech_params["accelerator"], max_epochs=max_epochs, callbacks=trainer_callbacks, logger=logger)

        # for some strange reason, mlflow logger always saves two copies of the checkpoint
        # First (desired) in mlartifacts folder and the second (unwanted) in the notebook's dir
        # This is a workaround to gitignore second copy and easily remove it after training 
        # see https://github.com/Lightning-AI/pytorch-lightning/issues/17904 for more info
        trainer.checkpoint_callback.dirpath = self.redundant_checkpoints_dir

        train_loader = self._get_loader(train_dataset, shuffle=True)
        val_loader = self._get_loader(val_dataset, shuffle=False) if val_dataset else None

        trainer.fit(self, train_loader, val_loader)

        metrics = {metric : value.item() for metric, value in trainer.logged_metrics.items()}

        return metrics

    def evaluate(self, test_dataset, logger_params = {}):
        
        if logger_params:
            logger_params = self.default_logger_params | logger_params
            logger = MLFlowLogger(**logger_params)
        else:
            logger = None

        trainer = L.Trainer(accelerator=self.tech_params["accelerator"], logger=logger)

        test_loader = self._get_loader(test_dataset, shuffle=False)

        metrics = trainer.test(self, test_loader)[0]

        return metrics

    def predict(self, dataset):

        self.cluster_centers = self.cluster_centers.to(self.device)
        self.thresholds = self.thresholds.to(self.device)
        
        model_mode = self.training
        self.eval()

        dataloader = self._get_loader(dataset, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                x, _, _ = batch
                x = x.to(self.device)
                x_latent = self.encoder(x)

                anomaly_scores, clusters = self._calc_clusters(x_latent)
                preds = self._calc_predictions(anomaly_scores, clusters)

                predictions.append(preds)

        predictions = torch.cat(predictions)

        self.train(mode=model_mode)

        return predictions

    def predict_raw(self, dataset):

        self.cluster_centers = self.cluster_centers.to(self.device)
        self.thresholds = self.thresholds.to(self.device)

        model_mode = self.training
        self.eval()

        dataloader = self._get_loader(dataset, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                x, _, _ = batch
                x = x.to(self.device)
                x_latent = self.encoder(x)

                anomaly_scores, _ = self._calc_clusters(x_latent)
                anomaly_scores = F.tanh(anomaly_scores)

                predictions.append(anomaly_scores)

        predictions = torch.cat(predictions)

        self.train(mode=model_mode)

        return predictions

    def save(self, path):
        raise NotImplementedError("Save method not implemented")

    @staticmethod
    def load(path):
        raise NotImplementedError("Load method not implemented")
    
    def _calc_clusters(self, x_latent : torch.Tensor) -> tuple:
        """ Calculate the clusters for the given latent representation
        Args:
            x_latent (torch.Tensor): Latent representation of the input data
        Returns:
            tuple: Distances to the closest cluster center (torch.Tensor), clusters (Torch.tensor)
        """
        distances_to_centers = torch.zeros((x_latent.shape[0], self.num_clusters), device=self.device)
        for i in range(self.num_clusters):
            distances_to_centers[:, i] = torch.linalg.vector_norm(x_latent - self.cluster_centers[i], ord=2, dim=1)
        clusters = torch.argmin(distances_to_centers, dim=1)
        distances_to_the_closest_center = torch.min(distances_to_centers, dim=1).values

        return distances_to_the_closest_center, clusters
    
    def _calc_loss(self, x : torch.Tensor, x_recon : torch.Tensor, distances_to_the_closest_center : torch.Tensor) -> tuple:
        """ Calculate the loss for the CAE model
        Args:
            x (torch.Tensor): Input tensor
            x_recon (torch.Tensor | None): Reconstructed input from the latent representation
            distances_to_the_closest_center (torch.Tensor): Distances to the closest cluster center from each sample in the latent space

        Returns:
            tuple: Total loss, reconstruction loss, clustering loss, centering loss
        """

        # CAE loss: reconstruction error + clustering loss + centering loss
        re_loss = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)
        clustering_loss = distances_to_the_closest_center ** 2
        centering_loss = torch.linalg.vector_norm(self.cluster_centers, ord=2, dim=1) ** 2
        
        loss = re_loss.mean() + self.clustering_force * clustering_loss.mean() + self.centering_force * centering_loss.mean()

        return loss, re_loss, clustering_loss, centering_loss
    
    def _calc_predictions(self, anomaly_scores : torch.Tensor, clusters : torch.Tensor) -> torch.Tensor:
        """ Calculate the predictions based on the anomaly scores and clusters
        Args:
            anomaly_scores (torch.Tensor): Anomaly scores for each sample
            clusters (torch.Tensor): A cluster each sample belongs to
        Returns:
            torch.Tensor: Predictions for each sample
        """

        preds = torch.zeros((anomaly_scores.shape[0]), dtype=bool, device=self.device)
        for i in range(self.num_clusters):
            preds[clusters == i] = anomaly_scores[clusters == i] > self.thresholds[i]

        return preds
    
    def calc_ROC(self, dataset):
        """ Calculate the ROC curve for the model on the given dataset """

        scores, labels, clusters = self.predict_raw_and_clusters(dataset)

        fprs = []
        tprs = []
        aurocs = []
        cluster_sizes = []

        for i in range(self.num_clusters):
            cluster_scores = scores[clusters == i]
            cluster_labels = labels[clusters == i]
            if len(cluster_labels) > 0:
                fpr, tpr, thresholds = tmf.roc(cluster_scores, cluster_labels, task="binary")
                auroc = tmf.auroc(cluster_scores, cluster_labels, task="binary")

            cluster_sizes.append(cluster_labels.shape[0])
            fprs.append(fpr.cpu().numpy())
            tprs.append(tpr.cpu().numpy())
            aurocs.append(auroc.item())


        return fprs, tprs, aurocs, cluster_sizes