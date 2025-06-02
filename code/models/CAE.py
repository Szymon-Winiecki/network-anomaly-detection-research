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
                 linear_lr_start_factor : float = 1.0, 
                 linear_lr_end_factor : float = 0.1, 
                 linear_lr_total_iters : int = 100,
                 optimizer : str = "Adam",
                 optimizer_params : dict = None,
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
            linear_lr_start_factor (float, optional): Start factor for the linear learning rate scheduler. Defaults to 1.
            linear_lr_end_factor (float, optional): End factor for the linear learning rate scheduler. Defaults to 0.1.
            linear_lr_total_iters (int, optional): Total iterations (num epochs) for the linear learning rate scheduler. Defaults to 100.
            optimizer (str, optional): Optimizer to use. Supported optimizers are: Adam, SGD, Adadelta, Adagrad, AdamW. Defaults to "Adam".
            optimizer_params (dict, optional): Additional parameters for the optimizer. Defaults to None.
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
            linear_lr_start_factor=linear_lr_start_factor,
            linear_lr_end_factor=linear_lr_end_factor,
            linear_lr_total_iters=linear_lr_total_iters,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            num_clusters=num_clusters,
            clustering_force=clustering_force,
            centering_force=centering_force,
            threshold_quantile=threshold_quantile
        )

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

        self.test_step_socores = []
        self.test_step_preds = []
        self.test_step_labels = []
        self.test_step_clusters = []

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

        distances_to_centers = torch.zeros((x_latent.shape[0], self.num_clusters), device=self.device)
        for i in range(self.num_clusters):
            distances_to_centers[:, i] = torch.linalg.vector_norm(x_latent - self.cluster_centers[i], ord=2, dim=1)

        distances_to_the_closest_center = torch.min(distances_to_centers, dim=1).values

        # CAE loss: reconstruction error + clustering loss + centering loss
        re_loss = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)
        clustering_loss = distances_to_the_closest_center ** 2
        centering_loss = torch.linalg.vector_norm(self.cluster_centers, ord=2, dim=1) ** 2

        loss = re_loss.mean() + self.clustering_force * clustering_loss.mean() + self.centering_force * centering_loss.mean()

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
        self.log("train_loss_std", losses.std())
        self.log("min_loss", losses.min())
        self.log("max_loss", losses.max())
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

        distances_to_centers = torch.zeros((x_latent.shape[0], self.num_clusters), device=self.device)
        for i in range(self.num_clusters):
            distances_to_centers[:, i] = torch.linalg.vector_norm(x_latent - self.cluster_centers[i], ord=2, dim=1)
        clusters = torch.argmin(distances_to_centers, dim=1)
        distances_to_the_closest_center = torch.min(distances_to_centers, dim=1).values

        anomaly_scores = distances_to_the_closest_center
        preds = torch.zeros((x_latent.shape[0]), dtype=bool, device=self.device)
        for i in range(self.num_clusters):
            preds[clusters == i] = anomaly_scores[clusters == i] > self.thresholds[i]

        self.validation_step_scores.append(anomaly_scores)
        self.validation_step_clusters.append(clusters)
        self.validation_step_preds.append(preds)
        self.validation_step_labels.append(y)
 
        return 0
    
    def on_validation_epoch_end(self):
        
        anomaly_scores = torch.cat(self.validation_step_scores)
        clusters = torch.cat(self.validation_step_clusters)
        preds = torch.cat(self.validation_step_preds)
        labels = torch.cat(self.validation_step_labels)

        accuracy = tmf.accuracy(preds, labels, task="binary")
        precision = tmf.precision(preds, labels, task="binary")
        recall = tmf.recall(preds, labels, task="binary")
        f1 = tmf.f1_score(preds, labels, task="binary")

        aurocs = torch.zeros((self.num_clusters), device=self.device)
        for i in range(self.num_clusters):
            cluster_labels = labels[clusters == i]
            cluster_anomaly_scores = anomaly_scores[clusters == i]
            if len(cluster_labels) > 0:
                aurocs[i] = tmf.auroc(cluster_anomaly_scores, cluster_labels, task="binary")
            else:
                aurocs[i] = torch.tensor(0.0, device=self.device)
        auroc = aurocs.mean()

        self.log("val_accuracy", accuracy)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)
        self.log("val_auroc", auroc)

        for i in range(self.num_clusters):
            self.log(f"val_auroc_cluster_{i}", aurocs[i])

        self.log("positive_rate", preds.float().mean())


        # Reset the losses and labels for the next epoch
        self.validation_step_scores.clear()
        self.validation_step_preds.clear()
        self.validation_step_labels.clear()
        self.validation_step_clusters.clear()

    def test_step(self, batch, batch_idx):
        x, y, attack_cat = batch

        x_latent = self.encoder(x)

        distances_to_centers = torch.zeros((x_latent.shape[0], self.num_clusters), device=self.device)
        for i in range(self.num_clusters):
            distances_to_centers[:, i] = torch.linalg.vector_norm(x_latent - self.cluster_centers[i], ord=2, dim=1)
        clusters = torch.argmin(distances_to_centers, dim=1)
        distances_to_the_closest_center = torch.min(distances_to_centers, dim=1).values

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
        f1 = tmf.f1_score(preds, labels, task="binary")

        aurocs = torch.zeros((self.num_clusters), device=self.device)
        for i in range(self.num_clusters):
            cluster_labels = labels[clusters == i]
            cluster_anomaly_scores = anomaly_scores[clusters == i]
            if len(cluster_labels) > 0:
                aurocs[i] = tmf.auroc(cluster_anomaly_scores, cluster_labels, task="binary")
            else:
                aurocs[i] = torch.tensor(0.0, device=self.device)

        auroc = aurocs.mean()

        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_auroc", auroc)

        for i in range(self.num_clusters):
            self.log(f"test_auroc_cluster_{i}", aurocs[i])

        # Reset the losses and labels for the next epoch
        self.test_step_socores.clear()
        self.test_step_preds.clear()
        self.test_step_labels.clear()
        self.test_step_clusters.clear()
    
    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        x_latent = self.encoder(x)

        distances_to_centers = torch.zeros((x_latent.shape[0], self.num_clusters), device=self.device)
        for i in range(self.num_clusters):
            distances_to_centers[:, i] = torch.linalg.vector_norm(x_latent - self.cluster_centers[i], ord=2, dim=1)
        clusters = torch.argmin(distances_to_centers, dim=1)
        distances_to_the_closest_center = torch.min(distances_to_centers, dim=1).values

        anomaly_scores = distances_to_the_closest_center
        preds = torch.zeros((x_latent.shape[0]), dtype=bool, device=self.device)
        for i in range(self.num_clusters):
            preds[clusters == i] = anomaly_scores[clusters == i] > self.thresholds[i]

        preds = preds.float().cpu().numpy()

        return preds

    def fit(self, 
            train_dataset, 
            val_dataset = None, 
            max_epochs=10, 
            log=False, 
            logger_params={},
            random_state=None):

        if log:
            logger_params = self.default_logger_params | logger_params
            logger = MLFlowLogger(**logger_params)
        else:
            logger = None

        trainer = L.Trainer(accelerator=self.tech_params["accelerator"], max_epochs=max_epochs, logger=logger)

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
        
        if self.occ_algorithm == "centroid":
            self.cen_dst_threshold = self.cen_dst_threshold.to(self.device)
        
        model_mode = self.training
        self.eval()

        dataloader = self._get_loader(dataset, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                x, _, _ = batch
                x = x.to(self.device)
                x_latent = self.encoder(x)

                preds = self._calc_predictions(x_latent)

                predictions.append(preds)

        predictions = torch.cat(predictions)

        self.train(mode=model_mode)

        return predictions

    def predict_raw(self, dataset):

        model_mode = self.training
        self.eval()

        dataloader = self._get_loader(dataset, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                x, _, _ = batch
                x = x.to(self.device)
                x_latent = self.encoder(x)

                preds = self._calc_anomaly_score(x_latent)

                predictions.append(preds)

        predictions = torch.cat(predictions)

        self.train(mode=model_mode)

        return predictions

    def save(self, path):
        raise NotImplementedError("Save method not implemented")

    @staticmethod
    def load(path):
        raise NotImplementedError("Load method not implemented")

    def plot_latent_space(self, dataset, save_path=None):
        """ Plot the latent space of the model

        Args:
            dataset (torch.utils.data.Dataset): Dataset to plot
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        dataloader = self._get_loader(dataset, shuffle=False)

        latents = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                x, y, _ = batch
                x = x.to(self.device)
                x_latent = self.encoder(x)

                latents.append(x_latent)
                labels.append(y)

        latents = torch.cat(latents).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

        pca = PCA(n_components=2)
        pca.fit(latents)
        x = pca.transform(latents)

        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', alpha=0.5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title("Latent Space")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()