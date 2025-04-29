from abc import ABCMeta, abstractmethod

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

from torch import nn, optim
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import LinearLR

# import torcheval.metrics.functional as tmf
import torchmetrics.functional as tmf

from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

import numpy as np

import matplotlib.pyplot as plt

from IADModel import IADModel

class SAE(L.LightningModule, IADModel):
    """ Shrink Autoencoder model 

    Implementation of model proposed in a paper:
    V. L. Cao, M. Nicolau and J. McDermott, Learning Neural Representations for
    Network Anomaly Detection, in IEEE Transactions on Cybernetics, vol. 49, no. 8,
    pp. 3074-3087, Aug. 2019, doi: 10.1109/TCYB.2018.2838668
    """

    def __init__(self, 
                 input_size : int, 
                 hidden_sizes : list[int], 
                 dropout : bool | float = False, 
                 batch_norm : bool = False, 
                 lmb : float = 0.1,
                 initial_lr : float = 1e-3, 
                 linear_lr_start_factor : float = 1.0, 
                 linear_lr_end_factor : float = 0.1, 
                 linear_lr_total_iters : int = 100,
                 fit_occ_once : bool = False,
                 occ_algorithm : str = "centroid",
                 occ_fit_sample_size : int | float = 1.0,
                 *occ_args,
                 **occ_kwargs) -> None:
        """
        Args:
            input_size (int): Size of the input layer
            hidden_sizes (list): List of sizes of the hidden layers of encoder, last element is the size of the bottleneck layer. Decoder will have the same hidden sizes in reverse order. 
            dropout (False | float, optional): If float, it will be used as the dropout rate. If False no dropout layers are used. Defaults to False. Float 0.0 is treated as False.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            lmb (float, optional): Balance parameter for the SAE loss. Defaults to 0.1.
            initial_lr (float, optional): Initial learning rate. Defaults to 1e-3.
            linear_lr_start_factor (float, optional): Start factor for the linear learning rate scheduler. Defaults to 1.
            linear_lr_end_factor (float, optional): End factor for the linear learning rate scheduler. Defaults to 0.1.
            linear_lr_total_iters (int, optional): Total iterations (num epochs) for the linear learning rate scheduler. Defaults to 100.
            fit_occ_once (bool, optional): if False, the model will fit the OCC algorithm every epoch. If True, the model will fit the OCC algorithm only once, at the end of the training, it also results in no validation metrics from non-last epochs. Defaults to False.
            occ_algorithm (str, optional): OCC algorithm to use. Supported algorithms are: centroid, lof, svm. Defaults to "centroid".
            occ_fit_sample_size (int | float, optional): If int, the number of samples to use for fitting the OCC algorithm. If float, the percentage of samples to use for fitting the OCC algorithm. Defaults to 1.0 (full training dataset).
            *occ_args: Additional arguments for the OCC algorithm.
            **occ_kwargs: Additional keyword arguments for the OCC algorithm.
        """
        super().__init__()
        IADModel.__init__(self)

        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.lmb = lmb
        
        self.initial_lr = initial_lr
        self.linear_lr_start_factor = linear_lr_start_factor
        self.linear_lr_end_factor = linear_lr_end_factor
        self.linear_lr_total_iters = linear_lr_total_iters

        self.fit_occ_once = fit_occ_once
        self.occ_algorithm = occ_algorithm
        self.occ_fit_sample_size = occ_fit_sample_size

        self.encoder = self._build_encoder()

        self.decoder = self._build_decoder()

        if occ_algorithm == "centroid":
            self.classifier = GaussianMixture(n_components = 1, *occ_args, **occ_kwargs)
            self.cen_dst_threshold = 0
        elif occ_algorithm == "lof":
            self.classifier = LocalOutlierFactor(novelty=True, *occ_args, **occ_kwargs)
        elif occ_algorithm == "svm":
            self.classifier = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1, shrinking=True)
            raise NotImplementedError("SVM is not implemented yet.")
        else:
            raise ValueError(f"Unknown OCC algorithm: {occ_algorithm}. Supported algorithms are: centroid, lof, svm.")

        self.classifier.fit(np.zeros((2, hidden_sizes[-1])))  # Dummy fit to initialize the classifier


        # store the latents of each sample to train the classifier
        self.train_step_latents = []

        # Store the losses of each sample in each train step in epoch
        # to compute anomaly detection threshold
        self.train_step_losses = []

        # Store the preds and labels to compute metrics at the end of the epoch
        self.validation_step_scores = []
        self.validation_step_preds = []
        self.validation_step_labels = []

        self.test_step_socores = []
        self.test_step_preds = []
        self.test_step_labels = []


    def _build_encoder(self):
        """ Create the encoder NN here """
        encoder = nn.Sequential()
        input_size = self.input_size
        for hidden_size in self.hidden_sizes[:-1]:
            encoder.append(nn.Linear(input_size, hidden_size))
            if self.batch_norm:
                encoder.append(nn.BatchNorm1d(hidden_size))
            encoder.append(nn.ReLU())
            if self.dropout:
                encoder.append(nn.Dropout(self.dropout))
            input_size = hidden_size

        encoder.append(nn.Linear(input_size, self.hidden_sizes[-1]))

        return encoder
    
    def _build_decoder(self):
        decoder = nn.Sequential()
        input_size = self.hidden_sizes[-1]
        for hidden_size in reversed(self.hidden_sizes[:-1]):
            decoder.append(nn.Linear(input_size, hidden_size))
            if self.batch_norm:
                decoder.append(nn.BatchNorm1d(hidden_size))
            decoder.append(nn.ReLU())
            if self.dropout:
                decoder.append(nn.Dropout(self.dropout))
            input_size = hidden_size

        decoder.append(nn.Linear(input_size, self.input_size))

        return decoder

    def on_fit_start(self):
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

        # SAE loss: reconstruction loss + l2 regularization (vector norm)
        loss = F.mse_loss(x, x_recon, reduction="none").mean(dim=1) + self.lmb * torch.linalg.vector_norm(x_latent, ord=2, dim=1)
        loss = F.tanh(loss)

        self.train_step_latents.append(x_latent)
        self.train_step_losses.append(loss)

        loss = loss.mean()

        return loss
    
    def on_train_epoch_end(self):

        latents = torch.cat(self.train_step_latents)
        losses = torch.cat(self.train_step_losses)
        
        if self._if_fit_occ_this_epoch():
            
            fit_sample = None
            if self.occ_fit_sample_size == 1.0:
                fit_sample = latents
            elif isinstance(self.occ_fit_sample_size, int):
                fit_sample_size = min(self.occ_fit_sample_size, latents.shape[0])
            elif isinstance(self.occ_fit_sample_size, float):
                fit_sample_size = int(self.occ_fit_sample_size * latents.shape[0])
            
            if not fit_sample:
                rng = np.random.default_rng()
                fit_sample_indices = rng.choice(latents.shape[0], size=fit_sample_size, replace=False)
                fit_sample = latents[fit_sample_indices].detach().cpu().numpy()


            self.classifier.fit(fit_sample)

            anomaly_scores = self._calc_anomaly_score(latents)

            if self.occ_algorithm == "centroid":
                self.cen_dst_threshold = anomaly_scores.quantile(0.9)
                self.log("cen_dst_threshold", self.cen_dst_threshold)
            elif self.occ_algorithm == "lof":
                self.log("lof_offset", self.classifier.offset_)

        self.log("train_loss", losses.mean())
        self.log("train_loss_std", losses.std())
        self.log("min_loss", losses.min())
        self.log("max_loss", losses.max())

        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])

        self.train_step_latents.clear()
        self.train_step_losses.clear()

    def validation_step(self, batch, batch_idx):

        if self._if_val_occ_this_epoch():
            x, y, attack_cat = batch
            x_latent = self.encoder(x)

            anomaly_score = self._calc_anomaly_score(x_latent)
            preds = self._calc_predictions(x_latent)
            
            self.validation_step_scores.append(anomaly_score)
            self.validation_step_preds.append(preds)
            self.validation_step_labels.append(y)
 
        return 0
    
    def on_validation_epoch_end(self):
        
        if self._if_val_occ_this_epoch():
            # Concatenate all losses and labels of the samples in the validation step
            anomaly_scores = torch.cat(self.validation_step_scores)
            preds = torch.cat(self.validation_step_preds)
            labels = torch.cat(self.validation_step_labels)

            accuracy = tmf.accuracy(preds, labels, task="binary")
            precision = tmf.precision(preds, labels, task="binary")
            recall = tmf.recall(preds, labels, task="binary")
            f1 = tmf.f1_score(preds, labels, task="binary")
            auroc = tmf.auroc(anomaly_scores, labels, task="binary")

            self.log("val_accuracy", accuracy)
            self.log("val_precision", precision)
            self.log("val_recall", recall)
            self.log("val_f1", f1)
            self.log("val_auroc", auroc)

            self.log("positive_rate", preds.float().mean())


            # Reset the losses and labels for the next epoch
            self.validation_step_scores.clear()
            self.validation_step_preds.clear()
            self.validation_step_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y, attack_cat = batch
        x_latent = self.encoder(x)

        anomaly_score = self._calc_anomaly_score(x_latent)
        preds = self._calc_predictions(x_latent)

        self.test_step_socores.append(anomaly_score)
        self.test_step_preds.append(preds)
        self.test_step_labels.append(y)

        return 0

    def on_test_epoch_end(self):
        # Concatenate all losses and labels of the samples in the validation step
        anomaly_scores = torch.cat(self.test_step_socores)
        preds = torch.cat(self.test_step_preds)
        labels = torch.cat(self.test_step_labels)

        accuracy = tmf.accuracy(preds, labels, task="binary")
        precision = tmf.precision(preds, labels, task="binary")
        recall = tmf.recall(preds, labels, task="binary")
        f1 = tmf.f1_score(preds, labels, task="binary")
        auroc = tmf.auroc(anomaly_scores, labels, task="binary")

        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_auroc", auroc)

        # Reset the losses and labels for the next epoch
        self.test_step_socores.clear()
        self.test_step_preds.clear()
        self.test_step_labels.clear()
    
    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        x_latent = self.encoder(x)

        preds = self._calc_predictions(x_latent)

        return preds

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = LinearLR(optimizer, start_factor=self.linear_lr_start_factor, end_factor=self.linear_lr_end_factor, total_iters=self.linear_lr_total_iters)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

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

    def evaluate(self, test_dataset, logger_params = {}):
        
        if logger_params:
            logger_params = self.default_logger_params | logger_params
            logger = MLFlowLogger(**logger_params)
        else:
            logger = None

        trainer = L.Trainer(accelerator=self.tech_params["accelerator"], logger=logger)

        test_loader = self._get_loader(test_dataset, shuffle=False)

        metrics = trainer.test(self, test_loader)

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
        pass

    @staticmethod
    def load(path):
        pass

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

    def _calc_anomaly_score(self, x_latent):
        if self.occ_algorithm == "centroid":
            cen = self.classifier.means_[0]
            cen_distance = np.linalg.norm(x_latent.detach().cpu().numpy() - cen, axis=1)
            score = torch.tensor(cen_distance, dtype=x_latent.dtype).to(x_latent.device)
            score = F.tanh(score)
        elif self.occ_algorithm == "lof":
            score = self.classifier.decision_function(x_latent.detach().cpu().numpy())
            score = torch.tensor(score, dtype=x_latent.dtype).to(x_latent.device)
            score = 1.0 - F.sigmoid(score)

        return score
    
    def _calc_predictions(self, x_latent):
        if self.occ_algorithm == "centroid":
            cen = self.classifier.means_[0]
            cen_distance = np.linalg.norm(x_latent.detach().cpu().numpy() - cen, axis=1)
            preds = torch.tensor(cen_distance, dtype=x_latent.dtype).to(x_latent.device) 
            preds = F.tanh(preds) > self.cen_dst_threshold
        elif self.occ_algorithm == "lof":
            preds = self.classifier.predict(x_latent.detach().cpu().numpy())
            preds[preds == 1] = 0
            preds[preds == -1] = 1
            preds = torch.tensor(preds, dtype=x_latent.dtype).to(x_latent.device)

        return preds
    
    def _if_fit_occ_this_epoch(self):
        if not self.fit_occ_once:
            return True
        if self.fit_occ_once and self.current_epoch == self.trainer.max_epochs - 2:
            return True
        
        return False
    
    def _if_val_occ_this_epoch(self):
        if not self.fit_occ_once:
            return True
        if self.fit_occ_once and self.current_epoch == self.trainer.max_epochs - 1:
            return True
        
        return False