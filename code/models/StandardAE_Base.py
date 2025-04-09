from abc import ABCMeta, abstractmethod

import lightning as L

from torch import optim
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import LinearLR

import torcheval.metrics.functional as tmf

class StandardAE_Base(L.LightningModule, metaclass=ABCMeta):
    """ Standard autoencoder model to make predicions based on reconstruction error """

    def __init__(self, input_size, latent_size, dropout=0.3):
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.latent_size = latent_size
        self.dropout = dropout

        self.encoder = self._build_encoder()

        self.decoder = self._build_decoder()

        self.threshold = 0

        # Store the losses of each sample in each train step in epoch
        # to compute anomaly detection threshold
        self.train_step_losses = []

        # Store the losses and labels to compute metrics at the end of the epoch
        self.validation_step_losses = []
        self.validation_step_labels = []


    @abstractmethod
    def _build_encoder(self):
        """ Create the encoder NN here """
        pass
    
    @abstractmethod
    def _build_decoder(self):
        """ Create the decoder NN here """
        pass

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
        x_recon = self.forward(x)
        loss = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)

        self.train_step_losses.append(loss)

        loss = loss.mean()

        return loss
    
    def on_train_epoch_end(self):
        losses = torch.cat(self.train_step_losses)

        self.threshold = losses.quantile(0.9)

        self.log("train_loss", losses.mean())
        self.log("train_loss_std", losses.std())
        self.log("min_loss", losses.min())
        self.log("max_loss", losses.max())

        self.log("threshold", self.threshold)

        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])

        self.train_step_losses.clear()

    def validation_step(self, batch, batch_idx):
        x, y, attack_cat = batch
        x_recon = self.forward(x)

        loss = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)

        self.validation_step_losses.append(loss)
        self.validation_step_labels.append(y)
 
        return loss.mean()
    
    def on_validation_epoch_end(self):
        # Concatenate all losses and labels of the samples in the validation step
        losses = torch.cat(self.validation_step_losses)
        labels = torch.cat(self.validation_step_labels)

        preds = losses > self.threshold

        accuracy = tmf.binary_accuracy(preds, labels)
        precision = tmf.binary_precision(preds, labels)
        recall = tmf.binary_recall(preds, labels)
        f1 = tmf.binary_f1_score(preds, labels)

        self.log("val_accuracy", accuracy)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

        self.log("positive_rate", preds.float().mean())


        # Reset the losses and labels for the next epoch
        self.validation_step_losses.clear()
        self.validation_step_labels.clear()
    
    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        x_recon = self.forward(x)
        loss = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)
        pred = loss > self.threshold
        return pred

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=30)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }