import lightning as L

from torch import nn, optim
import torch.nn.functional as F
import torch

class SimpleAE(L.LightningModule):
    """ Small and simple Autoencoder model to make predicions based on reconstruction error """

    def __init__(self, input_size, latent_size):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, input_size)
        )

        self.threshold = 0

        # Store the losses of each sample in each train step in epoch
        # to compute anomaly detection threshold
        self.train_step_losses = []

        # Store the losses and labels to compute metrics at the end of the epoch
        self.validation_step_losses = []
        self.validation_step_labels = []

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self.forward(x)
        loss = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)

        self.train_step_losses.append(loss)

        loss = loss.mean()

        self.log("train_loss", loss)

        return loss
    
    def on_train_epoch_end(self):
        losses = torch.cat(self.train_step_losses)

        self.threshold = losses.mean() + 1 * losses.std()

        self.log("threshold", self.threshold)

        self.train_step_losses.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_recon = self.forward(x)

        loss = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)

        self.validation_step_losses.append(loss)
        self.validation_step_labels.append(y)
        
        # penalize loss for normal samples and reward for loss of anomalies
        weighted_loss = loss * (y * -2 + 1)
        weighted_loss = weighted_loss.mean()

        self.log("val_loss", weighted_loss)
 
        return weighted_loss
    
    def on_validation_epoch_end(self):
        # Concatenate all losses and labels of the samples in the validation step
        losses = torch.cat(self.validation_step_losses)
        labels = torch.cat(self.validation_step_labels)

        # Compute the accuracy of the model
        preds = losses > self.threshold
        accuracy = (preds == labels).float().mean()

        self.log("val_accuracy", accuracy)
        self.log("positive_rate", preds.float().mean())


        # Reset the losses and labels for the next epoch
        self.validation_step_losses.clear()
        self.validation_step_labels.clear()
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self.forward(x)
        loss = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)
        pred = loss > self.threshold
        return pred

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)