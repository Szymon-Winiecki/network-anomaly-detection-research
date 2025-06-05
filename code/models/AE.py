from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

import torch.nn.functional as F
import torch

import torchmetrics.functional as tmf

from IADModel import IADModel
from AEBase import AEBase

class AE(AEBase, IADModel):
    """ Standard autoencoder model to make predicions based on reconstruction error """

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
                 threshold_quantile : float = 0.9,):
        """ Standard autoencoder model to make predicions based on reconstruction error

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
            threshold_quantile (float, optional): Detection threshold is set to the quantile of the training losses. Defaults to 0.9.
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
            threshold_quantile=threshold_quantile,
        )

        self.threshold_quantile = threshold_quantile

        self.threshold = 0

        # Store the losses of each sample in each train step in epoch
        # to compute anomaly detection threshold
        self.train_step_losses = []

        # Store the losses and labels to compute metrics at the end of the epoch
        self.validation_step_losses = []
        self.validation_step_labels = []

        self.test_step_losses = []
        self.test_step_labels = []

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
        loss = self._calc_loss(x, x_recon)

        self.train_step_losses.append(loss.detach().clone())

        loss = loss.mean()

        return loss
    
    def on_train_epoch_end(self):
        losses = torch.cat(self.train_step_losses)

        self.threshold = losses.quantile(self.threshold_quantile)

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

        loss = self._calc_loss(x, x_recon)
            
        self.validation_step_losses.append(loss)
        self.validation_step_labels.append(y)
 
        return loss.mean()
    
    def on_validation_epoch_end(self):
        # Concatenate all losses and labels of the samples in the validation step
        losses = torch.cat(self.validation_step_losses)
        labels = torch.cat(self.validation_step_labels)

        preds = losses > self.threshold

        normal_losses = losses[labels == 0]

        accuracy = tmf.accuracy(preds, labels, task="binary")
        precision = tmf.precision(preds, labels, task="binary")
        recall = tmf.recall(preds, labels, task="binary")
        specificity = tmf.specificity(preds, labels, task="binary")
        f1 = tmf.f1_score(preds, labels, task="binary")
        auroc = tmf.auroc(F.tanh(losses), labels, task="binary") # tanh to transform losses form (0, inf) to (0,1) range (for AUROC metric)
        average_precision = tmf.average_precision(F.tanh(losses), labels, task="binary")

        self.log("val_accuracy", accuracy)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_specificity", specificity)
        self.log("val_f1", f1)
        self.log("val_auroc", auroc)
        self.log("val_average_precision", average_precision)

        self.log("val_normal_loss", normal_losses.mean())

        self.log("positive_rate", preds.float().mean())


        # Reset the losses and labels for the next epoch
        self.validation_step_losses.clear()
        self.validation_step_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y, attack_cat = batch
        x_recon = self.forward(x)

        loss = self._calc_loss(x, x_recon)

        self.test_step_losses.append(loss)
        self.test_step_labels.append(y)

        return loss.mean()
    
    def on_test_epoch_end(self):
        # Concatenate all losses and labels of the samples in the test step
        losses = torch.cat(self.test_step_losses)
        labels = torch.cat(self.test_step_labels)

        preds = losses > self.threshold

        accuracy = tmf.accuracy(preds, labels, task="binary")
        precision = tmf.precision(preds, labels, task="binary")
        recall = tmf.recall(preds, labels, task="binary")
        specificity = tmf.specificity(preds, labels, task="binary")
        f1 = tmf.f1_score(preds, labels, task="binary")
        auroc = tmf.auroc(F.tanh(losses), labels, task="binary")
        average_precision = tmf.average_precision(F.tanh(losses), labels, task="binary")

        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_specificity", specificity)
        self.log("test_f1", f1)
        self.log("test_auroc", auroc)
        self.log("test_average_precision", average_precision)

        # Reset the losses and labels for the next epoch
        self.test_step_losses.clear()
        self.test_step_labels.clear()
    
    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        x_recon = self.forward(x)
        loss = self._calc_loss(x, x_recon)
        pred = loss > self.threshold
        return pred
    
    def adjust_threshold(self, score, y):
        """
        Adjust the thrshold to achieve best F1 score.
        
        Args:
            score (torch.Tensor): The anomaly scores.
            y (torch.Tensor): The labels.
        """

        thresholds = torch.linspace(0, 1, 500)
        f1_scores = []

        for t in thresholds:
            preds = score > t
            f1 = tmf.binary_f1_score(preds, y)
            f1_scores.append(f1)

        best_threshold = thresholds[torch.argmax(torch.tensor(f1_scores))]

        self.threshold = best_threshold

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
        self.threshold = self.threshold.to(device=self.device)

        model_mode = self.training
        self.eval()

        dataloader = self._get_loader(dataset, shuffle=False)

        predictions = []

        with torch.no_grad():
            for x, _, _ in dataloader:
                x = x.to(device=self.device)
                x_recon = self.forward(x)
                loss = self._calc_loss(x, x_recon)
                pred = loss > self.threshold

                predictions.append(pred)

        predictions = torch.cat(predictions)

        self.train(model_mode)

        return predictions


    def predict_raw(self, dataset):
        self.threshold = self.threshold.to(device=self.device)

        model_mode = self.training
        self.eval()

        dataloader = self._get_loader(dataset, shuffle=False)

        scores = []

        with torch.no_grad():
            for x, _, _ in dataloader:
                x = x.to(device=self.device)
                x_recon = self.forward(x)
                loss = self._calc_loss(x, x_recon)
                loss = F.tanh(loss)

                scores.append(loss)

        scores = torch.cat(scores)

        self.train(model_mode)

        return scores

    def save(self, path):

        if path is None:
            path = self._gen_default_checkpoint_path()

        if not isinstance(path, Path):
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "state_dict": self.state_dict(),
            "hyper_parameters": self.hparams,
            "threshold": self.threshold,
        }, path)

        return path

    @staticmethod
    def load(path):
        
        checkpoint = torch.load(path)
        model = AE(**checkpoint["hyper_parameters"])
        model.load_state_dict(checkpoint["state_dict"])
        model.threshold = checkpoint["threshold"]

        return model

    def _calc_loss(self, x, x_recon):
        """ Calculate the reconstruction loss for each sample in x"""
        return F.mse_loss(x, x_recon, reduction="none").mean(dim=1)
