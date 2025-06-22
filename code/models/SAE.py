from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

import torch.nn.functional as F
import torch

# import torcheval.metrics.functional as tmf
import torchmetrics.functional as tmf

from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

import numpy as np

from IADModel import IADModel
from AEBase import AEBase


class SAE(AEBase, IADModel):
    """ Shrink Autoencoder model 

    Implementation of model proposed in a paper:
    V. L. Cao, M. Nicolau and J. McDermott, Learning Neural Representations for
    Network Anomaly Detection, in IEEE Transactions on Cybernetics, vol. 49, no. 8,
    pp. 3074-3087, Aug. 2019, doi: 10.1109/TCYB.2018.2838668
    """

    occ_algorithm_acronyms = {
        "centroid": "CEN",
        "lof": "LOF",
        "svm": "OCSVM",
        "re": "RE"
    }

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
                 lmb : float = 0.1,
                 fit_occ_once : bool = False,
                 occ_algorithm : str = "centroid",
                 threshold_quantile : float = 0.9,
                 occ_fit_sample_size : int | float = 1.0,
                 *occ_args,
                 **occ_kwargs) -> None:
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
            lmb (float, optional): Balance parameter for the SAE loss. Defaults to 0.1.
            fit_occ_once (bool, optional): if False, the model will fit the OCC algorithm every epoch. If True, the model will fit the OCC algorithm only once, at the end of the training, it also results in no validation metrics from non-last epochs. Defaults to False.
            occ_algorithm (str, optional): OCC algorithm to use. Supported algorithms are: centroid, lof, svm, re. Defaults to "centroid".
                re (reconstruction error) is a special case, where the model uses the reconstruction error as the anomaly score and the threshold is calculated based on the training set (not an occ).
            threshold_quantile (float, optional): The quantile to use for the threshold calculation in case of "centroid" and "re" algorithms. Defaults to 0.9.
            occ_fit_sample_size (int | float, optional): If int, the number of samples to use for fitting the OCC algorithm. If float, the percentage of samples to use for fitting the OCC algorithm. Defaults to 1.0 (full training dataset).
            *occ_args: Additional arguments for the OCC algorithm.
            **occ_kwargs: Additional keyword arguments for the OCC algorithm.
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
            lmb=lmb,
            fit_occ_once=fit_occ_once,
            occ_algorithm=occ_algorithm,
            threshold_quantile=threshold_quantile,
            occ_fit_sample_size=occ_fit_sample_size,
            **occ_kwargs
        )
        IADModel.__init__(self, name=f"SAE({self.occ_algorithm_acronyms.get(occ_algorithm, occ_algorithm)})")

        self.shrink_weight = lmb

        self.threshold_quantile = threshold_quantile
        self.fit_occ_once = fit_occ_once
        self.occ_fit_sample_size = occ_fit_sample_size

        self._prepare_occ(occ_algorithm, **occ_kwargs)

        # store the latents of each sample to train the classifier
        self.train_step_latents = []
        # store the losses of each sample if method is "re" to calculate re detection threshold
        self.train_step_re_losss = []

        # Store the losses of each sample in each train step in epoch
        # to compute anomaly detection threshold
        self.train_step_losses = []

        # Store the preds and labels to compute metrics at the end of the epoch
        self.validation_step_scores = []
        self.validation_step_preds = []
        self.validation_step_labels = []
        self.validation_step_latents = []

        self.test_step_socores = []
        self.test_step_preds = []
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
        x_latent = self.encoder(x)
        x_recon = self.decoder(x_latent)

        loss, re_loss, shrink_loss = self._calc_loss(x, x_recon, x_latent)

        self.train_step_latents.append(x_latent.detach().clone())
        self.train_step_losses.append(loss.detach().clone())

        if self.occ_algorithm == "re":
            self.train_step_re_losss.append(re_loss.detach().clone())

        loss = loss.mean()

        return loss
    
    def on_train_epoch_end(self):

        latents = torch.cat(self.train_step_latents)
        losses = torch.cat(self.train_step_losses)

        # special case for detection based on reconstruction error (re)
        if self.occ_algorithm == "re":
            re_loss = torch.cat(self.train_step_re_losss)
            self.threshold = re_loss.quantile(self.threshold_quantile)
            self.log("re_threshold", self.threshold)
        
        elif self._if_fit_occ_this_epoch():
            
            self._fit_occ(latents)

            if self.occ_algorithm == "centroid":
                anomaly_scores = self._calc_anomaly_score(latents)
                self.cen_dst_threshold = anomaly_scores.quantile(self.threshold_quantile)
                self.log("cen_dst_threshold", self.cen_dst_threshold)
            elif self.occ_algorithm == "lof":
                self.log("lof_offset", self.classifier.offset_)

        self.log("train_loss", losses.mean())

        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])

        self.train_step_latents.clear()
        self.train_step_losses.clear()
        self.train_step_re_losss.clear()

    def validation_step(self, batch, batch_idx):
        
        if self.occ_algorithm == "re":
            x, y , attack_cat = batch
            x_latent = self.encoder(x)
            x_recon = self.decoder(x_latent)

            anomaly_score = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)

            preds = anomaly_score > self.threshold

            self.validation_step_scores.append(anomaly_score)
            self.validation_step_preds.append(preds)
            self.validation_step_labels.append(y)
            self.validation_step_latents.append(x_latent)

        elif self._if_val_occ_this_epoch():
            x, y, attack_cat = batch
            x_latent = self.encoder(x)

            anomaly_score = self._calc_anomaly_score(x_latent)
            preds = self._calc_predictions(x_latent)
            
            self.validation_step_scores.append(anomaly_score)
            self.validation_step_preds.append(preds)
            self.validation_step_labels.append(y)
            self.validation_step_latents.append(x_latent)
 
        return 0
    
    def on_validation_epoch_end(self):
        
        if self.occ_algorithm == "re" or self._if_val_occ_this_epoch():
            # Concatenate all losses and labels of the samples in the validation step
            anomaly_scores = torch.cat(self.validation_step_scores)
            preds = torch.cat(self.validation_step_preds)
            labels = torch.cat(self.validation_step_labels)

            accuracy = tmf.accuracy(preds, labels, task="binary")
            precision = tmf.precision(preds, labels, task="binary")
            recall = tmf.recall(preds, labels, task="binary")
            specificity = tmf.specificity(preds, labels, task="binary")
            f1 = tmf.f1_score(preds, labels, task="binary")
            mcc = tmf.matthews_corrcoef(preds, labels, task="binary")

            if self.occ_algorithm == "re":
                auroc = tmf.auroc(F.tanh(anomaly_scores), labels, task="binary")
                average_precision = tmf.average_precision(F.tanh(anomaly_scores), labels, task="binary")
            else:
                auroc = tmf.auroc(anomaly_scores, labels, task="binary")
                average_precision = tmf.average_precision(anomaly_scores, labels, task="binary")

            self.log("val_accuracy", accuracy)
            self.log("val_precision", precision)
            self.log("val_recall", recall)
            self.log("val_specificity", specificity)
            self.log("val_f1", f1)
            self.log("val_mcc", mcc)
            self.log("val_auroc", auroc)
            self.log("val_average_precision", average_precision)

            self.log("positive_rate", preds.float().mean())


            # Reset the losses and labels for the next epoch
            self.validation_step_scores.clear()
            self.validation_step_preds.clear()
            self.validation_step_labels.clear()
            self.validation_step_latents.clear()

    def test_step(self, batch, batch_idx):
        x, y, attack_cat = batch

        if self.occ_algorithm == "re":
            x_recon = self.forward(x)

            anomaly_score = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)

            preds = anomaly_score > self.threshold

        else:
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
        specificity = tmf.specificity(preds, labels, task="binary")
        f1 = tmf.f1_score(preds, labels, task="binary")
        mcc = tmf.matthews_corrcoef(preds, labels, task="binary")
        
        if self.occ_algorithm == "re":
            auroc = tmf.auroc(F.tanh(anomaly_scores), labels, task="binary")
            average_precision = tmf.average_precision(F.tanh(anomaly_scores), labels, task="binary")
        else:
            auroc = tmf.auroc(anomaly_scores, labels, task="binary")
            average_precision = tmf.average_precision(anomaly_scores, labels, task="binary")

        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_specificity", specificity)
        self.log("test_f1", f1)
        self.log("test_mcc", mcc)
        self.log("test_auroc", auroc)
        self.log("test_average_precision", average_precision)

        # Reset the losses and labels for the next epoch
        self.test_step_socores.clear()
        self.test_step_preds.clear()
        self.test_step_labels.clear()
    
    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        x_latent = self.encoder(x)

        preds = self._calc_predictions(x_latent)

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

                if self.occ_algorithm == "re":
                    x_recon = self.forward(x)
                    anomaly_score = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)
                    preds = anomaly_score > self.threshold
                else:
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
                
                if self.occ_algorithm == "re":
                    x_recon = self.forward(x)
                    preds = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)
                    preds = F.tanh(preds)
                else:
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

    def _calc_loss(self, x : torch.Tensor, x_recon : torch.Tensor, x_latent : torch.Tensor):
        """ Calculate the loss for the SAE model for each sample
        Args:
            x (torch.Tensor): Input tensor
            x_recon (torch.Tensor): Reconstructed tensor
            x_latent (torch.Tensor): Latent representation of the input tensor
        Returns:
            torch.Tensor: Calculated loss
        """

        # SAE loss: reconstruction loss + l2 regularization (vector norm)
        re_loss = F.mse_loss(x, x_recon, reduction="none").mean(dim=1)
        shrink_loss = torch.linalg.vector_norm(x_latent, ord=2, dim=1) ** 2
        loss = re_loss + self.shrink_weight * shrink_loss

        return loss, re_loss, shrink_loss

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
        elif self.occ_algorithm == "svm":
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
        elif self.occ_algorithm == "svm":
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
    
    def _prepare_occ(self, occ_algorithm, **occ_kwargs):
        self.occ_algorithm = occ_algorithm

        if occ_algorithm == "centroid":
            self.classifier = GaussianMixture(n_components = 1, **occ_kwargs)
            self.cen_dst_threshold = 0
        elif occ_algorithm == "lof":
            self.classifier = LocalOutlierFactor(novelty=True, **occ_kwargs)
        elif occ_algorithm == "svm":
            self.classifier = OneClassSVM(**occ_kwargs)
        elif occ_algorithm == "re":
            self.threshold = 0
        else:
            raise ValueError(f"Unknown OCC algorithm: {occ_algorithm}. Supported algorithms are: centroid, lof, svm, re.")

        if occ_algorithm != "re":
            self.classifier.fit(np.zeros((2, self.hidden_sizes[-1])))  # Dummy fit to initialize the classifier
    
    def _fit_occ(self, latents):
        fit_sample = None
        if self.occ_fit_sample_size == 1.0:
            fit_sample = latents.detach().cpu().numpy()
        elif isinstance(self.occ_fit_sample_size, int):
            fit_sample_size = min(self.occ_fit_sample_size, latents.shape[0])
        elif isinstance(self.occ_fit_sample_size, float):
            fit_sample_size = int(self.occ_fit_sample_size * latents.shape[0])
        
        if fit_sample is None:
            rng = np.random.default_rng()
            fit_sample_indices = rng.choice(latents.shape[0], size=fit_sample_size, replace=False)
            fit_sample = latents[fit_sample_indices].detach().cpu().numpy()

        self.classifier.fit(fit_sample)

    def calc_ROC(self, dataset):
        """ Calculate the ROC curve for the model on the given dataset """

        scores = self.predict_raw(dataset)
        labels = dataset.y

        fpr, tpr, thresholds = tmf.roc(scores, labels, task="binary")
        auroc = tmf.auroc(scores, labels, task="binary")

        cluster_size = labels.shape[0]

        return [fpr], [tpr], [auroc], [cluster_size]