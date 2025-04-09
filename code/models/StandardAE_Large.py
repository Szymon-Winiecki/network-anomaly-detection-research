from StandardAE_Base import StandardAE_Base

from torch import nn

class StandardAE_Large(StandardAE_Base):
    """ 684K params standard autoencoder """

    def _build_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.latent_size)
        )

    def _build_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_size, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, self.input_size)
        )