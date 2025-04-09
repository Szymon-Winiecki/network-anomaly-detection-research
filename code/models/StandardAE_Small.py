from StandardAE_Base import StandardAE_Base

from torch import nn

class StandardAE_Small(StandardAE_Base):
    """ 70K params standard autoencoder """

    def _build_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.latent_size)
        )

    def _build_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.input_size)
        )