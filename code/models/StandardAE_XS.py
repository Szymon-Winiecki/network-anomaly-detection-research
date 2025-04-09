from StandardAE_Base import StandardAE_Base

from torch import nn

class StandardAE_XS(StandardAE_Base):
    """ 30K params standard autoencoder """

    def _build_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, self.latent_size)
        )

    def _build_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_size, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.input_size)
        )