from torch import nn, optim

import lightning as L

class AEBase(L.LightningModule):
    
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
                 **kwargs):
        
        super().__init__()

        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.initial_lr = initial_lr
        self.linear_lr_start_factor = linear_lr_start_factor
        self.linear_lr_end_factor = linear_lr_end_factor
        self.linear_lr_total_iters = linear_lr_total_iters

        self.optimizer = optimizer
        self.optimizer_params = optimizer_params if optimizer_params else {}

        self.encoder = self._build_encoder()

        self.decoder = self._build_decoder()

    def _build_encoder(self):
        encoder = nn.Sequential()
        input_size = self.input_size
        for hidden_size in self.hidden_sizes[:-1]:
            encoder.append(nn.Linear(input_size, hidden_size))
            if self.batch_norm:
                encoder.append(nn.BatchNorm1d(hidden_size))
            encoder.append(nn.Tanh())
            if self.dropout:
                encoder.append(nn.Dropout(self.dropout))
            input_size = hidden_size

        encoder.append(nn.Linear(input_size, self.hidden_sizes[-1]))

        encoder.apply(_init_weights_xavier_uniform)

        return encoder
    
    def _build_decoder(self):
        decoder = nn.Sequential()
        input_size = self.hidden_sizes[-1]
        for hidden_size in reversed(self.hidden_sizes[:-1]):
            decoder.append(nn.Linear(input_size, hidden_size))
            if self.batch_norm:
                decoder.append(nn.BatchNorm1d(hidden_size))
            decoder.append(nn.Tanh())
            if self.dropout:
                decoder.append(nn.Dropout(self.dropout))
            input_size = hidden_size

        decoder.append(nn.Linear(input_size, self.input_size))
        
        decoder.apply(_init_weights_xavier_uniform)

        return decoder
    
    def configure_optimizers(self):
        
        match self.optimizer.lower():
            case "adam":
                optimizer = optim.Adam(self.parameters(), lr=self.initial_lr, **self.optimizer_params)
            case "sgd":
                optimizer = optim.SGD(self.parameters(), lr=self.initial_lr, **self.optimizer_params)
            case "adadelta":
                optimizer = optim.Adadelta(self.parameters(), lr=self.initial_lr, **self.optimizer_params)
            case "adagrad":
                optimizer = optim.Adagrad(self.parameters(), lr=self.initial_lr, **self.optimizer_params)
            case "adamw":
                optimizer = optim.AdamW(self.parameters(), lr=self.initial_lr, **self.optimizer_params)
            case _:
                raise ValueError(f"Unknown optimizer: {self.optimizer}. Supported optimizers are: Adam, SGD, Adadelta, Adagrad, AdamW.")
            
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=self.linear_lr_start_factor, end_factor=self.linear_lr_end_factor, total_iters=self.linear_lr_total_iters)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    
def _init_weights_xavier_uniform(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)