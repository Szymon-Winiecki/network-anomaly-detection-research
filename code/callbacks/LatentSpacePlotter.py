from datetime import datetime
from pathlib import Path

from lightning.pytorch.callbacks import Callback

import torch

import sys
sys.path.append('../visualization')

from visualization.space_visualization import plot_latent_space

class LatentSpacePlotter(Callback):
    """
    Callback to plot the latent space of a model in a evaluation loop.
    """

    def __init__(self, save_dir, filename, dataset_name, plot_epochs : list = []):
        """
        Args:
            save_dir (str): Directory to save the plots.
            filename (str): Base filename for the saved plots.
            dataset_name (str): Name of the dataset being used.
            plot_epochs (list): List of epochs at which to plot the latent space.
        """
        super().__init__()

        self.save_dir = Path(save_dir)
        self.filename = filename

        self.dataset_name = dataset_name

        self.plot_epochs = plot_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if trainer.current_epoch not in self.plot_epochs:
            return
        
        if hasattr(pl_module, 'validation_step_latents'):
            latents = pl_module.validation_step_latents
        else:
            raise ValueError("Model does not have 'validation_step_latents' attribute.")

        if hasattr(pl_module, 'validation_step_labels'):
            labels = pl_module.validation_step_labels
        else:
            raise ValueError("Model does not have 'validation_step_labels' attribute.")
        
        latents = torch.cat(latents).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        
        time_id = int((datetime.now().timestamp() - datetime(2025, 1, 1).timestamp()) * 100)
        save_path = self.save_dir / f"{self.filename}_{time_id}_{pl_module.name}_{self.dataset_name}_epoch_{trainer.current_epoch}.png"
        
        plot_latent_space(latents, labels, pl_module.name, self.dataset_name, save_path=save_path)
        
