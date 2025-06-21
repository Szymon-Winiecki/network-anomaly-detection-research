from datetime import datetime
from pathlib import Path

from lightning.pytorch.callbacks import Callback

import torch

import sys
sys.path.append('../visualization')

from visualization.space_visualization import plot_latent_space_evolution

class LatentSpaceEvolutionPlotter(Callback):
    """
    Callback to plot the evolution of the latent space of a model in a evaluation loop.
    """

    def __init__(self, save_dir, filename, dataset_name, plot_epochs=[], plots_per_row: int = 3,  plot_train : bool = True, plot_val : bool = True):
        """
        Args:
            save_dir (str): Directory to save the plots.
            filename (str): Base filename for the saved plots.
            dataset_name (str): Name of the dataset being used.
            plot_epochs (list): List of epochs at which to plot the latent space.
            plots_per_row (int): Number of plots to display per row in the figure.
            plot_train (bool): Whether to plot the training latent space.
            plot_val (bool): Whether to plot the validation latent space.
        """
        super().__init__()

        self.save_dir = Path(save_dir)
        self.filename = filename

        self.dataset_name = dataset_name

        self.plot_train = plot_train
        self.plot_val = plot_val

        self.plot_epochs = plot_epochs
        self.plots_per_row = plots_per_row

        self.val_fig = None
        self.val_axs = None
        self.val_counter = 0

        self.val_file_num = 0

        self.train_fig = None
        self.train_axs = None
        self.train_counter = 0

        self.train_file_num = 0
        

    def on_train_epoch_end(self, trainer, pl_module):

        if not self.plot_train:
            return
    
        if trainer.sanity_checking:
            return
        
        if trainer.current_epoch == 0:
            self.train_file_num += 1
            self.train_fig = None
            self.train_axs = None
            self.train_counter = 0

        if trainer.current_epoch not in self.plot_epochs:
            return
        
        if hasattr(pl_module, 'train_step_latents'):
            train_latents = pl_module.train_step_latents
        else:
            raise ValueError("Model does not have 'train_step_latents' attribute.")
    
        
        train_latents = torch.cat(train_latents).cpu().numpy()
        train_labels = torch.zeros(train_latents.shape[0], dtype=int)  # Assuming all training data is normal traffic

        time_id = int((datetime.now().timestamp() - datetime(2025, 1, 1).timestamp()) * 100)
        save_path = self.save_dir / f"{self.filename}_train_{self.train_file_num}_{time_id}_{pl_module.name}_{self.dataset_name}_epoch_{trainer.current_epoch}.png"

        t_max = len(self.plot_epochs) - 1

        self.train_fig, self.train_axs = plot_latent_space_evolution(train_latents, train_labels, pl_module.name, self.dataset_name, trainer.current_epoch, fig=self.train_fig, axs=self.train_axs, t=self.train_counter, t_max=t_max, plots_per_row=self.plots_per_row, save_path=save_path)
        self.train_counter += 1

    def on_validation_epoch_end(self, trainer, pl_module):

        if not self.plot_val:
            return

        if trainer.sanity_checking:
            return
        
        if trainer.current_epoch == 0:
            self.val_file_num += 1
            self.val_fig = None
            self.val_axs = None
            self.val_counter = 0
        
        if trainer.current_epoch not in self.plot_epochs:
            return
        
        if hasattr(pl_module, 'validation_step_latents'):
            val_latents = pl_module.validation_step_latents
        else:
            raise ValueError("Model does not have 'validation_step_latents' attribute.")

        if hasattr(pl_module, 'validation_step_labels'):
            val_labels = pl_module.validation_step_labels
        else:
            raise ValueError("Model does not have 'validation_step_labels' attribute.")
        
        val_latents = torch.cat(val_latents).cpu().numpy()
        val_labels = torch.cat(val_labels).cpu().numpy()
        
        time_id = int((datetime.now().timestamp() - datetime(2025, 1, 1).timestamp()) * 100)
        save_path = self.save_dir / f"{self.filename}_val_{self.val_file_num}_{time_id}_{pl_module.name}_{self.dataset_name}_epoch_{trainer.current_epoch}.png"
        
        t_max = len(self.plot_epochs) - 1

        self.val_fig, self.val_axs = plot_latent_space_evolution(val_latents, val_labels, pl_module.name, self.dataset_name, trainer.current_epoch, fig=self.val_fig, axs=self.val_axs, t=self.val_counter, t_max=t_max, plots_per_row=self.plots_per_row, save_path=save_path)
        self.val_counter += 1
        
