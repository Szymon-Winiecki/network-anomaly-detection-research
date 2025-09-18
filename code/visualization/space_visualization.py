from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from plotting_config import configure_plotting

def plot_latent_space(latents, labels, model, dataset, save_path=None):
    
    if latents.shape[1] > 2:
        pca = PCA(n_components=2)
        latents = pca.fit_transform(latents)

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    configure_plotting(font_size=25)

    fig.suptitle(f"{model} - przestrzeń ukryta, dane z {dataset}")

    axs[0].scatter(latents[:, 0], latents[:, 1], c=labels, cmap='viridis', alpha=0.5, vmin=0, vmax=1)
    axs[0].set_title(f"Wszystkie dane")
    
    legend = axs[0].legend(*axs[0].collections[0].legend_elements(), title="klasy")
    axs[0].add_artist(legend)

    axs[1].scatter(latents[labels == 0, 0], latents[labels == 0, 1], c=labels[labels == 0], cmap='viridis', alpha=0.5, vmin=0, vmax=1)
    axs[1].set_title("Dane normalne")
    axs[2].scatter(latents[labels == 1, 0], latents[labels == 1, 1], c=labels[labels == 1], cmap='viridis', alpha=0.5, vmin=0, vmax=1)
    axs[2].set_title("Anomalie")

    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    else:
        fig.show()
     

def plot_latent_space_evolution(latents, labels, model, dataset, epoch, fig= None, axs = None, t = 0, t_max = 1, plots_per_row = 3, save_path=None):
    plots_per_row = min(plots_per_row, t_max + 1)
    
    rows = t_max // plots_per_row + 1

    configure_plotting(font_size=25)

    
    if fig is None or axs is None:
        fig, axs = plt.subplots(rows, plots_per_row, figsize=(plots_per_row * 10, 10 * rows))

        fig.suptitle(f"{model} - ewolucja przestrzeni ukrytej, dane z {dataset}")

    if latents.shape[1] > 2:
            pca = PCA(n_components=2)
            latents = pca.fit_transform(latents)

    row = t // plots_per_row
    col = t % plots_per_row

    ax = axs[row, col] if rows > 1 else axs[col]
    ax.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='viridis', alpha=0.5, vmin=0, vmax=1)

    curr_epoch = epoch if epoch is not None else t
    ax.set_title(f"Epoka {curr_epoch}.")

    if t == t_max and save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path)

    return fig, axs


def plot_latent_space_evolution_gif(latents, labels, model, dataset, epoch, save_path=None):
    configure_plotting(font_size=25)

    fig, ax = plt.subplots(1, 1, figsize=( 10, 10 ))

    fig.suptitle(f"{model} - ewolucja przestrzeni ukrytej, dane z {dataset}")

    if latents.shape[1] > 2:
            pca = PCA(n_components=2)
            latents = pca.fit_transform(latents)

    ax.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='viridis', alpha=0.5, vmin=0, vmax=1)

    ax.set_title(f"Epoka {epoch}.")

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path)