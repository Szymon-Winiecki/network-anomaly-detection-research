from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch

import numpy as np

def plot_data_pca(features, labels, attack_cat):
    
    pca = PCA(n_components=2)
    x = pca.fit_transform(features)

    plt.figure(figsize=(10, 10))
    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title("PCA of Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar()
    plt.show()

# latent_features = model.encoder(val_dataset.x).detach().numpy()
# # print(latent_features)

# plot_data_pca(latent_features, val_dataset.y, val_dataset.attack_cat)

# small_model = SimpleAE.load_from_checkpoint("epoch=199-step=19600.ckpt")

# latent_features = small_model.encoder(val_dataset.x.to('cuda:0')).cpu().detach().numpy()
# plot_data_pca(latent_features, val_dataset.y, val_dataset.attack_cat)

def plot_latent_space_from_model(model, dataset, model_name, save_path=None):
        """ Plot the latent space of the model

        Args:
            dataset (torch.utils.data.Dataset): Dataset to plot
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        dataloader = model._get_loader(dataset, shuffle=False)

        latents = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                x, y, _ = batch
                x = x.to(model.device)
                x_latent = model.encoder(x)

                latents.append(x_latent)
                labels.append(y)

        latents = torch.cat(latents).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

        if latents.shape[1] > 2:
            pca = PCA(n_components=2)
            pca.fit(latents)
            x = pca.transform(latents)
        else:
            x = latents

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        scatter = axs[0].scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', alpha=0.5, vmin=0, vmax=1)
        scatter_norm = axs[1].scatter(x[labels==0, 0], x[labels==0, 1], c=labels[labels==0], cmap='viridis', alpha=0.5, vmin=0, vmax=1)
        scatter_anomaly = axs[2].scatter(x[labels==1, 0], x[labels==1, 1], c=labels[labels==1], cmap='viridis', alpha=0.5, vmin=0, vmax=1)
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        axs[0].set_title(f"{model_name} - {dataset.name} przestrzeń ukryta modelu")
        axs[0].set_xlabel("PC1")
        axs[0].set_ylabel("PC2")

        axs[1].set_title("Dane normalne")
        axs[2].set_title("Anomalie")
        
        legend = axs[0].legend(*scatter.legend_elements(), title="klasy")
        axs[0].add_artist(legend)
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

def plot_latent_space(latents, labels, model, dataset, save_path=None):
    
    if latents.shape[1] > 2:
        pca = PCA(n_components=2)
        latents = pca.fit_transform(latents)

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    font = {'family': 'Times New Roman',
            'size'   : 17}
    plt.rc('font', **font)

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

    font = {'family': 'Times New Roman',
            'size'   : 17}
    plt.rc('font', **font)

    
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