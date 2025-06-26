import locale

from pathlib import Path

import matplotlib.pyplot as plt

from plotting_config import configure_plotting

def plot_ROC_curve(fpr, tpr, auc, cluster_size, model, dataset, save_path=None):
    """
    Plots the ROC curves. fpr, tpr, auc and cluster_size are expected to be lists of K array-like objects, where K is the number of sub-classifiers.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    configure_plotting(font_size=27)

    cluster_sizes_sum = sum(cluster_size)

    for i in range(len(fpr)):
        if len(fpr) == 1:
            ax.plot(fpr[i], tpr[i], label=f'AUROC = {auc[i]:.3f}', linewidth=4)
        else:
            ax.plot(fpr[i], tpr[i], label=f'Klaster {i+1}: AUROC = {auc[i]:.3f} ({(cluster_size[i] /cluster_sizes_sum):.2f})', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--')


    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    fig.suptitle(f"Krzywa ROC - {model} na zbiorze {dataset}")

    if len(fpr) == 1:
        ax.legend(loc='lower right')
    else:
        avg_auc = 0
        for a, s in zip(auc, cluster_size):
            avg_auc += a * (s / cluster_sizes_sum)
        ax.legend(loc='lower right', title=f"Średnie AUROC = {locale.format_string('%.3f', avg_auc, grouping=True)}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.005])
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()