import locale

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from plotting_config import configure_plotting

line_styles = [
    "r--",
    "b--",
    "g--",
    "k--",
    "c--",
    "y--",
    "r:",
    "b:",
    "g:",
    "k:",
    "c:",
]

def get_line_style(i, color_blind_mode=True):
    if not color_blind_mode or i >= len(line_styles):
        return ["--"]
    else:
        return [line_styles[i]]


def compare_with_literature(my_means, my_stds, literature_means, model_labels, literature_index, title, metric="AUROC", save_path=None):

    configure_plotting(font_size=14)

    existing_lit = [i for i in range(len(literature_means)) if literature_means[i] is not None]

    def filter_list(list, filtered_indices):
        return [item for i, item in enumerate(list) if i in filtered_indices]
    
    my_means = filter_list(my_means, existing_lit)
    my_stds = filter_list(my_stds, existing_lit)
    literature_means = filter_list(literature_means, existing_lit)
    model_labels = filter_list(model_labels, existing_lit)

    data = {
        'własne eksperymenty': (my_means, my_stds, 'mediumslateblue'),
        f"wyniki z [{literature_index}]": (literature_means, None, 'deepskyblue'),
    }

    x = np.arange(len(model_labels))  # the label locations
    width = 0.45  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, (mean, std, color) in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, mean, width, label=attribute, color=color)
        if std:
            ax.errorbar(x + offset, mean, yerr=std, fmt='none', capsize=5, color='midnightblue', elinewidth=1)
        ax.bar_label(rects, fmt=lambda x: locale.format_string("%.3f", x, grouping=True), padding=-30)
        multiplier += 1

    
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x + width / 2, model_labels)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0.5, 1.1)


    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def compare_models(dataset, models, means, stds, metric="AUROC", save_path=None):

    configure_plotting(font_size=14)

    width = 0.45  # the width of the bars
    x = np.arange(len(models)) * (width + 0.05) # the label locations
    

    fig, ax = plt.subplots(layout='constrained')
    rects = ax.bar(x, means, width, label='Średnia', color='deepskyblue')
    ax.errorbar(x, means, yerr=stds, fmt='none', capsize=5, color='midnightblue', elinewidth=1)

    ax.set_ylabel(metric)
    ax.set_title(f'Porównanie modeli na zbiorze {dataset}')
    ax.set_xticks(x, models)
    ax.legend()
    ax.set_ylim(0.5, 1.1)
    ax.bar_label(rects, fmt=lambda x: locale.format_string("%.3f", x, grouping=True), padding=-30)

    plt.xticks(rotation=45, ha='right')

    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def compare_param(datasets, param_values, score_means, param_name="parametr", metric="AUROC", title="", ticks=None, continuous=False, color_blind_mode=False, save_path=None):
    """ Generates a line plot comparing diffrent values of a parameter on diffrent datasets."""

    configure_plotting(font_size=14)

    fig, ax = plt.subplots(layout='constrained')

    i = 0

    if continuous:
        for dataset, values, means in zip(datasets, param_values, score_means):
            ax.plot(values, means, *get_line_style(i, color_blind_mode), ls="-", label=dataset)
            i += 1
    else:
        for dataset, values, means in zip(datasets, param_values, score_means):
            ax.plot(values, means, *get_line_style(i, color_blind_mode), marker='o', label=dataset)
            i += 1

    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)

    ax.set_xlabel(param_name)
    ax.set_ylabel(metric)
    ax.set_title(title)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.legend()

    fig.tight_layout()
    

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
