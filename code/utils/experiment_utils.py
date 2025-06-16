import sys

from datetime import datetime
from pathlib import Path

import pandas as pd

this_directory = Path(__file__).parent.resolve()
sys.path.append(str(this_directory / '../datasets'))
sys.path.append(str(this_directory / '../models'))

from datasets.IDS_Dataset import IDS_Dataset

from models.IADModel import IADModel

class CSVLogger:
    """ Logger class to log experiment records to a CSV file. """

    def __init__(self, path : str | Path):
        """ Initialize the logger with the given path.
        Args:
            path (str | Path): Path to the CSV file where the records will be logged. 
                If the file does not exist, it will be created.
                If the file exists, it will be loaded to initialize the DataFrame.
        """
        self.path = Path(path)
        self.df = None

        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists():
            self.df = pd.read_csv(self.path)

    def log(self, records: dict | list):
        """ Log a record or records to the CSV file.
        Args:
            record (dict | list of dicts): Record to log or a list of records to log.
        """
        if isinstance(records, dict):
            records = [records]

        if self.df is None:
            self.df = pd.DataFrame(columns=records[0].keys())
        
        self.df = pd.concat([self.df, pd.DataFrame(records)], ignore_index=True)
        self.df.to_csv(self.path, index=False)

    def get_df(self):
        return self.df
    

def load_dataset(name, path, pipeline=None, random_state=42):
    """ Load the train, val and test datasets from the given path. """
    train_dataset = IDS_Dataset(path, "train", name=name, transformer=pipeline, random_state=random_state)
    val_dataset = IDS_Dataset(path, "val", name=name, transformer=pipeline, random_state=random_state)
    test_dataset = IDS_Dataset(path, "test", name=name, transformer=pipeline, random_state=random_state)

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }
    
def load_dataset_folds(dataset_name, dataset_dir, kfolds=3, pipeline=None, random_state=42):
    """ Load K folds of the dataset from the given directory for a modified cross_validation.
    Args:
        dataset_name (str): Name of the dataset.
        dataset_dir (str): Root dir to the preprocessed dataset.
        kfolds (int): Number of folds to load.
        pipeline (Pipeline, optional): Optional transformer to be applied on a sample.
    Returns:
        folds (list): List of dictionaries containing train and val datasets for each fold.
    """

    folds = []
    for i in range(kfolds):
        train_dataset = IDS_Dataset(dataset_dir, "train", strategy="cross_validation", fold_idx=i, name=dataset_name, transformer=pipeline, random_state=random_state)
        val_dataset = IDS_Dataset(dataset_dir, "val", strategy="cross_validation", fold_idx=i, name=dataset_name, transformer=pipeline, random_state=random_state)

        folds.append({
            "train": train_dataset,
            "val": val_dataset,
        })

    return folds


def run_experiment(model : IADModel, 
                   dataset : dict,
                   max_epochs : int = 10,
                   trainer_callbacks = None,
                   experiment_name : str = "undefined",
                   run_name : str = "undefined",
                   validation_set : str = "val",
                   save_model : bool = False,
                   fit_params = {}):
    
    """ Run an experiment with the given model and dataset.
    Args:
        model (IADModel): Model to train and evaluate.
        dataset (dict): Dictionary containing train and validation datasets.
        max_epochs (int): Maximum number of epochs to train the model.
        experiment_name (str): Name of the experiment.
        run_name (str): Name of the run.
        validation_set (str): Type of the set to use for evaluation. Possible values are "val" or "test". Default is "val".
        save_model (bool): Whether to save the model after training.
        fit_params (dict): Additional parameters to pass to the fit method.
    Returns:
        experiment_record (dict): Dictionary containing the experiment record with all parameters and metrics.
    """

    model.set_tech_params(
        accelerator='gpu',
        batch_size=1024, 
        num_workers=11, 
        persistent_workers=True
    )

    fit_start_time = datetime.now()
    metrics = model.fit(dataset['train'], dataset[validation_set], max_epochs=max_epochs, trainer_callbacks=trainer_callbacks, log=True, 
                        logger_params = {
                                "experiment_name": experiment_name,
                                "run_name": run_name,
                                "log_model": False,
                                "tags": {"dataset": dataset['train'].name},
                        },
                        **fit_params)
    fit_end_time = datetime.now()
    
    
    save_start_time = datetime.now()
    if save_model:
        save_dir = Path(f"../saved_models/{experiment_name}/{run_name}")
        file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model.__class__.__name__}"
        file_num = 0
        while (save_dir / f"{file_name}_{file_num}.pt").exists():
            file_num += 1

        save_path = save_dir / f"{file_name}_{file_num}.pt"
        model.save(save_path)
        save_path = str(save_path)
    else:
        save_path = ""
    save_end_time = datetime.now()

    hparams = model.hparams.copy()
    hparams_keys = list(hparams.keys())
    for key in hparams_keys:
        if isinstance(hparams[key], dict):
            for sub_key, sub_value in hparams[key].items():
                hparams[f"{key}_{sub_key}"] = sub_value
            del hparams[key]

    tech_params = model.tech_params
    experiment_params = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model.__class__.__name__,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "dataset_name": dataset['train'].name,
        "save_path": save_path,
    }
    experiment_timing = {
        "fit_duration": (fit_end_time - fit_start_time).total_seconds(),
        "save_duration": (save_end_time - save_start_time).total_seconds(),
    }

    experiment_record = experiment_params | tech_params | experiment_timing | hparams | metrics

    return experiment_record

def run_cross_validation(model : IADModel, 
                         dataset_folds : list,
                         max_epochs : int = 10,
                         trainer_callbacks = None,
                         experiment_name : str = "undefined",
                         run_name : str = "undefined",
                         save_model : bool = False,
                         fit_params = {}):
    """ Run modified cross-validation with the given model and dataset folds.
    Args:
        model (IADModel): Blueprint of the model to train and evaluate.
        dataset_folds (list): List of dictionaries containing train and val datasets for each fold.
        max_epochs (int): Maximum number of epochs to train the model.
        experiment_name (str): Name of the experiment.
        run_name (str): Base name for the runs.
        save_model (bool): Whether to save the models after training.
        fit_params (dict): Additional parameters to pass to the fit method.
    Returns:
        records (list): List of dictionaries containing the experiment records for each fold.
        models (list): List of models trained for each fold.
    """
    

    records = []
    models = []

    cv_id = int((datetime.now().timestamp() - datetime(2025, 1, 1).timestamp()) * 100)

    for i, fold in enumerate(dataset_folds):

        model_hparams = model.hparams.copy()
        model_sub_kwargs = {}

        to_delete = []
        for key, value in model_hparams.items():
            if 'kwargs' in key:
                model_sub_kwargs = model_sub_kwargs | value
                to_delete.append(key)

        for key in to_delete:
            del model_hparams[key]

        model_params = model_hparams | model_sub_kwargs

        new_model = model.__class__(**model_params)

        record = run_experiment(
            new_model, 
            fold, 
            max_epochs=max_epochs, 
            trainer_callbacks=trainer_callbacks,
            experiment_name=experiment_name,
            run_name=f"{run_name}_fold_{i}",
            validation_set="val",
            save_model=save_model,
            fit_params=fit_params
        )
        record['cv_id'] = cv_id
        record['cv_fold'] = i

        records.append(record)
        models.append(new_model)

    return records, models