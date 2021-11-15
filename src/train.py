import sys
import importlib
import ruamel.yaml

from hyperpyyaml import load_hyperpyyaml
from hyperpyyaml.core import recursive_update
import speechbrain as sb

from utils.data_io import prepare_datasets
from prepare import prepare_experiment


if __name__ == '__main__':
    prepared = prepare_experiment(sys.argv[1:], prepare_exp_dir=True)
    hparams = prepared['hparams']
    train_dataset, valid_dataset, test_dataset = prepared['datasets']
    model = prepared['model']

    # fit the model
    model.fit(
        hparams['model']['epoch_counter'],
        train_dataset,
        valid_dataset,
        train_loader_kwargs=hparams['train_dataloader_opts'],
        valid_loader_kwargs=hparams['valid_dataloader_opts'],
    )

    # Test
    model.evaluate(
        test_dataset,
        min_key='PER',
        test_loader_kwargs=hparams['test_dataloader_opts'],
    )
