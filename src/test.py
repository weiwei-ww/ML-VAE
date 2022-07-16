import sys
import logging

import torch

from prepare_experiment import prepare_experiment

torch.use_deterministic_algorithms(True, warn_only=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    prepared = prepare_experiment(sys.argv[1:], prepare_exp_dir=False)
    hparams = prepared['hparams']
    train_dataset, valid_dataset, test_dataset = prepared['datasets']
    model = prepared['model']

    # test the model
    model.evaluate(
        test_dataset,
        max_key=hparams['model'].get('max_key'),
        min_key=hparams['model'].get('min_key'),
        test_loader_kwargs=hparams['test_dataloader_opts'],
    )
