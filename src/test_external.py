import sys
import logging

import torch

from prepare_experiment import prepare_experiment
from utils.data_io_external_source import prepare_datasets
from utils.externel_metrics.dnn_hmm_metrics import compute_dnn_hmm_metrics

torch.use_deterministic_algorithms(True, warn_only=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    argv = sys.argv[1:]
    if len(argv) == 0:
        argv = ['config/test.yaml']

    prepared = prepare_experiment(argv, prepare_exp_dir=False)
    hparams = prepared['hparams']
    datasets, label_encoder = prepare_datasets(hparams)

    train_dataset, valid_dataset, test_dataset = datasets
    compute_dnn_hmm_metrics(test_dataset)
