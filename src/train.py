import sys

import torch.autograd

from prepare_experiment import prepare_experiment


if __name__ == '__main__':
    prepared = prepare_experiment(sys.argv[1:], prepare_exp_dir=True)
    hparams = prepared['hparams']
    train_dataset, valid_dataset, test_dataset = prepared['datasets']
    model = prepared['model']

    # fit the model
    # with torch.autograd.detect_anomaly():
    # model.fit(
    #     hparams['model']['epoch_counter'],
    #     train_dataset,
    #     valid_dataset,
    #     train_loader_kwargs=hparams['train_dataloader_opts'],
    #     valid_loader_kwargs=hparams['valid_dataloader_opts'],
    # )
