import sys
import logging

from prepare import prepare_experiment

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    prepared = prepare_experiment(sys.argv[1:], prepare_exp_dir=False)
    hparams = prepared['hparams']
    train_dataset, valid_dataset, test_dataset = prepared['datasets']
    model = prepared['model']

    # Test
    model.evaluate(
        test_dataset,
        min_key='PER',
        test_loader_kwargs=hparams['test_dataloader_opts'],
    )
