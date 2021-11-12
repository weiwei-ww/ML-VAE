import sys
import importlib

from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb

from utils.data_io import data_io_prep


if __name__ == '__main__':

    # parse command line args
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # load hparams
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)


    # json file preparation
    dataset_name = hparams['dataset']
    importlib.import_module(f'datasets.{dataset_name}.prepare').prepare(**hparams['prepare'])

    # Load parsed dataset
    train_dataset, valid_dataset, test_dataset, label_encoder = data_io_prep(hparams)

    # initialize model
    model_class = hparams['model_class']
    SBModel = importlib.import_module(f'models.{model_class}.model').SBModel
    asr_brain = SBModel(
        label_encoder=label_encoder,
        modules=hparams['model']['modules'],
        opt_class=hparams['model']['opt_class'],
        hparams=hparams['model'],
        run_opts=run_opts,
        checkpointer=hparams['model']['checkpointer'],
    )

    # create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams['output_dir'],
        hyperparams_to_save=hparams_file,
        overrides=overrides
    )


    # fit the model
    asr_brain.fit(
        hparams['model']['epoch_counter'],
        train_dataset,
        valid_dataset,
        train_loader_kwargs=hparams['train_dataloader_opts'],
        valid_loader_kwargs=hparams['valid_dataloader_opts'],
    )

    # Test
    asr_brain.evaluate(
        test_dataset,
        min_key='PER',
        test_loader_kwargs=hparams['test_dataloader_opts'],
    )
