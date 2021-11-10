import sys
import importlib

import torch
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb

from utils.phonemes import get_phoneme_set



def data_io_prep(hparams):
    'Creates the datasets and their data processing pipelines.'

    # 1. Define datasets:
    def dataset_prep(hparams, set_name):
        dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams['prepare'][f'{set_name}_json_path'])

        if hparams['sorting'] in ['ascending', 'descending']:
            reverse = True if hparams['sorting'] == 'descending' else False
            dataset = dataset.filtered_sorted(sort_key='duration', reverse=reverse)
            hparams['train_dataloader_opts']['shuffle'] = False

        return dataset

    train_dataset = dataset_prep(hparams, 'train')
    valid_dataset = dataset_prep(hparams, 'valid')
    test_dataset = dataset_prep(hparams, 'test')
    datasets = [train_dataset, valid_dataset, test_dataset]

    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    sb.dataio.dataset.add_dynamic_item(datasets, sb.dataio.dataio.read_audio, takes='wav_path', provides='sig')

    # 3. Define text pipeline:
    sb.dataio.dataset.add_dynamic_item(datasets, lambda p: label_encoder.encode_sequence_torch(p), takes='phonemes', provides='encoded_phonemes')

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ['id', 'sig', 'encoded_phonemes'])

    # 5. Fit encoder:
    phoneme_set = hparams['prepare']['phoneme_set_handler'].get_phoneme_set()

    label_encoder.update_from_iterable(phoneme_set, sequence_input=False)
    label_encoder.insert_blank(index=hparams['blank_index'])

    return train_dataset, valid_dataset, test_dataset, label_encoder


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
