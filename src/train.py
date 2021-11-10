import sys
import os
import importlib
import functools

import torch
from hyperpyyaml import load_hyperpyyaml
from hyperpyyaml import dump_hyperpyyaml
from hyperpyyaml import resolve_references
from hyperpyyaml.core import recursive_update
import speechbrain as sb

from utils.phonemes import get_phoneme_set


class SBModel(sb.Brain):
    def __init__(self, label_encoder, **kwargs):
        super(SBModel, self).__init__(**kwargs)
        self.label_encoder = label_encoder

    def compute_forward(self, batch, stage):
        'Given an input batch it computes the phoneme probabilities.'
        batch = batch.to(self.device)
        wavs, wav_lens = batch['sig']
        # Adding optional augmentation when specified:
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, 'env_corrupt'):
                wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
            if hasattr(self.hparams, 'augmentation'):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalizer(feats, wav_lens)
        out = self.modules.crdnn(feats)
        out = self.modules.output(out)
        pout = self.hparams.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        'Given the network predictions and targets computed the CTC loss.'
        pout, pout_lens = predictions
        phns, phn_lens = batch['encoded_phonemes']

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, 'env_corrupt'):
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        loss = self.hparams.compute_cost(pout, phns, pout_lens, phn_lens, self.label_encoder.get_blank_index())
        self.ctc_metrics.append(batch.id, pout, phns, pout_lens, phn_lens)


        if stage != sb.Stage.TRAIN:
            sequence = sb.decoders.ctc_greedy_decode(
                pout, pout_lens, blank_id=self.label_encoder.get_blank_index()
            )
            self.per_metrics.append(
                ids=batch.id,
                predict=sequence,
                target=phns,
                target_len=phn_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

        return loss

    def on_stage_start(self, stage, epoch):
        'Gets called when a stage (either training, validation, test) starts.'
        self.ctc_metrics = self.hparams.ctc_stats(functools.partial(self.hparams.compute_cost,
                                                                    blank_index=self.label_encoder.get_blank_index(),
                                                                    reduction='batch'))

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        '''Gets called at the end of a stage.'''
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize('error_rate')

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.scheduler(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={'epoch': epoch, 'lr': old_lr},
                train_stats={'loss': self.train_loss},
                valid_stats={'loss': stage_loss, 'PER': per},
            )
            self.checkpointer.save_and_keep_only(
                meta={'PER': per}, min_keys=['PER'],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={'Epoch loaded': self.hparams.epoch_counter.current},
                test_stats={'loss': stage_loss, 'PER': per},
            )
            with open(self.hparams.wer_file, 'w') as w:
                w.write('CTC loss stats:\n')
                self.ctc_metrics.write_stats(w)
                w.write('\nPER stats:\n')
                self.per_metrics.write_stats(w)
                print('CTC and PER stats written to ', self.hparams.wer_file)


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
    # with open('dict.txt') as f:
    #     # phoneme_dict = [line.split()[0] for line in f.readlines() if not line.startswith('err')][3:]
    #     phoneme_dict = [line.split()[0] for line in f.readlines()][3:]
    phoneme_set = get_phoneme_set(hparams['prepare']['n_phonemes'])

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
