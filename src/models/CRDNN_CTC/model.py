import functools

import torch
import speechbrain as sb

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
