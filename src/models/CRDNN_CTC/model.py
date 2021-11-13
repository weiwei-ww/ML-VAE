import functools


import torch
from torch.utils.tensorboard import SummaryWriter
import speechbrain as sb
import speechbrain.utils.data_utils

import utils.alignment
import utils.md_scoring


class SBModel(sb.Brain):
    def __init__(self, label_encoder, **kwargs):
        super(SBModel, self).__init__(**kwargs)
        self.label_encoder = label_encoder

    def on_stage_start(self, stage, epoch):
        'Gets called when a stage (either training, validation, test) starts.'
        self.ctc_stats = self.hparams.ctc_stats(functools.partial(self.hparams.compute_cost,
                                                                    blank_index=self.label_encoder.get_blank_index(),
                                                                    reduction='batch'))
        self.per_metrics = self.hparams.per_stats()

        self.md_stats = self.hparams.md_stats()

    def compute_forward(self, batch, stage):
        'Given an input batch it computes the phoneme probabilities.'
        batch = batch.to(self.device)
        if stage == sb.Stage.TRAIN:
            wavs, wav_lens = batch['aug_wav']
            feats, feat_lens = batch['aug_feat']
        else:
            wavs, wav_lens = batch['wav']
            feats, feat_lens = batch['feat']
        out = self.modules.crdnn(feats)
        out = self.modules.output(out)
        pout = self.hparams.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        'Given the network predictions and targets computed the CTC loss.'
        pout, pout_lens = predictions
        phns, phn_lens = batch['gt_phn_seq']

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, 'env_corrupt'):
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        loss = self.hparams.compute_cost(pout, phns, pout_lens, phn_lens, self.label_encoder.get_blank_index())
        self.ctc_stats.append(batch['id'], pout, phns, pout_lens, phn_lens)


        # if stage != sb.Stage.TRAIN:
        sequences = sb.decoders.ctc_greedy_decode(
            pout, pout_lens, blank_id=self.label_encoder.get_blank_index()
        )
        self.per_metrics.append(
            ids=batch.id,
            predict=sequences,
            target=phns,
            target_len=phn_lens,
            ind2lab=self.label_encoder.decode_ndim,
        )

        # pred_phns = sb.decoders.ctc_greedy_decode(
        #     pout, pout_lens, blank_id=self.label_encoder.get_blank_index()
        # )
        #
        # # unpad sequences
        # gt_phn_seqs, gt_phn_seq_lens = batch['gt_phn_seq']
        # gt_cnncl_seqs, gt_cnncl_seq_lens = batch['gt_cnncl_seq']
        #
        # gt_phn_seqs = sb.utils.data_utils.undo_padding(gt_phn_seqs, gt_phn_seq_lens)
        # gt_cnncl_seqs = sb.utils.data_utils.undo_padding(gt_cnncl_seqs, gt_cnncl_seq_lens)
        #
        # # align sequences
        # ali_pred_phn_seqs, ali_gt_phn_seqs, ali_gt_cnncl_seqs = \
        #     utils.alignment.batch_align_sequences(pred_phns, gt_phn_seqs, gt_cnncl_seqs)
        #
        # self.md_stats.append(
        #     batch['id'],
        #     batch_pred_phn_seqs=ali_pred_phn_seqs,
        #     batch_gt_phn_seqs=ali_gt_phn_seqs,
        #     batch_gt_cnncl_seqs=ali_gt_cnncl_seqs
        # )
        # print(self.md_stats.summarize())

        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        '''Gets called at the end of a stage.'''
        loss = stage_loss
        per = self.per_metrics.summarize('error_rate')

        if stage == sb.Stage.TRAIN:
            self.train_loss = loss

        # tensorboard logging
        if stage != sb.Stage.TEST:
            stage_name = str(stage).split('.')[1].lower()
            self.hparams.tb_writer.add_scalar(f'loss/{stage_name}', loss, global_step=epoch)
            self.hparams.tb_writer.add_scalar(f'PER/{stage_name}', per, global_step=epoch)


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
                self.ctc_stats.write_stats(w)
                w.write('\nPER stats:\n')
                self.per_metrics.write_stats(w)
                print('CTC and PER stats written to ', self.hparams.wer_file)

