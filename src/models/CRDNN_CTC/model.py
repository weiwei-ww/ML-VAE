import functools
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
import speechbrain as sb
import speechbrain.utils.data_utils
from speechbrain.utils.metric_stats import MetricStats, ErrorRateStats

import utils.alignment
import utils.md_scoring
from utils.md_metric_stats import MDMetricStats
from models.md_model import MDModel

logger = logging.getLogger(__name__)


class SBModel(MDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        # ctc_stats = MetricStats(functools.partial(self.hparams.compute_cost,
        #                                                blank_index=self.label_encoder.get_blank_index(),
        #                                                reduction='batch'))
        per_stats = ErrorRateStats()
        md_stats = MDMetricStats()

        # self.stats['ctc_stats'] = ctc_stats
        self.stats['per_stats'] = per_stats
        self.stats['md_stats'] = md_stats

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

        predictions = {
            'pout': pout,
            'wav_lens': wav_lens
        }

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        pout = predictions['pout']
        pout_lens = predictions['wav_lens']

        # compute CTC loss
        phns, phn_lens = batch['gt_phn_seq']
        loss = self.hparams.compute_cost(pout, phns, pout_lens, phn_lens, self.label_encoder.get_blank_index())
        # self.stats['ctc_stats'].append(batch['id'], pout, phns, pout_lens, phn_lens)

        # compute PER
        sequences = sb.decoders.ctc_greedy_decode(
            pout, pout_lens, blank_id=self.label_encoder.get_blank_index()
        )
        self.stats['per_stats'].append(
            ids=batch.id,
            predict=sequences,
            target=phns,
            target_len=phn_lens,
            ind2lab=self.label_encoder.decode_ndim,
        )

        # compute MD metrics
        pred_phns = sb.decoders.ctc_greedy_decode(
            pout, pout_lens, blank_id=self.label_encoder.get_blank_index()
        )

        # unpad sequences
        gt_phn_seqs, gt_phn_seq_lens = batch['gt_phn_seq']
        gt_cnncl_seqs, gt_cnncl_seq_lens = batch['gt_cnncl_seq']

        gt_phn_seqs = sb.utils.data_utils.undo_padding(gt_phn_seqs, gt_phn_seq_lens)
        gt_cnncl_seqs = sb.utils.data_utils.undo_padding(gt_cnncl_seqs, gt_cnncl_seq_lens)

        # align sequences
        ali_pred_phn_seqs, ali_gt_phn_seqs, ali_gt_cnncl_seqs = \
            utils.alignment.batch_align_sequences(pred_phns, gt_phn_seqs, gt_cnncl_seqs)

        self.stats['md_stats'].append(
            batch['id'],
            batch_pred_phn_seqs=ali_pred_phn_seqs,
            batch_gt_phn_seqs=ali_gt_phn_seqs,
            batch_gt_cnncl_seqs=ali_gt_cnncl_seqs
        )

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_name = str(stage).split('.')[1].lower()

        if stage == sb.Stage.TRAIN or stage == sb.Stage.VALID:
            # log stats
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            per = self.stats['per_stats'].summarize('error_rate')
            train_logger_stats = {
                'stats_meta': {'stage': stage_name, 'epoch': epoch, 'lr': lr},
                f'{stage_name}_stats': {'loss': stage_loss, 'PER': per}
            }
            self.hparams.train_logger.log_stats(**train_logger_stats)

            # TB log
            tb_metrics = {}
            tb_metrics['loss'] = stage_loss
            tb_metrics['PER'] = per
            tb_metrics['lr'] = lr

            md_summary = self.stats['md_stats'].summarize()
            for key in md_summary:
                tb_metrics[f'MD_{key}'] = md_summary[key]

            # tensorboard logging
            for key in tb_metrics:
                self.tb_writer.add_scalar(f'{key}/{stage_name}', tb_metrics[key], global_step=epoch)

            # save checkpoint after the VALID stage
            if stage == sb.Stage.VALID:
                self.checkpointer.save_and_keep_only(
                    meta={'PER': per}, min_keys=['PER'],
                )

        if stage == sb.Stage.TEST:
            # log test results
            per = self.stats['per_stats'].summarize('error_rate')
            logger.info(f'Best epoch: {self.hparams.epoch_counter.current}, loss: {stage_loss}, PER: {per}')

            with open(self.hparams.wer_file, 'w') as w:
                # w.write('CTC loss stats:\n')
                # self.ctc_stats.write_stats(w)
                w.write('PER stats:\n')
                self.stats['per_stats'].write_stats(w)
                logger.info(f'PER stats written to {self.hparams.wer_file}')

            if hasattr(self, 'md_stats'):
                with open(self.hparams.md_metrics_file, 'w') as w:
                    md_summary = self.stats['md_stats'].summarize()
                    for key in self.stats['md_stats'].summarize():
                        w.write(f'MD_{key} = {md_summary[key]}\n')
                logger.info(f'MD metrics written to {self.hparams.md_metrics_file}')

