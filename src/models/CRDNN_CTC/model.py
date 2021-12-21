import logging
from pathlib import Path

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.nnet.losses import ctc_loss

import utils.alignment
from utils.metric_stats.md_metric_stats import MDMetricStats
from models.md_model import MDModel

logger = logging.getLogger(__name__)


class SBModel(MDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        self.stats_loggers['phn_per_stats'] = ErrorRateStats()
        self.stats_loggers['cnncl_per_stats'] = ErrorRateStats()
        self.stats_loggers['plvl_md_stats'] = MDMetricStats()

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch['wav']
        feats, feat_lens = batch['feat']

        # feature normalization
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalizer(feats, feat_lens, epoch=current_epoch)

        out = self.modules.crdnn(feats)
        out = self.modules.output(out)
        pout = self.hparams.log_softmax(out)

        predictions = {
            'pout': pout,
            'wav_lens': wav_lens
        }

        return predictions

    def compute_loss(self, predictions, batch):
        # get model outputs
        pout = predictions['pout']
        pout_lens = batch['feat'][1]

        phns, phn_lens = batch['gt_phn_seq']
        loss = ctc_loss(pout, phns, pout_lens, phn_lens, self.label_encoder.get_blank_index())
        return loss

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        pout = predictions['pout']
        pout_lens = batch['feat'][1]

        # compute CTC loss
        loss = self.compute_loss(predictions, batch)

        # compute PER
        pred_phns = sb.decoders.ctc_greedy_decode(
            pout, pout_lens, blank_id=self.label_encoder.get_blank_index()
        )
        phns, phn_lens = batch['gt_phn_seq']
        self.stats_loggers['phn_per_stats'].append(
            ids=batch.id,
            predict=pred_phns,
            target=phns,
            target_len=phn_lens,
            ind2lab=self.label_encoder.decode_ndim,
        )
        cnncls, cnncl_lens = batch['gt_cnncl_seq']
        self.stats_loggers['cnncl_per_stats'].append(
            ids=batch.id,
            predict=pred_phns,
            target=cnncls,
            target_len=cnncl_lens,
            ind2lab=self.label_encoder.decode_ndim,
        )

        # compute MD metrics
        pred_phns = sb.decoders.ctc_greedy_decode(
            pout, pout_lens, blank_id=self.label_encoder.get_blank_index()
        )

        # unpad sequences
        gt_phn_seqs, gt_phn_seq_lens = batch['gt_phn_seq']
        gt_cnncl_seqs, gt_cnncl_seq_lens = batch['gt_cnncl_seq']

        gt_phn_seqs = undo_padding(gt_phn_seqs, gt_phn_seq_lens)
        gt_cnncl_seqs = undo_padding(gt_cnncl_seqs, gt_cnncl_seq_lens)

        # align sequences
        ali_pred_phn_seqs, ali_gt_phn_seqs, ali_gt_cnncl_seqs = \
            utils.alignment.batch_align_sequences(pred_phns, gt_phn_seqs, gt_cnncl_seqs)

        self.stats_loggers['plvl_md_stats'].append(
            batch['id'],
            pred_phn_seqs=ali_pred_phn_seqs,
            gt_phn_seqs=ali_gt_phn_seqs,
            gt_cnncl_seqs=ali_gt_cnncl_seqs
        )

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        super(SBModel, self).on_stage_end(stage, stage_loss, epoch)

        if stage == sb.Stage.TEST:
            output_path = Path(self.hparams.output_dir) / 'test_output' / 'md_result_seqs.txt'
            self.stats_loggers['plvl_md_stats'].write_seqs_to_file(output_path, self.label_encoder)
