import logging
from pathlib import Path

import speechbrain as sb
import speechbrain.utils.data_utils
from speechbrain.utils.metric_stats import ErrorRateStats

import utils.alignment
import utils.md_scoring
from utils.md_metric_stats import MDMetricStats
from models.md_model import MDModel

logger = logging.getLogger(__name__)


class SBModel(MDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        per_stats = ErrorRateStats()
        plvl_md_stats = MDMetricStats()

        self.stats_loggers['per_stats'] = per_stats
        self.stats_loggers['plvl_md_stats'] = plvl_md_stats

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

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        pout = predictions['pout']
        pout_lens = predictions['wav_lens']

        # compute CTC loss
        phns, phn_lens = batch['gt_phn_seq']
        loss = self.hparams.compute_cost(pout, phns, pout_lens, phn_lens, self.label_encoder.get_blank_index())

        # compute PER
        sequences = sb.decoders.ctc_greedy_decode(
            pout, pout_lens, blank_id=self.label_encoder.get_blank_index()
        )
        self.stats_loggers['per_stats'].append(
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

        self.stats_loggers['plvl_md_stats'].append(
            batch['id'],
            batch_pred_phn_seqs=ali_pred_phn_seqs,
            batch_gt_phn_seqs=ali_gt_phn_seqs,
            batch_gt_cnncl_seqs=ali_gt_cnncl_seqs
        )

        return loss
