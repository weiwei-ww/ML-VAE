import logging
import functools

import speechbrain as sb
import torch.nn
import torch.nn.functional as F
from speechbrain.utils.data_utils import undo_padding
from speechbrain.nnet.losses import compute_masked_loss

from utils.metric_stats.phn_acc_metric_stats import PhnAccMetricStats
from models.md_model import MDModel
from utils.data_utils import undo_padding_tensor

logger = logging.getLogger(__name__)


class SBModel(MDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        self.stats_loggers['phn_acc_stats'] = PhnAccMetricStats()

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats, feat_lens = batch['feat']

        # feature normalization
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalizer(feats, feat_lens, epoch=current_epoch)

        out = self.modules['phoneme_recognizer'](feats)

        predictions = {
            'out': out,
        }

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        out = predictions['out']
        feat_lens = batch['feat'][1]
        plvl_cnnl_phn_seqs, plvl_cnnl_phn_seq_lens = batch['gt_cnncl_seq']
        boundary_seqs = batch['gt_boundary_seq'][0]

        # compute BCE loss
        loss = self.modules['phoneme_recognizer'].calculate_losses(
            out, feat_lens, plvl_cnnl_phn_seqs, plvl_cnnl_phn_seq_lens, boundary_seqs
        )['phoneme_bce_loss']

        # unpad sequences and compute metrics
        out = undo_padding_tensor(out, feat_lens)
        flvl_gt_cnncl_seqs = undo_padding_tensor(*batch['flvl_gt_cnncl_seq'])
        plvl_gt_cnncl_seqs = undo_padding_tensor(*batch['gt_cnncl_seq'])
        boundary_seqs = undo_padding_tensor(*batch['gt_boundary_seq'])

        self.stats_loggers['phn_acc_stats'].append(
            batch['id'],
            predictions=out,
            flvl_targets=flvl_gt_cnncl_seqs,
            plvl_targets=plvl_gt_cnncl_seqs,
            boundary_seqs=boundary_seqs
        )

        return loss
