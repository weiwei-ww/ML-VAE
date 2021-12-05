import logging
import functools

import speechbrain as sb
import torch.nn
import torch.nn.functional as F
from speechbrain.utils.data_utils import undo_padding
from speechbrain.nnet.losses import compute_masked_loss

from utils.metric_stats.boundary_metric_stats import BoundaryMetricStats
from models.md_model import MDModel
from utils.data_utils import undo_padding_tensor

logger = logging.getLogger(__name__)


class SBModel(MDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        self.stats_loggers['boundary_stats'] = BoundaryMetricStats()

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats, feat_lens = batch['feat']
        fa_boundary_seqs = batch['fa_boundary_seq'][0]

        # feature normalization
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalizer(feats, feat_lens, epoch=current_epoch)

        b_detector_out_dict = self.modules['boundary_detector'](feats, feat_lens, fa_boundary_seqs)

        predictions = b_detector_out_dict

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # compute loss
        losses = predictions['losses']
        loss = 0
        for loss_key in losses:
            weight_key = loss_key.replace('_loss', '_weight')
            weight = getattr(self.hparams, weight_key, 1)
            loss = weight * losses[loss_key]


        # get model outputs
        boundary_v_seqs = predictions['boundary_v']
        feat_lens = batch['feat'][1]
        fa_boundary_seqs, fa_boundary_seq_lens = batch['fa_boundary_seq']

        # unpadding
        boundary_v_seqs = undo_padding_tensor(boundary_v_seqs, feat_lens)

        # get top-k values of boundary_v, where k is number of phoneme boundaries
        pred_boundary_seqs = []
        for i, boundary_v in enumerate(boundary_v_seqs):
            num_segments = torch.sum(fa_boundary_seqs[i, :]).int()
            pred_boundary_seq = torch.zeros_like(boundary_v)
            _, indices = torch.topk(boundary_v, k=num_segments)
            pred_boundary_seq[indices] = 1
            pred_boundary_seqs.append(pred_boundary_seq)

        # unpad target boundary sequences
        targets = undo_padding_tensor(*batch['gt_boundary_seq'])

        self.stats_loggers['boundary_stats'].append(
            batch['id'],
            predictions=pred_boundary_seqs,
            targets=targets
        )

        return loss
