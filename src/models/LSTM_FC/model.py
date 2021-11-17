import logging
import functools

import speechbrain as sb
import torch.nn
import torch.nn.functional as F
from speechbrain.utils.data_utils import undo_padding
from speechbrain.nnet.losses import compute_masked_loss

from utils.md_metric_stats import MDMetricStats
from models.md_model import MDModel

logger = logging.getLogger(__name__)


class SBModel(MDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        self.stats_loggers['flvl_md_stats'] = MDMetricStats()

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        if stage == sb.Stage.TRAIN:
            feats, feat_lens = batch['aug_feat']
        else:
            feats, feat_lens = batch['feat']


        out = self.modules['lstm'](feats)[0]
        out = self.modules['fc'](out)
        out = self.modules['output'](out)

        predictions = {
            'out': out,
        }

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        out = predictions['out']
        if stage == sb.Stage.TRAIN:
            feat_lens = batch['aug_feat'][1]
            flvl_gt_md_lbl_seqs, flvl_gt_md_lbl_seq_lens = batch['aug_flvl_gt_md_lbl_seq']
        else:
            feat_lens = batch['feat'][1]
            flvl_gt_md_lbl_seqs, flvl_gt_md_lbl_seq_lens = batch['flvl_gt_md_lbl_seq']

        # compute BCE loss
        pos_weight = torch.tensor([1, 10]).type(out.dtype).to(out.device)
        loss_fn = functools.partial(F.binary_cross_entropy_with_logits, reduction='none', pos_weight=pos_weight)
        targets = torch.stack([1 - flvl_gt_md_lbl_seqs, flvl_gt_md_lbl_seqs], dim=-1).type(out.dtype)
        loss = compute_masked_loss(loss_fn, out, targets)

        # compute MD metrics
        # get model output
        flvl_pred_md_lbl_seqs = torch.argmax(out, dim=-1)

        # unpad sequences
        flvl_gt_md_lbl_seqs = undo_padding(flvl_gt_md_lbl_seqs, flvl_gt_md_lbl_seq_lens)
        flvl_pred_md_lbl_seqs = undo_padding(flvl_pred_md_lbl_seqs, feat_lens)

        self.stats_loggers['flvl_md_stats'].append(
            batch['id'],
            batch_pred_md_lbl_seqs=flvl_pred_md_lbl_seqs,
            batch_gt_md_lbl_seqs=flvl_gt_md_lbl_seqs
        )

        return loss
