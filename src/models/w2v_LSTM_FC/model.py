import logging
import functools

import speechbrain as sb
import torch.nn
import torch.nn.functional as F
from speechbrain.utils.data_utils import undo_padding
from speechbrain.nnet.losses import compute_masked_loss

from utils.metric_stats.md_metric_stats import MDMetricStats
from models.md_model import MDModel

logger = logging.getLogger(__name__)


class SBModel(MDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        self.stats_loggers['flvl_md_stats'] = MDMetricStats()

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        wavs, wav_lens = batch['wav']
        feats = self.modules['wav2vec2'](wavs)  # (B, T, 1024)

        logits = self.modules['classifier'](feats)  # (B, T, 1)
        logits = logits.squeeze(dim=-1)  # (B, T)

        predictions = {
            'logits': logits
        }

        flvl_gt_md_lbl_seqs = batch['flvl_gt_md_lbl_seq'][0]
        if logits.shape[1] - flvl_gt_md_lbl_seqs.shape[1] > 0:
            raise ValueError(f'Inconsistent sequence lengths: {logits.shape[1]} != {flvl_gt_md_lbl_seqs.shape[1]}')

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        logits = predictions['logits']
        feat_lens = batch['feat'][1]

        # get ground truth
        flvl_gt_md_lbl_seqs = batch['flvl_gt_md_lbl_seq'][0]

        # change to the same length
        if logits.shape[1] - flvl_gt_md_lbl_seqs.shape[1] > 1:
            raise ValueError(f'Inconsistent sequence lengths: {logits.shape[1]} != {flvl_gt_md_lbl_seqs.shape[1]}')
        min_len = min(logits.shape[1], flvl_gt_md_lbl_seqs.shape[1])
        logits = logits[:, :min_len, ...]
        flvl_gt_md_lbl_seqs = flvl_gt_md_lbl_seqs[:, :min_len, ...]

        # compute BCE loss
        # pos_weight = torch.tensor([1, getattr(self.hparams, 'misp_weight')]).type(logits.dtype).to(logits.device)
        # loss_fn = functools.partial(F.binary_cross_entropy_with_logits, reduction='none', pos_weight=pos_weight)
        loss_fn = functools.partial(F.binary_cross_entropy_with_logits, reduction='none')
        targets = flvl_gt_md_lbl_seqs.type(logits.dtype)
        loss = compute_masked_loss(loss_fn, logits, targets, length=feat_lens)

        # compute MD metrics
        # get model output
        flvl_pred_md_lbl_seqs = torch.round(torch.sigmoid(logits))

        # unpad sequences
        flvl_gt_md_lbl_seqs = undo_padding(flvl_gt_md_lbl_seqs, feat_lens)
        flvl_pred_md_lbl_seqs = undo_padding(flvl_pred_md_lbl_seqs, feat_lens)

        self.stats_loggers['flvl_md_stats'].append(
            batch['id'],
            pred_md_lbl_seqs=flvl_pred_md_lbl_seqs,
            gt_md_lbl_seqs=flvl_gt_md_lbl_seqs
        )
        # print(round(loss.item() * 100, 3))

        return loss
