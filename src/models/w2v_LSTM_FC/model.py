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

    def init_optimizers(self):
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(self.modules.wav2vec2.parameters())
        self.adam_optimizer = self.hparams.adam_opt_class(self.modules.classifier.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable('wav2vec_opt', self.wav2vec_optimizer)
            self.checkpointer.add_recoverable('adam_opt', self.adam_optimizer)

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        wavs, wav_lens = batch['wav']
        feats = self.modules['wav2vec2'](wavs)  # (B, T, 1024)

        logits = self.modules['classifier'](feats)  # (B, T, 1)
        logits = logits.squeeze(dim=-1)  # (B, T)

        predictions = {
            'logits': logits,
            'logit_lens': wav_lens
        }

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        logits = predictions['logits']

        # get ground truth
        flvl_gt_md_lbl_seqs, flvl_gt_md_lbl_seq_lens = batch['flvl_gt_md_lbl_seq']

        # compute BCE loss
        pos_weight = torch.tensor([1, getattr(self.hparams, 'misp_weight')]).type(logits.dtype).to(logits.device)
        loss_fn = functools.partial(F.binary_cross_entropy_with_logits, reduction='none', pos_weight=pos_weight)
        loss = compute_masked_loss(loss_fn, logits, flvl_gt_md_lbl_seqs)

        # compute MD metrics
        # get model output
        logit_lens = predictions['logit_lens']
        flvl_pred_md_lbl_seqs = torch.round(torch.sigmoid(logits))

        # unpad sequences
        flvl_gt_md_lbl_seqs = undo_padding(flvl_gt_md_lbl_seqs, flvl_gt_md_lbl_seq_lens)
        flvl_pred_md_lbl_seqs = undo_padding(flvl_pred_md_lbl_seqs, logit_lens)

        self.stats_loggers['flvl_md_stats'].append(
            batch['id'],
            batch_pred_md_lbl_seqs=flvl_pred_md_lbl_seqs,
            batch_gt_md_lbl_seqs=flvl_gt_md_lbl_seqs
        )

        return loss
