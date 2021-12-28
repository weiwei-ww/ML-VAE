import logging
from pathlib import Path

import torch
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.nnet.losses import ctc_loss

import utils.alignment
from utils.metric_stats.md_metric_stats import MDMetricStats
from utils.data_utils import undo_padding_tensor
import models.CRDNN_CTC_cnncl.model as CRDNN_CTC_cnncl

logger = logging.getLogger(__name__)


class SBModel(CRDNN_CTC_cnncl.SBModel):
    # def on_stage_start(self, stage, epoch=None):
    #     super(SBModel, self).on_stage_start(stage, epoch)
    #     self.saved_pouts = {}  # save pout to write to file


    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        wavs, wav_lens = batch['wav']
        w2v_feats = self.modules['wav2vec2'](wavs)  # (B, T, 1024)

        out = self.modules.crdnn(w2v_feats)
        out = self.modules.output(out)
        pout = self.hparams.log_softmax(out)

        predictions = {
            'pout': pout,
        }

        # feats, feat_lens = batch['feat']
        # unpadded_pouts = undo_padding_tensor(pout, feat_lens)
        # unpadded_feats = undo_padding_tensor(feats, feat_lens)
        # ids = batch['id']
        # for id, unpadded_pout, feat in zip(ids, unpadded_pouts, unpadded_feats):
        #     self.saved_pouts[id] = unpadded_pout.detach()

        return predictions

    # def on_stage_end(self, stage, stage_loss, epoch=None):
    #     super(SBModel, self).on_stage_end(stage, stage_loss, epoch)
    #
    #     if stage == sb.Stage.TEST:
    #         saved_pouts_path = Path(self.hparams.hparams) / 'saved_pouts.pt'
    #         if saved_pouts_path.exists():
    #             saved_pouts = torch.load(saved_pouts_path)
    #         else:
    #             saved_pouts = {}
    #         for key in self.saved_pouts:
    #             if key in saved_pouts:
    #                 raise ValueError(f'duplicate key {key}')
    #         saved_pouts.update(self.saved_pouts)
    #         torch.save(saved_pouts, saved_pouts_path)