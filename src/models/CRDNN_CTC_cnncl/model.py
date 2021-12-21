import logging
from pathlib import Path

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.nnet.losses import ctc_loss

import utils.alignment
from utils.metric_stats.md_metric_stats import MDMetricStats
from models.CRDNN_CTC.model import SBModel as CRDNN_CTC

logger = logging.getLogger(__name__)


class SBModel(CRDNN_CTC):
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

        cnncl_phns, cnncl_phn_lens = batch['gt_cnncl_seq']
        loss = ctc_loss(pout, cnncl_phns, pout_lens, cnncl_phn_lens, self.label_encoder.get_blank_index())
        return loss

