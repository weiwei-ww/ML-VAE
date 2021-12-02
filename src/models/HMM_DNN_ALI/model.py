import logging
import functools

import speechbrain as sb
import torch.nn
import torch.nn.functional as F
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.utils.data_utils import undo_padding
from speechbrain.nnet.losses import compute_masked_loss

from utils.md_metric_stats import MDMetricStats
from models.md_model import MDModel

logger = logging.getLogger(__name__)


class SBModel(MDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        self.training_type = self.hparams.init_training_type
        self.stats_loggers['accuracy_stats'] = MetricStats(
            metric=self.hparams.aligner.calc_accuracy
        )

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats, feat_lens = batch['feat']

        # Adding augmentation when specified:
        # if stage == sb.Stage.TRAIN:
        #     if hasattr(self.modules, 'env_corrupt'):
        #         wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
        #         wavs = torch.cat([wavs, wavs_noise], dim=0)
        #         wav_lens = torch.cat([wav_lens, wav_lens])
        #     if hasattr(self.hparams, 'augmentation'):
        #         wavs = self.hparams.augmentation(wavs, wav_lens)
        #
        # feats = self.hparams.compute_features(wavs)
        if hasattr(self.hparams, 'normalize'):
            feats = self.modules.normalize(feats, feat_lens)
        out = self.modules.model(feats)
        out = self.modules.output(out)
        out = out - out.mean(1).unsqueeze(1)
        pout = self.hparams.log_softmax(out)
        return pout, feat_lens

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        pout, pout_lens = predictions
        ids = batch['id']
        phns, phn_lens = batch['gt_cnncl_seq']
        phn_ends, _ = batch['gt_phn_end_seq']

        # if stage == sb.Stage.TRAIN and hasattr(self.modules, 'env_corrupt'):
        #     phns = torch.cat([phns, phns], dim=0)
        #     phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)
        phns_orig = sb.utils.data_utils.undo_padding(phns, phn_lens)
        phns = self.hparams.aligner.expand_phns_by_states_per_phoneme(phns, phn_lens)

        phns = phns.int()

        if self.training_type == 'forward':
            forward_scores = self.hparams.aligner(
                pout, pout_lens, phns, phn_lens, 'forward'
            )
            loss = -forward_scores

        elif self.training_type == 'ctc':
            loss = self.hparams.compute_cost_ctc(
                pout, phns, pout_lens, phn_lens
            )
        elif self.training_type == 'viterbi':
            prev_alignments = self.hparams.aligner.get_prev_alignments(
                ids, pout, pout_lens, phns, phn_lens
            )
            prev_alignments = prev_alignments.to(self.hparams.device)
            loss = self.hparams.compute_cost_nll(pout, prev_alignments)

        viterbi_scores, alignments = self.hparams.aligner(
            pout, pout_lens, phns, phn_lens, 'viterbi'
        )

        if self.training_type in ['viterbi', 'forward']:
            self.hparams.aligner.store_alignments(ids, alignments)

        # if stage != sb.Stage.TRAIN:
        self.stats_loggers['accuracy_stats'].append(ids, alignments, phn_ends, phns_orig)

        return loss
