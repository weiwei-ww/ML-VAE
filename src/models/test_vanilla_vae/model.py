import logging

from speechbrain.nnet.losses import compute_masked_loss

from utils.metric_stats.loss_metric_stats import LossMetricStats
from models.md_model import MDModel
from utils.data_utils import apply_lens_to_loss

logger = logging.getLogger(__name__)


class SBModel(MDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        self.stats_loggers['kld_loss_stats'] = LossMetricStats('kld_loss')
        self.stats_loggers['recon_loss_stats'] = LossMetricStats('recon_loss')

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats, feat_lens = batch['feat']

        # feature normalization
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalizer(feats, feat_lens, epoch=current_epoch)

        encoder_out = self.modules['encoder'](feats)
        decoder_out = self.modules['decoder'](encoder_out['sampled_h'], feats)

        predictions = {
            'encoder_out': encoder_out,
            'decoder_out': decoder_out
        }

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        encoder_out = predictions['encoder_out']
        decoder_out = predictions['decoder_out']
        feats, feat_lens = batch['feat']

        # compute losses
        losses = {}

        # compute KLD loss
        losses['kld_loss'] = apply_lens_to_loss(encoder_out['loss'], feat_lens)

        # compute recon loss
        losses['recon_loss'] = apply_lens_to_loss(decoder_out['losses']['recon_loss'], feat_lens)

        # compute and save total loss
        loss = super(SBModel, self).compute_and_save_losses(losses)

        return loss
