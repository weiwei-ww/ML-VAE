import logging

from torch.distributions import Categorical
import torch.nn
from speechbrain.nnet.losses import compute_masked_loss

from models.md_model import MDModel
from utils.data_utils import apply_lens_to_loss

logger = logging.getLogger(__name__)


class SBModel(MDModel):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats, feat_lens = batch['feat']

        # feature normalization
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalizer(feats, feat_lens, epoch=current_epoch)

        # RNN
        rnn_out = self.modules['rnn'](feats)[0]  # (B, T, C)

        # compute Pi
        pi_logits = self.modules['pi_fc'](rnn_out)  # (B, T, 2)
        dist = Categorical(logits=pi_logits)
        if self.modules.training:
            sampled_pi = dist.sample().float()  # (B, T)
        else:
            sampled_pi = torch.argmax(pi_logits, dim=-1).float()  # (B, T)
        sampled_pi = torch.stack([1 - sampled_pi, sampled_pi], dim=2)  # (B, T, 2)

        # VAE encoder and decoder
        encoder_out = self.modules['encoder'](rnn_out, sampled_pi)
        sampled_h = encoder_out['sampled_h']  # (B, T, C)
        decoder_out = self.modules['decoder'](sampled_h, feats)

        predictions = {
            'encoder_out': encoder_out,
            'decoder_out': decoder_out
        }

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        encoder_out = predictions['encoder_out']
        decoder_out = predictions['decoder_out']

        original_losses = encoder_out['losses']
        original_losses['recon_loss'] = decoder_out['losses']['recon_loss']

        feats, feat_lens = batch['feat']

        # compute losses
        losses = {}

        # compute mean loss using lens
        for key in original_losses:
            losses[key] = apply_lens_to_loss(original_losses[key], feat_lens)

        # compute and save total loss
        loss = super(SBModel, self).compute_and_save_losses(losses)

        return loss
