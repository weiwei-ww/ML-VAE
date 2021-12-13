import logging
from enum import Enum, auto

from torch.distributions import Categorical
import torch.nn
from torch.nn.utils.rnn import pad_sequence

import speechbrain as sb
from speechbrain.nnet.losses import compute_masked_loss
from speechbrain.utils.data_utils import undo_padding

from models.MD_VAE.model import SBModel as MD_VAE
from models.MD_VAE.model import Target
from utils.metric_stats.loss_metric_stats import LossMetricStats
from utils.metric_stats.md_metric_stats import MDMetricStats
from utils.metric_stats.boundary_metric_stats import BoundaryMetricStats
from utils.data_utils import apply_lens_to_loss, undo_padding_tensor
from utils.decode_utils import decode_plvl_md_lbl_seqs_full as decode_plvl_md_lbl_seqs

logger = logging.getLogger(__name__)


class SBModel(MD_VAE):
    def compute_forward(self, batch, stage):
        if not hasattr(self, 'target'):
            raise ValueError('target is not defined')

        batch = batch.to(self.device)
        # feats, feat_lens = batch['feat']

        wavs, wav_lens = batch['wav']
        feats = self.modules['wav2vec2'](wavs)  # (B, T, 1024)
        feat_lens = batch['feat'][1]

        # handle inconsistent lengths
        ori_feats = batch['feat'][0]
        shape_diff = feats.shape[1] - ori_feats.shape[1]
        assert -2 <= shape_diff <= 0, f'shape_diff = {shape_diff}'
        if shape_diff < 0:
            zeros = torch.zeros(feats.shape[0], abs(shape_diff), feats.shape[2]).float().to(feats.device)
            feats = torch.concat([feats, zeros], dim=1)
        assert feats.shape[1] == ori_feats.shape[1]

        # initialize predictions
        fake_loss = torch.zeros_like(feats)
        fake_loss.requires_grad = True
        predictions = {'losses': {'loss': fake_loss}}

        # phoneme recognizer
        if self.target in [Target.PHN_RECOG, Target.VAE, Target.TEST]:
            plvl_cnnl_phn_seqs, plvl_cnnl_phn_seq_lens = batch['gt_cnncl_seq']
            fa_boundary_seqs = batch['fa_boundary_seq'][0]
            phn_recog_out_dict = self.modules['phoneme_recognizer'](
                feats, feat_lens, plvl_cnnl_phn_seqs, plvl_cnnl_phn_seq_lens, fa_boundary_seqs
            )
            predictions['phn_recog_out'] = phn_recog_out_dict['out']

            losses = phn_recog_out_dict['losses']
            if self.target != Target.PHN_RECOG:
                for key in losses:
                    losses[key] = losses[key].detach()
            predictions['losses'].update(losses)

            loss_keys = predictions['losses'].keys()
            logger.info(f'losses for phn_recog: {loss_keys}')


        # boundary detector
        if self.target in [Target.B_DETECTOR, Target.VAE, Target.TEST]:
            fa_boundary_seqs = batch['fa_boundary_seq'][0]
            b_detector_out = self.modules['boundary_detector'](feats, feat_lens, fa_boundary_seqs)
            predictions['boundary_v'] = b_detector_out['boundary_v']

            losses = b_detector_out['losses']
            if self.target != Target.B_DETECTOR:
                for key in losses:
                    losses[key] = losses[key].detach()
            predictions['losses'].update(losses)

        # VAE
        if self.target in [Target.VAE, Target.TEST]:
            # feat FC
            feat_fc_out = self.modules['feat_fc'](feats)

            # FC for phoneme recognizer output
            phn_recog_fc_out = self.modules['phn_recog_fc'](phn_recog_out_dict['out'].detach())

            # concatenation
            rnn_in = torch.cat([feat_fc_out, phn_recog_fc_out], dim=-1)
            rnn_in = self.modules['concat_fc'](rnn_in)

            # RNN
            rnn_out = self.modules['rnn'](rnn_in)[0]  # (B, T, C)

            # compute Pi
            pi_logits = self.modules['pi_fc'](rnn_out)  # (B, T, 2)
            predictions['pi_logits'] = pi_logits

            dist = Categorical(logits=pi_logits)
            if self.modules.training:
                sampled_pi = dist.sample().float()  # (B, T)
            else:
                sampled_pi = torch.argmax(pi_logits, dim=-1).float()  # (B, T)
            sampled_pi = torch.stack([1 - sampled_pi, sampled_pi], dim=2)  # (B, T, 2)
            predictions['sampled_pi'] = sampled_pi

            # compute Pi NLL loss
            plvl_cnnl_seqs, plvl_cnnl_seq_lens = batch['gt_cnncl_seq']
            weight = getattr(self.hparams, 'dec_weight', 1.0)
            _, decoded_flvl_md_lvl_seqs, _ = decode_plvl_md_lbl_seqs(
                predictions,
                utt_ids=batch['id'],
                feat_lens=feat_lens,
                plvl_cnnl_seqs=plvl_cnnl_seqs,
                plvl_cnnl_seq_lens=plvl_cnnl_seq_lens,
                prior=batch['prior'][0][0],
                weight=weight
            )
            decoded_flvl_md_lvl_seqs = \
                [torch.tensor(seq).float().to(sampled_pi.device) for seq in decoded_flvl_md_lvl_seqs]
            decoded_flvl_md_lvl_seqs = pad_sequence(decoded_flvl_md_lvl_seqs, batch_first=True)
            pi_nll_loss = -dist.log_prob(decoded_flvl_md_lvl_seqs)
            predictions['losses']['pi_nll_loss'] = pi_nll_loss

            # VAE encoder and decoder
            encoder_out_dict = self.modules['encoder'](rnn_out, sampled_pi)
            predictions['losses'].update(encoder_out_dict['losses'])

            sampled_h = encoder_out_dict['sampled_h']  # (B, T, C)
            decoder_out_dict = self.modules['decoder'](sampled_h, feats)
            predictions['losses'].update(decoder_out_dict['losses'])

        return predictions
