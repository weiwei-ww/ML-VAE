import logging
from enum import Enum, auto

from torch.distributions import Categorical
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import speechbrain as sb
from speechbrain.nnet.losses import compute_masked_loss
from speechbrain.utils.data_utils import undo_padding

from models.md_model import MDModel
from utils.metric_stats.loss_metric_stats import LossMetricStats
from utils.metric_stats.md_metric_stats import MDMetricStats
from utils.metric_stats.boundary_metric_stats import BoundaryMetricStats
from utils.data_utils import apply_lens_to_loss, undo_padding_tensor
from utils.decode_utils import decode_plvl_md_lbl_seqs_full as decode_plvl_md_lbl_seqs

logger = logging.getLogger(__name__)


class Target(Enum):
    PHN_RECOG = auto()
    B_DETECTOR = auto()
    VAE = auto()
    TEST = auto()


class SBModel(MDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)

        # define the target
        if stage == sb.Stage.TRAIN or stage == sb.Stage.VALID:
            assert epoch is not None
            train_targets = [Target.PHN_RECOG, Target.B_DETECTOR, Target.VAE]
            self.target = train_targets[(epoch - 1) % 3]
        elif stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            self.target = Target.TEST
        else:
            raise ValueError(f'invalid stage {stage}')
        logger.info(f'Epoch {epoch}, stage {stage}: target is {self.target}')

        # initialize metric stats
        self.stats_loggers = {}
        # only compute metrics when on validation set with VAE target, or test set
        if self.to_run_evaluation(stage):

            # initialize metric stats for losses
            for loss_key in self.hparams.metric_keys:
                if loss_key.endswith('_loss'):
                    stats_key = loss_key + '_stats'
                    self.stats_loggers[stats_key] = LossMetricStats(loss_key)

            self.stats_loggers['plvl_md_stats'] = MDMetricStats()
            self.stats_loggers['boundary_stats'] = BoundaryMetricStats()


    def compute_forward(self, batch, stage):
        if not hasattr(self, 'target'):
            raise ValueError('target is not defined')

        batch = batch.to(self.device)
        feats, feat_lens = batch['feat']

        # initialize predictions
        predictions = {'losses': {}}

        # feature normalization
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalizer(feats, feat_lens, epoch=current_epoch)

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

        # boundary detector
        if self.target in [Target.B_DETECTOR, Target.VAE, Target.TEST]:
            fa_boundary_seqs = batch['fa_boundary_seq'][0]
            b_detector_out = self.modules['boundary_detector'](feats, feat_lens, fa_boundary_seqs)
            predictions['boundary_v'] = b_detector_out['boundary_v']
            # predictions['losses'].update(b_detector_out['losses'])

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
            phn_recog_in_fc_out = self.modules['phn_recog_in_fc'](phn_recog_out_dict['out'].detach())

            # concatenation
            rnn_in = torch.cat([feat_fc_out, phn_recog_in_fc_out], dim=-1)
            rnn_in = self.modules['concat_fc'](rnn_in)

            # RNN
            rnn_out = self.modules['rnn'](rnn_in)[0]  # (B, T, C)

            # compute Pi
            pi_logits = self.modules['pi_fc'](rnn_out)  # (B, T, 2)
            predictions['pi_logits'] = pi_logits

            # distribution for Pi
            dist = Categorical(logits=pi_logits)

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
                [torch.tensor(seq).float().to(self.device) for seq in decoded_flvl_md_lvl_seqs]
            decoded_flvl_md_lvl_seqs = pad_sequence(decoded_flvl_md_lvl_seqs, batch_first=True)
            pi_nll_loss = -dist.log_prob(decoded_flvl_md_lvl_seqs)
            predictions['losses']['pi_nll_loss'] = pi_nll_loss

            pi_mcmc_num = self.hparams.pi_mcmc_num if self.modules.training else 1
            sfl_losses = {
                'vae_kld_loss': 0,
                'recon_loss': 0,
                'rif_loss': 0,
                'entropy_loss': 0,
                'baseline_loss': 0
            }
            for _ in range(pi_mcmc_num):
                if self.modules.training:
                    sampled_pi = dist.sample().float()  # (B, T)
                else:
                    sampled_pi = torch.argmax(pi_logits, dim=-1).float()  # (B, T)
                sampled_pi_one_hot = torch.stack([1 - sampled_pi, sampled_pi], dim=2)  # (B, T, 2)
                predictions['sampled_pi'] = sampled_pi

                # VAE encoder and decoder
                encoder_out_dict = self.modules['encoder'](rnn_out, sampled_pi_one_hot)
                for key in encoder_out_dict['losses']:
                    if key not in sfl_losses:
                        raise ValueError(f'unexpected loss: {key}')
                    sfl_losses[key] += encoder_out_dict['losses'][key]

                sampled_h = encoder_out_dict['sampled_h']  # (B, T, C)
                decoder_out_dict = self.modules['decoder'](sampled_h, feats)
                predictions['losses'].update(decoder_out_dict['losses'])
                for key in decoder_out_dict['losses']:
                    if key not in sfl_losses:
                        raise ValueError(f'unexpected loss: {key}')
                    sfl_losses[key] += decoder_out_dict['losses'][key]

                # compute SFL losses
                nll = -dist.log_prob(sampled_pi)
                baseline = self.modules['baseline_fc'](rnn_out).squeeze(dim=-1)  # shape = (B, T)
                vae_kld_loss = torch.mean(encoder_out_dict['losses']['vae_kld_loss'], dim=-1)
                recon_loss = torch.mean(decoder_out_dict['losses']['recon_loss'], dim=-1)
                reward = -(self.hparams.recon_weight * recon_loss.detach()
                           + self.hparams.vae_kld_weight * vae_kld_loss.detach()
                           + self.hparams.pi_nll_weight * pi_nll_loss.detach())  # shape = (B, T)
                # reward = -recon_loss.detach()  # shape = (B, T)

                sfl_losses['rif_loss'] += (reward - baseline.detach()) * nll
                sfl_losses['entropy_loss'] += -dist.entropy()
                sfl_losses['baseline_loss'] += F.mse_loss(baseline, reward, reduction='none')

            # update SFL losses
            for key in sfl_losses:
                sfl_losses[key] /= pi_mcmc_num
            predictions['losses'].update(sfl_losses)

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        original_losses = predictions['losses']

        feats, feat_lens = batch['feat']

        # compute losses
        losses = {}

        # compute mean loss using lens
        for key in original_losses:
            losses[key] = apply_lens_to_loss(original_losses[key], feat_lens)

        # compute and save total loss
        loss = self.compute_and_save_losses(losses)

        # phoneme classification metrics
        if (stage == sb.Stage.VALID and self.target == Target.PHN_RECOG) or (stage == sb.Stage.TEST):
            pass

        # boundary metrics
        if (stage == sb.Stage.VALID and self.target == Target.B_DETECTOR) or (stage == sb.Stage.TEST):
            pass

        # MD metrics
        if self.to_run_evaluation(stage):
            # decoding
            plvl_cnnl_seqs, plvl_cnnl_seq_lens = batch['gt_cnncl_seq']
            weight = getattr(self.hparams, 'dec_weight', 1.0)
            decoded_boundary_seqs, pred_flvl_md_lbl_seqs, pred_plvl_md_lvl_seqs = decode_plvl_md_lbl_seqs(
                predictions,
                utt_ids=batch['id'],
                feat_lens=feat_lens,
                plvl_cnnl_seqs=plvl_cnnl_seqs,
                plvl_cnnl_seq_lens=plvl_cnnl_seq_lens,
                prior=batch['prior'][0][0],
                weight=weight
            )

            # MD metrics
            gt_md_lbl_seqs = undo_padding(*batch['plvl_gt_md_lbl_seq'])
            self.stats_loggers['plvl_md_stats'].append(
                ids=batch['id'],
                pred_md_lbl_seqs=pred_plvl_md_lvl_seqs,
                gt_md_lbl_seqs=gt_md_lbl_seqs,
            )

            # boundary metrics
            gt_boundary_seqs = undo_padding(*batch['gt_boundary_seq'])
            self.stats_loggers['boundary_stats'].append(
                ids=batch['id'],
                predictions=decoded_boundary_seqs,
                targets=gt_boundary_seqs
            )

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if self.to_run_evaluation(stage):
            super(SBModel, self).on_stage_end(stage, stage_loss, epoch)

    def to_run_evaluation(self, stage):
        return (stage == sb.Stage.VALID and self.target == Target.VAE) or (stage == sb.Stage.TEST)