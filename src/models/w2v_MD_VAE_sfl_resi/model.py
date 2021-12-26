import logging

from torch.distributions import Categorical
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models.MD_VAE.model import SBModel as MD_VAE
from models.MD_VAE.model import Target
from utils.decode_utils import decode_plvl_md_lbl_seqs_full as decode_plvl_md_lbl_seqs

logger = logging.getLogger(__name__)


class SBModel(MD_VAE):
    def compute_forward(self, batch, stage):
        if not hasattr(self, 'target'):
            raise ValueError('target is not defined')

        batch = batch.to(self.device)
        feats, feat_lens = batch['feat']

        wavs, wav_lens = batch['wav']
        w2v_feats = self.modules['wav2vec2'](wavs)  # (B, T, 1024)

        # handle inconsistent lengths
        shape_diff = w2v_feats.shape[1] - feats.shape[1]
        assert -2 <= shape_diff <= 0, f'shape_diff = {shape_diff}'
        if shape_diff < 0:
            zeros = torch.zeros(w2v_feats.shape[0], abs(shape_diff), w2v_feats.shape[2]).float().to(feats.device)
            w2v_feats = torch.concat([w2v_feats, zeros], dim=1)
        assert w2v_feats.shape[1] == feats.shape[1]

        # initialize predictions
        predictions = {'losses': {}}

        # phoneme recognizer
        if self.target in [Target.PHN_RECOG, Target.VAE, Target.TEST]:
            phn_recog_in_fc_out = self.modules['phn_recog_in_fc'](w2v_feats)
            phn_recog_in = torch.cat([feats, phn_recog_in_fc_out], dim=-1)

            plvl_cnnl_phn_seqs, plvl_cnnl_phn_seq_lens = batch['gt_cnncl_seq']
            fa_boundary_seqs = batch['fa_boundary_seq'][0]
            phn_recog_out_dict = self.modules['phoneme_recognizer'](
                phn_recog_in, feat_lens, plvl_cnnl_phn_seqs, plvl_cnnl_phn_seq_lens, fa_boundary_seqs
            )
            predictions['phn_recog_out'] = phn_recog_out_dict['out']

            losses = phn_recog_out_dict['losses']
            if self.target != Target.PHN_RECOG:
                for key in losses:
                    losses[key] = losses[key].detach()
            predictions['losses'].update(losses)

        # boundary detector
        if self.target in [Target.B_DETECTOR, Target.VAE, Target.TEST]:
            b_detector_in_fc_out = self.modules['b_detector_in_fc'](w2v_feats)
            b_detector_in = torch.cat([feats, b_detector_in_fc_out], dim=-1)

            fa_boundary_seqs = batch['fa_boundary_seq'][0]
            b_detector_out = self.modules['boundary_detector'](b_detector_in, feat_lens, fa_boundary_seqs)
            predictions['boundary_v'] = b_detector_out['boundary_v']

            losses = b_detector_out['losses']
            if self.target != Target.B_DETECTOR:
                for key in losses:
                    losses[key] = losses[key].detach()
            predictions['losses'].update(losses)

        # VAE
        if self.target in [Target.VAE, Target.TEST]:
            # feat FC
            w2v_feat_fc_out = self.modules['w2v_feat_fc'](w2v_feats)

            # FC for phoneme recognizer output
            phn_recog_out_fc_out = self.modules['phn_recog_out_fc'](phn_recog_out_dict['out'].detach())

            # concatenation
            rnn_in = torch.cat([feats, w2v_feat_fc_out, phn_recog_out_fc_out], dim=-1)
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
