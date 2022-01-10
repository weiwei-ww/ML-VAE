import logging
import warnings
from pathlib import Path
import numpy as np
import torch
from ctc_segmentation import CtcSegmentationParameters, ctc_segmentation, determine_utterance_segments

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.nnet.losses import ctc_loss

from utils.alignment import batch_align_sequences
from utils.data_utils import undo_padding_tensor, resample_tensor
from utils.metric_stats.md_metric_stats import MDMetricStats
from utils.metric_stats.boundary_metric_stats import BoundaryMetricStats
from models.md_model import MDModel

logger = logging.getLogger(__name__)


class SBModel(MDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        self.stats_loggers['phn_per_stats'] = ErrorRateStats()
        self.stats_loggers['cnncl_per_stats'] = ErrorRateStats()
        self.stats_loggers['plvl_md_stats'] = MDMetricStats()
        self.stats_loggers['boundary_stats'] = BoundaryMetricStats()

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

        phns, phn_lens = batch['gt_phn_seq']
        loss = ctc_loss(pout, phns, pout_lens, phn_lens, self.label_encoder.get_blank_index())
        return loss

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        pout = predictions['pout']
        pout_lens = batch['feat'][1]

        # compute CTC loss
        loss = self.compute_loss(predictions, batch)

        # compute PER
        pred_phns = sb.decoders.ctc_greedy_decode(
            pout, pout_lens, blank_id=self.label_encoder.get_blank_index()
        )
        phns, phn_lens = batch['gt_phn_seq']
        self.stats_loggers['phn_per_stats'].append(
            ids=batch.id,
            predict=pred_phns,
            target=phns,
            target_len=phn_lens,
            ind2lab=self.label_encoder.decode_ndim,
        )
        cnncls, cnncl_lens = batch['gt_cnncl_seq']
        self.stats_loggers['cnncl_per_stats'].append(
            ids=batch.id,
            predict=pred_phns,
            target=cnncls,
            target_len=cnncl_lens,
            ind2lab=self.label_encoder.decode_ndim,
        )

        # compute MD metrics
        pred_phns = sb.decoders.ctc_greedy_decode(
            pout, pout_lens, blank_id=self.label_encoder.get_blank_index()
        )

        # unpad sequences
        gt_phn_seqs, gt_phn_seq_lens = batch['gt_phn_seq']
        gt_cnncl_seqs, gt_cnncl_seq_lens = batch['gt_cnncl_seq']

        gt_phn_seqs = undo_padding(gt_phn_seqs, gt_phn_seq_lens)
        gt_cnncl_seqs = undo_padding(gt_cnncl_seqs, gt_cnncl_seq_lens)

        # align sequences
        ali_gt_phn_seqs, ali_pred_phn_seqs, ali_gt_cnncl_seqs = \
            batch_align_sequences(gt_phn_seqs, pred_phns, gt_cnncl_seqs, ignore_insertion=True)

        # compute CTC segmentation results
        ctc_segmentation_boundary_seqs = self.compute_ctc_segmentation(batch, pout)
        unpadded_gt_boundary_seqs = undo_padding_tensor(*batch['gt_boundary_seq'])
        unpadded_gt_boundary_seqs = [seq.cpu() for seq in unpadded_gt_boundary_seqs]

        self.stats_loggers['plvl_md_stats'].append(
            batch['id'],
            pred_phn_seqs=ali_pred_phn_seqs,
            gt_phn_seqs=ali_gt_phn_seqs,
            gt_cnncl_seqs=ali_gt_cnncl_seqs,
            pred_boundary_seqs=ctc_segmentation_boundary_seqs,
            gt_boundary_seqs=unpadded_gt_boundary_seqs
        )

        self.stats_loggers['boundary_stats'].append(
            batch['id'],
            predictions=ctc_segmentation_boundary_seqs,
            targets=unpadded_gt_boundary_seqs
        )

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        super(SBModel, self).on_stage_end(stage, stage_loss, epoch)

        if stage == sb.Stage.TEST:
            output_path = Path(self.hparams.output_dir) / 'test_output' / 'md_result_seqs.txt'
            self.stats_loggers['plvl_md_stats'].write_seqs_to_file(output_path, self.label_encoder)

    def compute_ctc_segmentation(self, batch, pouts):
        feats, feat_lens = batch['feat']
        pouts = resample_tensor(pouts, feats, dim=1)

        unpadded_pouts = undo_padding_tensor(pouts, feat_lens)
        assert len(batch['id']) == len(unpadded_pouts)
        unpadded_gt_cnncl_seq = undo_padding(*batch['gt_cnncl_seq'])

        config = CtcSegmentationParameters()
        config.char_list = list(range(pouts.shape[-1]))
        boundary_seqs = []
        for b, (utt_id, pout) in enumerate(zip(batch['id'], unpadded_pouts)):
            pout = pout.cpu().detach().numpy()
            y = unpadded_gt_cnncl_seq[b]  # shape = (L)
            new_y = [-1, 0]
            utt_start_indices = [1]
            for item in y:
                new_y.extend([item])
                utt_start_indices.append(len(new_y) - 1)
            new_y = np.array(new_y).reshape(-1, 1)

            timings, char_probs, state_list = ctc_segmentation(config, pout, new_y)
            segments = determine_utterance_segments(
                config, utt_start_indices, char_probs, timings, utt_start_indices[:-1]
            )

            boundary_seq = torch.zeros(pout.shape[0])
            for i, (start, end, _) in enumerate(segments):
                if i == 0:
                    start_index = 0
                else:
                    start_index = int(np.ceil(start / config.index_duration))
                end_index = int(np.ceil(end / config.index_duration))
                # assert start_index < end_index
                # if start_index >= end_index:
                #     warnings.warn(f'start_index = {start_index}, end_index = {end_index}')
                while boundary_seq[start_index] == 1:
                    start_index += 1
                    warnings.warn('move one')
                boundary_seq[start_index] = 1
            boundary_seqs.append(boundary_seq)

        return boundary_seqs
