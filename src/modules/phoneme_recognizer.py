import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from modules.fc_block import FCBlock


class PhonemeRecognizer(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, rnn_num_layers, fc_sizes, n_phonemes):
        super(PhonemeRecognizer, self).__init__()

        self.rnn = nn.LSTM(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.fc = FCBlock(fc_sizes)
        self.n_phonemes = n_phonemes

    def forward(
            self,
            feats,  # (B, T, C)
            feat_lens,  # (B)
            plvl_cnnl_phn_seqs,  # (B, L)
            plvl_cnnl_phn_seq_lens,  # (B)
            boundary_seqs,  # (B, T)
    ):
        out = self.rnn(feats)[0]
        out = self.fc(out)

        losses = self.compute_losses(out, feat_lens, plvl_cnnl_phn_seqs, plvl_cnnl_phn_seq_lens, boundary_seqs)

        return {
            'out': out,
            'losses': losses
        }

    def compute_losses(
            self,
            out,  # (B, T, C)
            feat_lens,  # (B)
            plvl_cnnl_phn_seqs,  # (B, L)
            plvl_cnnl_phn_seq_lens,  # (B)
            boundary_seqs,  # (B, T)
    ):
        abs_feat_lens = torch.round(out.shape[1] * feat_lens).int()  # (B)
        abs_phn_seq_lens = torch.round(plvl_cnnl_phn_seqs.shape[1] * plvl_cnnl_phn_seq_lens).int()  # (B)

        # convert to one-hot
        num_classes = self.n_phonemes + 2
        plvl_cnnl_phn_seqs = F.one_hot(plvl_cnnl_phn_seqs, num_classes=num_classes).type(out.dtype)
        # unpadded_out =

        loss_list = []
        for i in range(out.shape[0]):  # for each batch
            # get number of frames and phonemes
            T_i = abs_feat_lens[i]
            L_i = abs_phn_seq_lens[i]

            # get output & ground truth for the i-th sample in the batch
            out_i = out[i, :T_i, :]  # shape = (T_i, N)
            y_i = plvl_cnnl_phn_seqs[i, :L_i, :]  # shape = (L_i, N)
            boundary_seq_i = boundary_seqs[i, :T_i]  # shape = (T_i)

            # compute durations for each phoneme
            boundary_index_list = torch.where(boundary_seq_i == 1)[0].tolist()  # shape = (L_i)
            assert len(boundary_index_list) == L_i
            boundary_index_list.append(T_i)  # append the last boundary
            boundary_index_list = torch.tensor(boundary_index_list).to(out.device)
            duration = boundary_index_list[1:] - boundary_index_list[:-1]
            assert torch.sum(duration) == T_i

            # extend y
            y_i = torch.repeat_interleave(y_i, duration, dim=0)  # shape = (T_i, N)

            # compute loss
            batch_loss = F.binary_cross_entropy_with_logits(out_i, y_i, reduction='none')
            loss_list.append(batch_loss)

        loss = pad_sequence(loss_list, batch_first=True)

        assert loss.shape == out.shape

        return {'phn_recog_bce_loss': loss}


