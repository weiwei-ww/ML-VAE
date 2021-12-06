import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.fc_block import FCBlock
from utils.data_utils import undo_padding_tensor


class PhonemeRecognizer(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, rnn_num_layers, fc_sizes, n_phonemes):
        super(PhonemeRecognizer, self).__init__()

        self.rnn = nn.LSTM(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.fc = FCBlock(fc_sizes)
        self.n_phonemes = n_phonemes

    def forward(self, feats):
        out = self.rnn(feats)[0]
        out = self.fc(out)
        return out

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

        loss = 0
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

            # copy y
            y_i = torch.repeat_interleave(y_i, duration, dim=0)  # shape = (T_i, N)

            # compute loss
            batch_loss = F.binary_cross_entropy_with_logits(out_i, y_i)
            loss = loss + batch_loss

        return {'phoneme_bce_loss': loss}


    # def old_forward(self,
    #             x,  # x.shape = (B, T, C)
    #             lens,  # lens.shape = (B)
    #             can_seqs,  # can_seqs.shape = (B, L, N)
    #             can_seq_lens,  # can_seq_lens.shape = (B)
    #             boundaries  # shape = (B, T)
    #             ):
    #     B, T, _ = x.shape
    #     L = can_seqs.shape[1]
    #
    #     # RNN
    #     x = self.rnn(x)[0]  # shape = (B, T, C)
    #
    #     # phoneme classification
    #     fc_out = self.fc(x)  # shape = (B, T, N)
    #
    #     # compute loss
    #     loss = 0
    #     for i in range(B):
    #         T_i = lens[i]
    #         L_i = can_seq_lens[i]
    #         output_i = fc_out[i, :T_i, :]  # shape = (T_i, N)
    #         y_i = can_seqs[i, :L_i, :]  # shape = (L_i, N)
    #         boundary = boundaries[i, :T_i]  # shape = (T_i)
    #
    #         # compute durations for each phoneme
    #         boundary_index_list = list(torch.where(boundary == 1)[0])  # shape = (L_i)
    #         assert len(boundary_index_list) == L_i
    #         boundary_index_list.append(T_i)
    #         boundary_index_list = torch.tensor(boundary_index_list, device=boundary.device)
    #         duration = boundary_index_list[1:] - boundary_index_list[:-1]
    #         assert torch.sum(duration) == T_i
    #
    #         # copy y
    #         y_i = torch.repeat_interleave(y_i, duration, dim=0)  # shape = (T_i, N)
    #
    #         # compute loss
    #         batch_loss = F.binary_cross_entropy_with_logits(output_i, y_i.to(torch.float32))
    #         loss = loss + batch_loss
    #
    #     loss = loss / B
    #
    #     ret = {
    #         'out': fc_out,  # shape = (B, T, N)
    #         'loss': {
    #             'phoneme_bce_loss': loss,
    #         }
    #     }
    #
    #     return ret
