import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.fc_block import FCBlock


class PhonemePredictor(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, rnn_num_layers, fc_sizes):
        super(PhonemePredictor, self).__init__()

        self.rnn = nn.LSTM(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)

        self.fc = FCBlock(fc_sizes)


    def forward(
            self,
            feats
    ):
        out = self.rnn(feats)[0]
        out = self.fc(out)
        return out

    def calculate_losses(
            self,
            feat_lens,
            cnnl_phn_seqs,
            plvl_cnnl_phn_seq_lens
    ):
        return None


    def old_forward(self,
                x,  # x.shape = (B, T, C)
                lens,  # lens.shape = (B)
                can_seqs,  # can_seqs.shape = (B, L, N)
                can_seq_lens,  # can_seq_lens.shape = (B)
                boundaries  # shape = (B, T)
                ):
        B, T, _ = x.shape
        L = can_seqs.shape[1]

        # RNN
        x = self.rnn(x)[0]  # shape = (B, T, C)

        # phoneme classification
        fc_out = self.fc(x)  # shape = (B, T, N)

        # compute loss
        loss = 0
        for i in range(B):
            T_i = lens[i]
            L_i = can_seq_lens[i]
            output_i = fc_out[i, :T_i, :]  # shape = (T_i, N)
            y_i = can_seqs[i, :L_i, :]  # shape = (L_i, N)
            boundary = boundaries[i, :T_i]  # shape = (T_i)

            # compute durations for each phoneme
            boundary_index_list = list(torch.where(boundary == 1)[0])  # shape = (L_i)
            assert len(boundary_index_list) == L_i
            boundary_index_list.append(T_i)
            boundary_index_list = torch.tensor(boundary_index_list, device=boundary.device)
            duration = boundary_index_list[1:] - boundary_index_list[:-1]
            assert torch.sum(duration) == T_i

            # copy y
            y_i = torch.repeat_interleave(y_i, duration, dim=0)  # shape = (T_i, N)

            # compute loss
            batch_loss = F.binary_cross_entropy_with_logits(output_i, y_i.to(torch.float32))
            loss = loss + batch_loss

        loss = loss / B

        ret = {
            'out': fc_out,  # shape = (B, T, N)
            'loss': {
                'phoneme_bce_loss': loss,
            }
        }

        return ret
