import torch


def undo_padding_tensor(batch, lengths):
    """
    A tensor version of undo_padding. Return a list of tensors, instead of a list of lists.

    Parameters
    ----------
    batch : torch.Tensor
        (B, T, *)
    lengths : torch.Tensor

    Returns
    -------

    """
    batch_max_len = batch.shape[1]
    as_list = []
    for seq, seq_length in zip(batch, lengths):
        actual_size = int(torch.round(seq_length * batch_max_len))
        seq_true = seq.narrow(0, 0, actual_size)
        as_list.append(seq_true)
    return as_list