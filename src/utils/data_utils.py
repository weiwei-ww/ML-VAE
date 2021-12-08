import torch


def undo_padding_tensor(batch, lengths):
    """
    A tensor version of undo_padding. Return a list of tensors, instead of a list of lists.

    Parameters
    ----------
    batch : torch.Tensor
        (B, T, *)
    lengths : torch.Tensor
        (B)

    Returns
    -------
    as_list : list
        A list of unpadded tensors.

    """
    batch_max_len = batch.shape[1]
    as_list = []
    for seq, seq_length in zip(batch, lengths):
        actual_size = int(torch.round(seq_length * batch_max_len))
        seq_true = seq.narrow(0, 0, actual_size)
        as_list.append(seq_true)
    return as_list


def apply_weight(x, weight):
    """
    Apply weight using torch.bmm

    Parameters
    ----------
    x : torch.tensor
        (B, T, N, C) or (B, T, N * C)
    weight : torch.tensor
        (B, T, N)

    Returns
    -------
    weighted_x : torch.tensor
        (B, T, C)
    """
    B, T, N = weight.shape
    C = x.shape[-1]
    if x.ndim == 3:
        C = C // N
        x = x.view(B, T, N, C)
    weight = weight.view(B, T, 1, N)
    weighted_x = torch.bmm(weight, x)
    weighted_x = weighted_x.view(B, T, C)
    return weighted_x
