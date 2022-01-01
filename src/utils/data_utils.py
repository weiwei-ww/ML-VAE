import torch

from speechbrain.nnet.losses import length_to_mask


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

    # reshape tensors
    x = x.view(B * T, N, C)
    weight = weight.view(B * T, 1, N)

    # apply weight
    weighted_x = torch.bmm(weight, x)

    # reshape weighted x
    weighted_x = weighted_x.view(B, T, C)

    return weighted_x


def apply_lens_to_loss(loss, lens, reduction='mean'):
    """
    Compute the mean loss of a batch while considering the lengths of each sample.

    Parameters
    ----------
    loss : torch.tensor
        (B, T, C)
    lens : torch.Tensor
        (B)
    reduction : str
        'mean', 'batchmean' or 'batch'

    Returns
    -------
    loss : torch.tensor
        Single value. Mean loss.
    """
    # compute and apply mask
    mask = torch.ones_like(loss)
    length_mask = length_to_mask(lens * mask.shape[1], max_len=mask.shape[1])
    length_mask = length_mask.type(mask.dtype)
    # handle any dimensionality of input
    while len(length_mask.shape) < len(mask.shape):
        length_mask = length_mask.unsqueeze(-1)
    mask *= length_mask

    # compute loss
    loss = loss * mask
    B = loss.size(0)
    if reduction == 'mean':
        loss = loss.sum() / torch.sum(mask)
    elif reduction == 'batchmean':
        loss = loss.sum() / B
    elif reduction == 'batch':
        loss = loss.reshape(B, -1).sum(dim=-1) / mask.reshape(B, -1).sum(dim=-1)

    return loss


def resample_tensor(source, target, dim):
    """
    Resample the time dimension of source tensor to match target tensor.

    Parameters
    ----------
    source : torch.Tensor
        The tensor to be resampled.
    target : torch.Tensor
        The target tensor.
    dim : int
        The time dimension.

    Returns
    -------
    resampled_source: torch.Tensor
        (B, T, *)
    """
    source_shape = source.shape
    source_ind = [slice(None) for _ in source_shape]

    source_T = source.shape[dim]
    target_T = target.shape[dim]

    factor = round(target_T // source_T)
    if not factor > 0:
        raise ValueError(f'non-positive factor for input lengths: {source_T} and {target_T}')

    # repeat the values in source tensor
    resampled_source = torch.repeat_interleave(source, factor, dim=dim)

    # check shape difference
    shape_diff = resampled_source.shape[dim] - target_T
    diff_tol = 3
    if not -diff_tol <= shape_diff <= diff_tol:
        raise ValueError(f'length difference between resampled tensor and target tensor is too large: {shape_diff}')

    if shape_diff > 0:  # remove extra values from resampled tensor
        source_ind[dim] = slice(target_T)
        resampled_source = resampled_source[source_ind]
    elif shape_diff < 0:  # pad resampled tensor with zeros
        zeros = torch.zeros_like(resampled_source)
        resampled_ind = [slice(None) for _ in resampled_source.shape]
        resampled_ind[dim] = slice(-shape_diff)
        zeros = zeros[resampled_ind]
        resampled_source = torch.cat([resampled_source, zeros], dim=dim)

    assert resampled_source.shape[dim] == target_T

    return resampled_source



