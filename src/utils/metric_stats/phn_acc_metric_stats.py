import torch

from utils.metric_stats.base_metric_stats import BaseMetricStats


class PhnAccMetricStats(BaseMetricStats):
    def __init__(self):
        super(PhnAccMetricStats, self).__init__(metric_fn=batch_phn_acc_scoring)

    def summarize(self, field=None):
        mean_scores = super(PhnAccMetricStats, self).summarize()

        for key in mean_scores:
            mean_scores[key] = round(mean_scores[key].item(), 2)

        if field is None:
            return mean_scores
        else:
            return mean_scores[field]


def flvl_phn_acc_scoring(prediction, target):
    """
    Compute frame level phoneme classification accuracy.

    Parameters
    ----------
    prediction : torch.Tensor
        (T, N), model output
    target : torch.Tensor
        (T), ground truth

    Returns
    -------
    acc: float
        phoneme classification accuracy
    """
    if prediction.ndim != 2 or target.ndim != 1:
        raise ValueError('Prediction must have two dimensions, and target must have one dimension')
    if prediction.shape[0] != target.shape[0]:
        raise ValueError(f'Inconsistent input lengths: {prediction.shape[0]} != {target.shape[0]}')
    argmax_prediction = torch.argmax(prediction, dim=-1)  # (T)
    target = target.type(argmax_prediction.dtype)

    acc = (argmax_prediction == target).float().mean().item() * 100

    return acc


def plvl_phn_acc_scoring(prediction, target, boundary_seq):
    """
    Compute phoneme level phoneme classification accuracy.

    Parameters
    ----------
    prediction : torch.Tensor
        (T, N), model output
    target : torch.Tensor
        (L), ground truth
    boundary_seq : torch.Tensor
        (T), boundary indicator sequence

    Returns
    -------
    acc: float
        phoneme classification accuracy
    """
    # get the absolute durations of each phoneme
    assert torch.sum(boundary_seq) == len(target)
    boundary_index_seq = torch.where(boundary_seq == 1)[0]
    boundary_index_seq = torch.cat([boundary_index_seq, boundary_index_seq.new_full((1,), len(boundary_seq))])
    durations = boundary_index_seq[1:] - boundary_index_seq[:-1]
    assert torch.sum(durations) == prediction.shape[0]

    # split the output
    prediction_split = torch.split(prediction, durations.tolist())  # split the prediction into a list of tensors
    prediction_split = [torch.sum(p, dim=0) for p in prediction_split]  # sum over each phoneme segment
    assert len(prediction_split) == len(target)
    plvl_prediction = torch.stack(prediction_split, dim=0)  # (L, N)
    assert plvl_prediction.shape[0] == len(target)

    # calculate the accuracy
    acc = flvl_phn_acc_scoring(plvl_prediction, target)

    return acc


def batch_phn_acc_scoring(
        predictions,
        flvl_targets,
        plvl_targets=None,
        boundary_seqs=None
):
    """
    Compute phoneme classification accuracy for a batch.

    Parameters
    ----------
    predictions : list
        A list of model output.
    flvl_targets : list
        A list of frame level ground truth.
    plvl_targets : list
        A list of phoneme level ground truth.
    boundary_seqs : list
        A list of boundary indicator sequence.

    Returns
    -------
    phn_acc : list
        A list of phoneme classification accuracy for the batch. Each item is a tuple: (flvl_acc, plvl_acc).

    """
    # check input
    for x in [predictions, flvl_targets, plvl_targets, boundary_seqs]:
        if x is not None and type(x) is not list:
            raise TypeError(f'Input type must be list, not {type(x).__name__}')
    for x in [flvl_targets, plvl_targets, boundary_seqs]:
        if x is not None and len(x) != len(predictions):
            raise ValueError(f'Inconsistent batch size: {len(x)} != {len(predictions)}')
    if plvl_targets is not None and boundary_seqs is None:
        raise ValueError('boundary_seqs must be provided when plvl_targets is not None')

    # calculate the frame and phoneme level accuracies
    phn_acc = []
    for i in range(len(predictions)):
        flvl_acc = flvl_phn_acc_scoring(predictions[i], flvl_targets[i])
        plvl_acc = 0
        if plvl_targets is not None:
            plvl_acc = plvl_phn_acc_scoring(predictions[i], plvl_targets[i], boundary_seqs[i])
        phn_acc.append({'flvl_acc': flvl_acc, 'plvl_acc': plvl_acc})

    return phn_acc
