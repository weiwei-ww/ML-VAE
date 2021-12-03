import numpy as np

import torch

from utils.metric_stats.base_metric_stats import BaseMetricStats


class BoundaryMetricStats(BaseMetricStats):
    def __init__(self):
        super(BoundaryMetricStats, self).__init__(metric_fn=batch_boundary_scoring)

    def summarize(self, field=None):
        mean_scores = super(BoundaryMetricStats, self).summarize()

        for key in mean_scores:
            mean_scores[key] = round(mean_scores[key].item(), 2)

        if field is None:
            return mean_scores
        else:
            return mean_scores[field]


def boundary_scoring(prediction, target):
    """
    Compute boundary metrics.

    Parameters
    ----------
    prediction : torch.Tensor
        (T), model output
    target : torch.Tensor
        (T), ground truth

    Returns
    -------
    boundary_scores : dict
        Boundary metrics.
    """
    if prediction.ndim != 1 or target.ndim != 1:
        raise ValueError('Only one-dimensional inputs are supported')
    if len(prediction) != len(target):
        raise ValueError(f'Inconsistent input lengths: {len(prediction)} != {len(target)}')

    # convert boundary sequence into index sequence
    prediction_index_seq = torch.where(target == 1)[0]

    target_index_seq = torch.where(target == 1)[0]
    target_index_seq = torch.cat([target_index_seq, target_index_seq.new_full((1,), len(target))])

    # convert to intervals
    target_intervals = [(target_index_seq[i - 1], target_index_seq[i]) for i in range(1, len(target_index_seq))]

    # compute metrics
    predict_i = 0
    target_i = 0

    correct_num = 0
    while target_i < len(target_intervals) and predict_i < len(prediction_index_seq):
        left, right = target_intervals[target_i]
        boundary_index = prediction_index_seq[predict_i]
        # print(left, right, boundary_index)

        if boundary_index < left:
            predict_i += 1
            # print('too small')
        elif left <= boundary_index <= right:
            target_i += 1
            predict_i += 1
            correct_num += 1
            # print('correct')
        elif boundary_index > right:
            target_i += 1

    eps = 1e-6

    pre = (correct_num / (torch.sum(prediction) + eps) * 100).item()
    rec = (correct_num / (torch.sum(target) + eps) * 100).item()
    f1 = 2 * pre * rec / (pre + rec + eps)

    os = pre / (rec + eps) - 1
    r1 = np.sqrt((100 - rec) ** 2 + os ** 2)
    r2 = np.abs(rec - os - 100) / np.sqrt(2)
    r_value = (1 - (r1 + r2) / 200) * 100
    scores = {
        'pre': pre,
        'rec': rec,
        'f1': f1,
        'r-value': r_value
    }

    return scores


def batch_boundary_scoring(predictions, targets):
    """
    Compute the boundary scores for a batch.

    Parameters
    ----------
    predictions : list
        A list of model output.
    targets : list
        A list of ground truth.

    Returns
    -------
    scores_list : list
        A list of dictionaries of boundary scores.

    """
    # check input
    for x in [predictions, targets]:
        if type(x) is not list:
            raise TypeError(f'Input type must be list, not {type(x).__name__}')
    if len(predictions) != len(targets):
            raise ValueError(f'Inconsistent batch size: {len(predictions)} != {len(targets)}')

    # calculate boundary scores for each sample in the batch
    scores_list = []
    for prediction, target in zip(predictions, targets):
        scores_list.append(boundary_scoring(prediction, target))

    return scores_list
