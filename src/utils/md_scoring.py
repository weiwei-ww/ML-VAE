import numpy as np

import torch


def binary_seq_md_scoring(prediction, target):
    """
    return MD scores of two binary sequences

    Parameters
    ----------
    prediction : np.ndarray or torch.Tensor or list
        MD results predicted by the model
    target : torch.Tensor or list
        MD ground truth

    Returns
    -------
    md_score : dict
        a dictionary of MD scores
    """
    # # check input
    # valid_input_types = (np.ndarray, torch.Tensor, list)
    # if not isinstance(predict, valid_input_types):
    #     raise TypeError(f'Unsupported input type: {type(predict).__name__}')
    # if not isinstance(target, valid_input_types):
    #     raise TypeError(f'Unsupported input type: {type(target).__name__}')

    # convert to torch.LongTensor
    def convert_to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif isinstance(x, list):
            x = torch.Tensor(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f'Unsupported input type: {type(x).__name__}')
        x = x.long().squeeze()

        if x.ndim() > 1:
            raise ValueError('Only one-dimension input is allowed')

        if not torch.all(torch.logical_or(x == 0, x == 1)):
            raise ValueError('Only binary input values are supported')


    prediction = convert_to_tensor(prediction)
    target = convert_to_tensor(target)

    TP = torch.sum((1 - prediction) * (1 - target))
    TN = torch.sum(prediction * target)
    FP = torch.sum((1 - prediction) * target)
    FN = torch.sum(prediction * (1 - target))

    eps = 1e-6
    ACC = (TP + TN) / (TP + TN + FP + FN + eps) * 100
    PRE = TN / (TN + FN + eps) * 100
    REC = TN / (TN + FP + eps) * 100

    md_score = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'ACC': ACC,
        'PRE': PRE,
        'REC': REC
    }

    return md_score


def batch_seq_md_scoring(
        batch_pred_md_lbls=None,
        batch_pred_phns=None,
        batch_gt_md_lbls=None,
        batch_gt_phns=None,
        batch_gt_cnncls=None
):
    # check input
    for x in [batch_pred_md_lbls, batch_pred_phns, batch_gt_md_lbls, batch_gt_phns, batch_gt_cnncls]:
        if x is not None and not isinstance(x, list):
            raise TypeError(f'Input type must be list, not {type(x).__name__}')

    # generate binary MD labels
    def generate_batch_md_lbls(batch_phns, batch_cnncls):
        # check input
        if batch_phns is None:
            raise ValueError('batch_phns cannot be None')
        if batch_cnncls is None:
            raise ValueError('batch_cnncls cannot be None')
        if len(batch_phns) != len(batch_cnncls):
            raise ValueError(f'Inconsistent batch size: {len(batch_phns)} != {len(batch_cnncls)}')

        # generate MD labels for each batch
        batch_md_lbls = []
        for phns, cnncls in zip(batch_phns, batch_cnncls):
            if len(phns) != len(cnncls):
                raise ValueError(f'Inconsistent sequence lengths: {len(phns)} != {len(cnncls)}')
            # 0: correct pronunciation; 1: mispronunciation
            md_lbls = [int(p != c) for p, c in zip(phns, cnncls)]
            batch_md_lbls.append(md_lbls)
        return batch_md_lbls

    if batch_pred_md_lbls is None:
        batch_pred_md_lbls = generate_batch_md_lbls(batch_pred_phns, batch_gt_cnncls)
    if batch_gt_md_lbls is None:
        batch_pred_md_lbls = generate_batch_md_lbls(batch_gt_phns, batch_gt_cnncls)






# def generate_md_seqs(predict_seq, phoneme_seq, canonical_seq):
