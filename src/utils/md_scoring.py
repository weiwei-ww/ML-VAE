import numpy as np

import torch


def binary_seq_md_scoring(prediction, target):
    """
    Compute MD scores of two binary sequences.

    Parameters
    ----------
    prediction : np.ndarray or torch.Tensor or list
        MD results predicted by the model
    target : torch.Tensor or list
        MD ground truth

    Returns
    -------
    md_scores : dict
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

        if x.ndim > 1:
            raise ValueError('Only one-dimension input is allowed')

        if not torch.all(torch.logical_or(x == 0, x == 1)):
            raise ValueError('Only binary input values are supported')

        return x


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

    md_scores = {
        # 'TP': TP,
        # 'TN': TN,
        # 'FP': FP,
        # 'FN': FN,
        'ACC': ACC,
        'PRE': PRE,
        'REC': REC
    }

    return md_scores


def batch_seq_md_scoring(
        batch_pred_md_lbl_seqs=None,
        batch_pred_phn_seqs=None,
        batch_gt_md_lbl_seqs=None,
        batch_gt_phn_seqs=None,
        batch_gt_cnncl_seqs=None
):
    """
    Compute MD scores for a batch.

    Parameters
    ----------
    batch_pred_md_lbl_seqs : list
        list of predicted MD labels
    batch_pred_phn_seqs : list
        list of predicted phonemes
    batch_gt_md_lbl_seqs : list
        list of ground truth MD labels
    batch_gt_phn_seqs : list
        list of ground truth phonemes
    batch_gt_cnncl_seqs : list
        list of ground truth canonicals

    Returns
    -------
    batch_md_scores : list
        list of MD scores

    """
    # check input
    for x in [batch_pred_md_lbl_seqs, batch_pred_phn_seqs, batch_gt_md_lbl_seqs, batch_gt_phn_seqs, batch_gt_cnncl_seqs]:
        if x is not None and not isinstance(x, list):
            raise TypeError(f'Input type must be list, not {type(x).__name__}')

    # generate binary MD labels if not provided
    def generate_batch_md_lbls(batch_phn_seqs, batch_cnncl_seqs):
        # check input
        if batch_phn_seqs is None:
            raise ValueError('batch_phn_seqs cannot be None')
        if batch_cnncl_seqs is None:
            raise ValueError('batch_cnncl_seqs cannot be None')
        if len(batch_phn_seqs) != len(batch_cnncl_seqs):
            raise ValueError(f'Inconsistent batch size: {len(batch_phn_seqs)} != {len(batch_cnncl_seqs)}')

        # generate MD labels for each batch
        batch_md_lbl_seqs = []
        for phn_seq, cnncl_seq in zip(batch_phn_seqs, batch_cnncl_seqs):
            if len(phn_seq) != len(cnncl_seq):
                raise ValueError(f'Inconsistent sequence lengths: {len(phn_seq)} != {len(cnncl_seq)}')
            # 0: correct pronunciation; 1: mispronunciation
            md_lbl_seq = [int(p != c) for p, c in zip(phn_seq, cnncl_seq)]
            batch_md_lbl_seqs.append(md_lbl_seq)
        return batch_md_lbl_seqs

    if batch_pred_md_lbl_seqs is None:
        batch_pred_md_lbl_seqs = generate_batch_md_lbls(batch_pred_phn_seqs, batch_gt_cnncl_seqs)
    if batch_gt_md_lbl_seqs is None:
        batch_gt_md_lbl_seqs = generate_batch_md_lbls(batch_gt_phn_seqs, batch_gt_cnncl_seqs)

    # compute MD scores for each sample in the batch
    if len(batch_pred_md_lbl_seqs) != len(batch_gt_md_lbl_seqs):
        raise ValueError(f'Inconsistent batch size: {len(batch_pred_md_lbl_seqs)} != {len(batch_gt_md_lbl_seqs)}')

    batch_md_scores = []
    for pred_md_lbl_seq, gt_md_lbl_seq in zip(batch_pred_md_lbl_seqs, batch_gt_md_lbl_seqs):
        md_scores = binary_seq_md_scoring(pred_md_lbl_seq, gt_md_lbl_seq)
        batch_md_scores.append(md_scores)

    return batch_md_scores
