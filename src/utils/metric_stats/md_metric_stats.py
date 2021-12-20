import numpy as np

import torch

from utils.metric_stats.base_metric_stats import BaseMetricStats


class MDMetricStats(BaseMetricStats):
    def __init__(self):
        super(MDMetricStats, self).__init__(metric_fn=batch_seq_md_scoring)
        self.saved_seqs = {}

    def append(self, ids, **kwargs):
        if self.metric_fn is None:
            raise ValueError('No metric_fn has been provided')
        self.ids.extend(ids)  # save ID
        scores, seqs = self.metric_fn(**kwargs)
        self.scores_list.extend(scores)  # save metrics
        if len(self.metric_keys) == 0:  # save metric keys
            self.metric_keys = list(self.scores_list[0].keys())

        # update saved sequences
        seqs['utt_ids'] = ids
        if len(self.saved_seqs) == 0:
            self.saved_seqs = seqs
        else:
            for key in self.saved_seqs:
                self.saved_seqs[key].extend(seqs[key])

    def summarize(self, field=None):
        mean_scores = super(MDMetricStats, self).summarize()

        eps = 1e-6
        PRE = mean_scores['PRE']
        REC = mean_scores['REC']
        mean_scores['F1'] = (2 * PRE * REC) / (PRE + REC + eps)

        for key in mean_scores:
            mean_scores[key] = round(mean_scores[key].item(), 2)

        if field is None:
            return mean_scores
        else:
            return mean_scores[field]

    def write_seqs_to_file(self, path, label_encoder=None):
        with open(path, 'w') as f:
            batch_write_md_results(fp=f, scores_list=self.scores_list, label_encoder=label_encoder, **self.saved_seqs)



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
        x = x.int().squeeze()

        if x.ndim > 1:
            raise ValueError('Only one-dimension input is allowed')

        if not torch.all(torch.logical_or(x == 0, x == 1)):
            raise ValueError('Only binary input values are supported')

        return x


    prediction = convert_to_tensor(prediction)
    target = convert_to_tensor(target)

    if abs(len(prediction) - len(target)) > 0:
        raise ValueError(f'Inconsistent lengths for prediction and target sequences: {len(prediction)} != {len(target)}')

    TP = torch.sum((1 - prediction) * (1 - target))
    TN = torch.sum(prediction * target)
    FP = torch.sum((1 - prediction) * target)
    FN = torch.sum(prediction * (1 - target))

    eps = 1e-6
    ACC = (TP + TN) / (TP + TN + FP + FN + eps) * 100
    PRE = TN / (TN + FN + eps) * 100
    REC = TN / (TN + FP + eps) * 100
    F1 = 2 * PRE * REC / (PRE + REC)

    md_scores = {
        'ACC': ACC,
        'PRE': PRE,
        'REC': REC,
        'F1': F1
    }

    return md_scores


def batch_seq_md_scoring(
        pred_md_lbl_seqs=None,
        pred_phn_seqs=None,
        gt_md_lbl_seqs=None,
        gt_phn_seqs=None,
        gt_cnncl_seqs=None
):
    """
    Compute MD scores for a batch.

    Parameters
    ----------
    pred_md_lbl_seqs : list
        list of predicted MD labels
    pred_phn_seqs : list
        list of predicted phonemes
    gt_md_lbl_seqs : list
        list of ground truth MD labels
    gt_phn_seqs : list
        list of ground truth phonemes
    gt_cnncl_seqs : list
        list of ground truth canonicals

    Returns
    -------
    batch_md_scores : list
        list of MD scores

    """
    # check input
    for x in [pred_md_lbl_seqs, pred_phn_seqs, gt_md_lbl_seqs, gt_phn_seqs, gt_cnncl_seqs]:
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
        md_lbl_seqs = []
        for phn_seq, cnncl_seq in zip(batch_phn_seqs, batch_cnncl_seqs):
            if len(phn_seq) != len(cnncl_seq):
                raise ValueError(f'Inconsistent sequence lengths: {len(phn_seq)} != {len(cnncl_seq)}')
            # 0: correct pronunciation; 1: mispronunciation
            md_lbl_seq = [int(p != c) for p, c in zip(phn_seq, cnncl_seq)]
            md_lbl_seqs.append(md_lbl_seq)
        return md_lbl_seqs

    if pred_md_lbl_seqs is None:
        pred_md_lbl_seqs = generate_batch_md_lbls(pred_phn_seqs, gt_cnncl_seqs)
    if gt_md_lbl_seqs is None:
        gt_md_lbl_seqs = generate_batch_md_lbls(gt_phn_seqs, gt_cnncl_seqs)

    # compute and save MD scores for each sample in the batch
    if len(pred_md_lbl_seqs) != len(gt_md_lbl_seqs):
        raise ValueError(f'Inconsistent batch size: {len(pred_md_lbl_seqs)} != {len(gt_md_lbl_seqs)}')
    md_scores = []
    for pred_md_lbl_seq, gt_md_lbl_seq in zip(pred_md_lbl_seqs, gt_md_lbl_seqs):
        md_scores.append(binary_seq_md_scoring(pred_md_lbl_seq, gt_md_lbl_seq))

    # save sequences for writing to the file
    seqs_keys = ['gt_phn_seqs', 'gt_cnncl_seqs', 'gt_md_lbl_seqs', 'pred_phn_seqs', 'pred_md_lbl_seqs']
    seqs_dict = {key: [] for key in seqs_keys}
    for i in range(len(md_scores)):
        L = len(pred_md_lbl_seqs[i])
        def get_seq(seqs, i):
            if seqs is None:
                return [7] * L
            else:
                return seqs[i]

        # save sequences
        seqs_dict['gt_phn_seqs'].append(get_seq(gt_phn_seqs, i))
        seqs_dict['gt_cnncl_seqs'].append(get_seq(gt_cnncl_seqs, i))
        seqs_dict['gt_md_lbl_seqs'].append(get_seq(gt_md_lbl_seqs, i))
        seqs_dict['pred_phn_seqs'].append(get_seq(pred_phn_seqs, i))
        seqs_dict['pred_md_lbl_seqs'].append(get_seq(pred_md_lbl_seqs, i))

    return md_scores, seqs_dict


def write_md_results(
        fp,
        scores,
        utt_id,
        gt_phn_seq,
        gt_cnncl_seq,
        gt_md_lbl_seq,
        pred_phn_seq=None,
        pred_md_lbl_seq=None,
        label_encoder=None
):
    """
    Write MD results to a file.

    Parameters
    ----------
    scores : dict
        MD scores.
    fp : File
        File object to write the results.
    utt_id : str
        Utterance ID.
    gt_phn_seq : list
        List of pronounced phonemes.
    gt_cnncl_seq : list
        List of canonical phonemes.
    gt_md_lbl_seq : list
        Ground truth MD labels.
    pred_phn_seq : list
        Predicted phonemes.
    pred_md_lbl_seq : list
        Predicted MD labels.
    label_encoder : LabelEncoder
        The label encoder.
    """
    # input check
    if pred_phn_seq is None and pred_md_lbl_seq is None:
        raise ValueError('pred_phn_seq and pred_md_lbl_seq cannot be None at the same time.')
    length = len(gt_phn_seq)
    assert len(gt_cnncl_seq) == length
    assert len(gt_md_lbl_seq) == length
    assert pred_phn_seq is None or len(pred_phn_seq) == length
    assert pred_md_lbl_seq is None or len(pred_md_lbl_seq) == length

    # handle None input
    if pred_phn_seq is None:
        pred_phn_seq = ['NA'] * length
    if pred_md_lbl_seq is None:
        pred_md_lbl_seq = []
        for cnncl, pred_phn in zip(gt_cnncl_seq, pred_phn_seq):
            if cnncl == pred_phn:
                pred_md_lbl_seq.append(0)
            else:
                pred_md_lbl_seq.append(1)

    # correctness_seq
    correctness_seq = []
    for gt_md_lbl, pred_md_lbl in zip(gt_md_lbl_seq, pred_md_lbl_seq):
        if gt_md_lbl == pred_md_lbl:
            correctness_seq.append('c')
        else:
            correctness_seq.append('x')

    # decoded phonemes
    if label_encoder is not None:
        def decode_seq(seq):
            decoded_seq = []
            for p in seq:
                if p == -1:
                    decoded_seq.append('**')
                else:
                    decoded_seq.append(label_encoder.ind2lab[int(p)])
            return decoded_seq

        gt_phn_seq = decode_seq(gt_phn_seq)
        gt_cnncl_seq = decode_seq(gt_cnncl_seq)
        pred_phn_seq = decode_seq(pred_phn_seq)

    # initialize lines with the first ID line
    lines = [f'ID: {utt_id}\n']

    # template for each line
    line_template = '{:11s}: |' + '|'.join(['{:^4s}'] * length) + '|\n'

    lines.append(line_template.format('phn', *gt_phn_seq))
    lines.append(line_template.format('cnncl', *gt_cnncl_seq))
    lines.append(line_template.format('md_lbl', *[str(x) for x in gt_md_lbl_seq]))
    lines.append(line_template.format('pred_phn', *pred_phn_seq))
    lines.append(line_template.format('pred_md_lbl', *[str(x) for x in pred_md_lbl_seq]))
    lines.append(line_template.format('correctness', *correctness_seq))

    # scores
    for key, value in scores.items():
        lines.append(f'{key}: {value}\n')

    lines.append('\n')

    # write to file
    fp.writelines(lines)


def batch_write_md_results(
        fp,
        scores_list,
        utt_ids,
        gt_phn_seqs,
        gt_cnncl_seqs,
        gt_md_lbl_seqs,
        pred_phn_seqs=None,
        pred_md_lbl_seqs=None,
        label_encoder=None
):
    """
    Parameters
    ----------
    Batched version of write_md_results().
    """
    B = len(utt_ids)
    assert len(gt_phn_seqs) == B
    assert len(gt_cnncl_seqs) == B
    assert len(gt_md_lbl_seqs) == B
    assert pred_phn_seqs is None or len(pred_phn_seqs) == B
    assert pred_md_lbl_seqs is None or len(pred_phn_seqs) == B

    if pred_phn_seqs is None:
        pred_phn_seqs = [None] * B
    if pred_md_lbl_seqs is None:
        pred_md_lbl_seqs = [None] * B

    for i in range(B):
        write_md_results(
            fp,
            scores_list[i],
            utt_ids[i],
            gt_phn_seqs[i],
            gt_cnncl_seqs[i],
            gt_md_lbl_seqs[i],
            pred_phn_seqs[i],
            pred_md_lbl_seqs[i],
            label_encoder
        )
