import torch

from utils.metric_stats.md_metric_stats import binary_seq_md_scoring, boundary_md_scoring
from utils.metric_stats.boundary_metric_stats import boundary_scoring

def compute_fa_metrics(dataset):
    metrics = {}

    for data_sample in dataset:
        utt_id = data_sample['id']
        sample_metrics = {}

        # boundary metrics for forced alignment results
        fa_boundary_seq = data_sample['fa_boundary_seq']
        gt_boundary_seq = data_sample['gt_boundary_seq']
        boundary_metrics = boundary_scoring(fa_boundary_seq, gt_boundary_seq)
        for key, metric in boundary_metrics.items():
            sample_metrics['boundary.' + key] = metric

        # MD metrics
        gt_md_lbl_seq = data_sample['plvl_gt_md_lbl_seq']
        pred_md_lbl_seq = torch.zeros_like(gt_md_lbl_seq)
        md_metrics = binary_seq_md_scoring(pred_md_lbl_seq, gt_md_lbl_seq)
        for key, metric in md_metrics.items():
            sample_metrics['MD.' + key] = metric

        # boundary MD metrics
        boundary_md_metrics = boundary_md_scoring(fa_boundary_seq, gt_boundary_seq, pred_md_lbl_seq, gt_md_lbl_seq)
        for key, metric in boundary_md_metrics.items():
            sample_metrics['boundary_MD.' + key] = metric

        # save metrics
        for key, metric in sample_metrics.items():
            if key in metrics:
                metrics[key].append(metric)
            else:
                metrics[key] = [metric]

    # compute the average metrics
    mean_metrics = {key: torch.mean(torch.tensor(metrics[key])).item() for key in metrics}

    return mean_metrics

def compute_asr_metrics(dataset):
    metrics = {}

    for i, data_sample in enumerate(dataset):

        utt_id = data_sample['id']
        sample_metrics = {}

        # boundary metrics for forced alignment results
        dnn_hmm_boundary_seq = data_sample['ext_dnn_hmm_boundary_seq']
        gt_boundary_seq = data_sample['gt_boundary_seq']
        boundary_metrics = boundary_scoring(dnn_hmm_boundary_seq, gt_boundary_seq)
        for key, metric in boundary_metrics.items():
            sample_metrics['boundary.' + key] = metric

        # # MD metrics
        gt_md_lbl_seq = data_sample['plvl_gt_md_lbl_seq']
        pred_md_lbl_seq = data_sample['ext_plvl_dnn_hmm_md_lbl_seq']
        md_metrics = binary_seq_md_scoring(pred_md_lbl_seq, gt_md_lbl_seq)
        for key, metric in md_metrics.items():
            sample_metrics['MD.' + key] = metric

        # boundary MD metrics
        boundary_md_metrics = boundary_md_scoring(dnn_hmm_boundary_seq, gt_boundary_seq, pred_md_lbl_seq, gt_md_lbl_seq)
        for key, metric in boundary_md_metrics.items():
            sample_metrics['boundary_MD.' + key] = metric

        # save metrics
        for key, metric in sample_metrics.items():
            if key in metrics:
                metrics[key].append(metric)
            else:
                metrics[key] = [metric]

    # compute the average metrics
    mean_metrics = {key: torch.mean(torch.tensor(metrics[key])).item() for key in metrics}

    return mean_metrics


def compute_dnn_hmm_metrics(dataset):
    fa_metrics = compute_fa_metrics(dataset)
    for key, metric in fa_metrics.items():
        print(f'fa.{key}: {round(metric, 2)}')

    asr_metrics = compute_asr_metrics(dataset)
    for key, metric in asr_metrics.items():
        print(f'asr.{key}: {round(metric, 2)}')
