import torch
import utils.md_scoring


class MDMetricStats:
    def __init__(self):
        self.clear()

    def clear(self):
        self.md_metric_keys = []
        self.ids = []
        self.scores_list = []

    def append(self, ids, **kwargs):
        self.ids.extend(ids)
        self.scores_list.extend(utils.md_scoring.batch_seq_md_scoring(**kwargs))
        if len(self.md_metric_keys):
            self.md_metric_keys = list(self.scores_list[0].keys())

    def summarize(self):
        if len(self.md_metric_keys) == 0:
            raise ValueError('No metrics saved yet')

        mean_scores = {}
        for key in self.md_metric_keys:
            mean_scores[key] = [scores[key] for scores in self.scores_list]
            mean_scores[key] = torch.mean(torch.tensor(mean_scores[key]))