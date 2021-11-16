import functools

import torch
from torch.utils.tensorboard import SummaryWriter
import speechbrain as sb
import speechbrain.utils.data_utils
from speechbrain.utils.metric_stats import MetricStats, ErrorRateStats

import utils.alignment
import utils.md_scoring
from utils.md_metric_stats import MDMetricStats


class MDModel(sb.Brain):
    def __init__(self, label_encoder, **kwargs):
        super(MDModel, self).__init__(**kwargs)
        self.label_encoder = label_encoder

    def on_fit_start(self):
        super(MDModel, self).on_fit_start()
        self.tb_writer = SummaryWriter(log_dir=self.hparams.output_dir)

    def on_stage_start(self, stage, epoch=None):
        # initialize metric stats
        self.stats = {}

