import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import speechbrain as sb
from speechbrain.utils.train_logger import FileTrainLogger

logger = logging.getLogger(__name__)


class MDModel(sb.Brain):
    def __init__(self, label_encoder, **kwargs):
        super(MDModel, self).__init__(**kwargs)
        self.label_encoder = label_encoder

    def on_fit_start(self):
        super(MDModel, self).on_fit_start()
        train_logger_save_file = Path(self.hparams.output_dir) / 'train_log.txt'
        self.train_logger = FileTrainLogger(save_file=train_logger_save_file)
        self.tb_writer = SummaryWriter(log_dir=self.hparams.output_dir)

    def on_stage_start(self, stage, epoch=None):
        # initialize metric stats
        self.stats_loggers = {}


    def on_stage_end(self, stage, stage_loss, epoch=None, log_metrics={}):
        if log_metrics is None:
            log_metrics = {}
        stage_name = str(stage).split('.')[1].lower()

        # get metrics
        log_metrics['loss'] = round(stage_loss, 3)
        for metric_key in self.hparams.metric_keys:  # metric_key = 'PER' or 'md.F1'
            metric_key_list = metric_key.split('.')
            stats = self.stats_loggers[f'{metric_key_list[0].lower()}_stats']
            stats_key = None if len(metric_key_list) == 1 else metric_key_list[1]

            log_metrics[metric_key] = round(float(stats.summarize(stats_key)), 2)

        if stage == sb.Stage.TRAIN or stage == sb.Stage.VALID:
            # log stats
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            train_logger_stats = {
                'stats_meta': {'stage': stage_name, 'epoch': epoch, 'lr': lr},
                f'{stage_name}_stats': log_metrics
            }
            self.train_logger.log_stats(**train_logger_stats)

            # TB log
            for metric_key, metric_value in log_metrics.items():
                self.tb_writer.add_scalar(f'{metric_key}/{stage_name}', metric_value, global_step=epoch)

            # save checkpoint after the VALID stage
            if stage == sb.Stage.VALID:
                self.checkpointer.save_and_keep_only(
                    meta=log_metrics, min_keys=[self.hparams.min_key],
                )

        if stage == sb.Stage.TEST:
            test_output_dir = Path(self.hparams.output_dir) / 'test_output'
            test_output_dir.mkdir(exist_ok=True)

            # log test metrics
            log_str = ', '.join([f'{k}: {v}' for k, v in log_metrics.items()])
            logger.info(f'Best epoch: {self.hparams.epoch_counter.current}, {log_str}')
            metric_values = []
            with open(test_output_dir / 'test_metrics.txt', 'w') as f:
                for metric_key, metric_value in log_metrics.items():
                    f.write(f'{metric_key}: {metric_value}\n')
                    metric_values.append(str(metric_value))
                f.write('\t'.join(metric_values) + '\n')
                logger.info(f'Test metrics saved to {f.name}')

            # save all stats loggers
            for stats_key, stats_logger in self.stats_loggers.items():
                stats_key = stats_key.replace('_stats', '')
                with open(test_output_dir / f'{stats_key}.txt', 'w') as f:
                    stats_logger.write_stats(f)
                    logger.info(f'{stats_key} stats saved to {f.name}')

