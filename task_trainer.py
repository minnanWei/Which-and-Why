"""SFT training with generation-based task validation instead of evidence loss."""
import time
from pathlib import Path
from trl import SFTTrainer
from transformers import Trainer
from evaluation import generate_predictions, score_predictions


class TaskSFTTrainer(SFTTrainer):
    def __init__(self, *args, eval_max_new_tokens=1024, **kwargs):
        self.eval_max_new_tokens = eval_max_new_tokens
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError('当前生成式任务评估仅支持单进程训练；多卡请先单卡运行')
        self.task_eval_count = 0

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        data = eval_dataset if eval_dataset is not None else self.eval_dataset
        if data is None:
            raise ValueError('没有评估数据')
        records = list(data)
        self.task_eval_count += 1
        destination = Path(self.args.output_dir) / 'predictions' / (
            f'{metric_key_prefix}-step-{self.state.global_step}-call-{self.task_eval_count}.jsonl')
        started = time.perf_counter()
        model = self.accelerator.unwrap_model(self.model_wrapped)
        predictions = generate_predictions(
            model, self.processing_class, records, self.args.per_device_eval_batch_size,
            self.eval_max_new_tokens, destination, autocast_factory=self.accelerator.autocast)
        metrics = {f'{metric_key_prefix}_{key}': value for key, value in score_predictions(records, predictions).items()}
        metrics[f'{metric_key_prefix}_runtime'] = time.perf_counter() - started
        # Avoid TRL.log merging training token statistics into generation metrics.
        Trainer.log(self, metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
