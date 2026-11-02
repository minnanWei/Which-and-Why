"""Weighted SFT, exact group exposure and generation-based task validation."""
from itertools import islice
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import Trainer
from data_utils import GroupRatioSampler
from evaluation import generate_predictions, score_predictions


class WeightedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch = self.tokenizer.pad(
            [{'input_ids': e['input_ids'], 'attention_mask': [1] * len(e['input_ids'])} for e in examples],
            padding=True, return_tensors='pt')
        labels = torch.full_like(batch['input_ids'], -100)
        weights = torch.zeros_like(batch['input_ids'], dtype=torch.float32)
        for i, example in enumerate(examples):
            size = len(example['input_ids'])
            labels[i, :size] = torch.tensor([t if m else -100 for t, m in zip(example['input_ids'], example['assistant_masks'])])
            weights[i, :size] = torch.tensor(example['loss_weights'], dtype=torch.float32)
        return {**batch, 'labels': labels, 'loss_weights': weights}


class TaskSFTTrainer(SFTTrainer):
    def __init__(self, *args, eval_max_new_tokens=1024, group_ratio=(5, 3, 1), **kwargs):
        self.eval_max_new_tokens = eval_max_new_tokens
        self.group_ratio = group_ratio
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError('当前生成式任务评估仅支持单进程训练；多卡请先单卡运行')
        self.task_eval_count = 0
        # compute_loss consumes a denominator for the entire accumulation window.
        self.model_accepts_loss_kwargs = True

    def _get_train_sampler(self, train_dataset=None):
        data = self.train_dataset if train_dataset is None else train_dataset
        return GroupRatioSampler(data['cwe_group'], self.group_ratio, self.args.seed,
                                 self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches = list(islice(epoch_iterator, num_batches))
        denominator = sum(batch['loss_weights'][:, 1:].sum() for batch in batches).to(device) if batches else None
        return batches, denominator

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')[:, 1:]
        weights = inputs.pop('loss_weights')[:, 1:]
        outputs = model(**inputs, use_cache=False)
        supervised = labels != -100
        # Compute CE only where supervision exists, avoiding a prompt-sized FP32 copy.
        logits = outputs.logits[:, :-1, :][supervised].float()
        losses = F.cross_entropy(logits, labels[supervised], reduction='none')
        active_weights = weights[supervised]
        denominator = active_weights.sum() if num_items_in_batch is None else num_items_in_batch
        loss = (losses * active_weights).sum() / denominator
        return (loss, outputs) if return_outputs else loss

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
