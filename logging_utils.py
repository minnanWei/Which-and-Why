"""Flush every Trainer log to JSONL/CSV; survives ordinary interrupted runs."""
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from transformers import TrainerCallback

FIELDS = ['run_id', 'time_utc', 'step', 'epoch', 'loss', 'eval_loss', 'learning_rate', 'grad_norm',
          'eval_line_precision', 'eval_line_recall', 'eval_line_f1', 'eval_line_exact_match',
          'eval_cwe_precision', 'eval_cwe_recall', 'eval_cwe_f1', 'eval_cwe_exact_match',
          'eval_joint_f1', 'eval_joint_exact_match', 'eval_format_valid_rate', 'eval_truncated_samples']


class LossLoggerCallback(TrainerCallback):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.run_id = uuid4().hex

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = dict(logs, step=state.global_step, epoch=state.epoch,
                     run_id=self.run_id, time_utc=datetime.now(timezone.utc).isoformat())
        entry = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                 for k, v in entry.items()}
        with (self.directory / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str, allow_nan=False) + '\n')
            stream.flush()
        csv_path = self.directory / 'loss.csv'
        new = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open('a', encoding='utf-8', newline='') as stream:
            fieldnames = FIELDS
            if not new:
                with csv_path.open(encoding='utf-8', newline='') as existing:
                    fieldnames = next(csv.reader(existing))
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
            if new:
                writer.writeheader()
            writer.writerow(entry)
            stream.flush()
