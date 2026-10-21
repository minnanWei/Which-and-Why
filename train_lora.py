"""Qwen3 LoRA SFT with verified assistant masks and durable loss logs."""
import argparse
import importlib.metadata
import json
import os
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from data_utils import (ROOT, check_overlap, convert_records, project_path,
                        read_records, tokenize_records)


@dataclass
class TrainSettings:
    model_path: str = 'models/Qwen3-0.6B'
    train_file: str = 'dataset/train.jsonl'
    eval_file: str = 'dataset/validation.jsonl'
    output_dir: str = 'outputs/Qwen3-0.6B-LoRA'
    line_format: str = 'numbers'
    max_length: int = 4096
    long_policy: str = 'error'
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: str = 'all-linear'
    batch_size: int = 1
    eval_batch_size: int = 1
    eval_max_new_tokens: int = 1024
    gradient_accumulation_steps: int = 24
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    gradient_checkpointing: bool = True
    load_in_4bit: bool = False
    dtype: str = 'auto'
    device: str = 'auto'
    seed: int = 42


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='JSON configuration; paths are project-relative')
    parser.add_argument('--preflight', action='store_true', help='Validate/tokenize all data, without weights or training')
    parser.add_argument('--resume-from-checkpoint', help='Checkpoint directory or latest')
    for field in fields(TrainSettings):
        option = '--' + field.name.replace('_', '-')
        if field.type is bool:
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        else:
            parser.add_argument(option, type=field.type, default=argparse.SUPPRESS)
    cli = vars(parser.parse_args())
    config = cli.pop('config')
    preflight, resume = cli.pop('preflight'), cli.pop('resume_from_checkpoint')
    values = json.loads(project_path(config).read_text(encoding='utf-8-sig')) if config else {}
    values.update(cli)
    settings = TrainSettings(**values)  # Unknown config keys fail instead of being ignored.
    validate(settings)
    return settings, preflight, resume


def validate(cfg):
    if cfg.line_format not in ('code', 'numbers') or cfg.long_policy not in ('error', 'drop'):
        raise ValueError('line_format=code/numbers; long_policy=error/drop')
    if cfg.dtype not in ('auto', 'bf16', 'fp16', 'fp32') or cfg.device not in ('auto', 'cpu'):
        raise ValueError('dtype=auto/bf16/fp16/fp32; device=auto/cpu')
    for name in ('r', 'lora_alpha', 'max_length', 'batch_size', 'eval_batch_size',
                 'gradient_accumulation_steps', 'eval_max_new_tokens', 'logging_steps', 'eval_steps', 'save_steps', 'save_total_limit'):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
            raise ValueError(f'{name} 必须为正整数')
    if cfg.eval_file and cfg.eval_max_new_tokens >= cfg.max_length:
        raise ValueError('eval_max_new_tokens 必须小于 max_length')
    if cfg.eval_file and cfg.save_steps % cfg.eval_steps:
        raise ValueError('启用最佳模型选择时 save_steps 必须是 eval_steps 的整数倍')
    if not 0 <= cfg.lora_dropout < 1 or not 0 <= cfg.warmup_ratio <= 1:
        raise ValueError('lora_dropout 应在 [0,1)，warmup_ratio 应在 [0,1]')
    if cfg.learning_rate <= 0 or cfg.num_train_epochs <= 0 or cfg.max_steps == 0 or cfg.max_steps < -1:
        raise ValueError('学习率/epochs 必须为正；max_steps 为 -1 或正整数')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    cfg, preflight, resume = parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('当前生成式评估仅支持单进程训练')
    from evaluation import prepare_eval_records
    from transformers import AutoConfig, AutoTokenizer, set_seed
    model_path = project_path(cfg.model_path)
    if not (model_path / 'config.json').is_file():
        raise FileNotFoundError(f'模型配置不存在: {model_path}；先运行 download_models.py')
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if model_config.model_type != 'qwen3':
        raise ValueError('本项目模板针对 Qwen3 稠密文本模型；其他架构需单独适配')
    if cfg.max_length > model_config.max_position_embeddings:
        raise ValueError('max_length 超过模型原生上下文长度')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    tokenizer.chat_template = (ROOT / 'chat_template.jinja').read_text(encoding='utf-8')
    tokenizer.padding_side = 'right'
    tokenizer.eos_token = '<|im_end|>'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<|endoftext|>'
    train_raw = read_records(project_path(cfg.train_file))
    train_messages = convert_records(train_raw, cfg.line_format)
    train_data, train_stats = tokenize_records(train_messages, tokenizer, cfg.max_length, cfg.long_policy)
    eval_data, eval_stats = None, None
    if cfg.eval_file:
        eval_raw = read_records(project_path(cfg.eval_file))
        eval_data, eval_stats = prepare_eval_records(
            eval_raw, tokenizer, cfg.line_format, cfg.max_length, cfg.eval_max_new_tokens, cfg.long_policy)
        check_overlap(train_raw, eval_raw)
    stats = {'train': train_stats, 'validation': eval_stats}
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if preflight:
        print('Preflight passed: training targets/masks and evidence-free validation prompts checked; no weights loaded.')
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig
    from task_trainer import TaskSFTTrainer
    from logging_utils import LossLoggerCallback

    set_seed(cfg.seed)
    cpu = cfg.device == 'cpu'
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError('未检测到 CUDA；请安装 CUDA 版 torch，或仅小规模调试时显式 --device cpu --dtype fp32')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not cpu:
        torch.cuda.set_device(local_rank)
    dtype = cfg.dtype
    if dtype == 'auto':
        dtype = 'fp32' if cpu else ('bf16' if torch.cuda.is_bf16_supported() else 'fp16')
    if cpu and dtype != 'fp32':
        raise ValueError('CPU 调试使用 fp32')
    if dtype == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('当前 GPU 不支持 bf16；使用 --dtype fp16')
    if cfg.load_in_4bit and cpu:
        raise ValueError('本项目 QLoRA 路径需要 NVIDIA CUDA GPU')
    output = project_path(cfg.output_dir)
    if resume == 'latest':
        resume = get_last_checkpoint(str(output)) if output.is_dir() else None
        if resume is None:
            raise ValueError('没有可恢复的 checkpoint')
    elif resume:
        resume = str(project_path(resume))
    if resume and not (Path(resume) / 'trainer_state.json').is_file():
        raise ValueError('恢复训练需要含 trainer_state.json/优化器状态的 checkpoint，不能使用 final_adapter')
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('output_dir 非空；请更换输出目录或使用 --resume-from-checkpoint')
    previous = output / 'run_config.json'
    if resume:
        if not previous.exists():
            raise ValueError('续训须使用原 output_dir（缺少 run_config.json）')
        old = json.loads(previous.read_text(encoding='utf-8'))
        for key in ('model_path', 'train_file', 'eval_file', 'line_format', 'r', 'lora_alpha',
                    'lora_dropout', 'target_modules', 'load_in_4bit', 'max_length', 'seed', 'eval_max_new_tokens'):
            if old.get(key, 1024 if key == 'eval_max_new_tokens' else None) != getattr(cfg, key):
                raise ValueError(f'续训配置 {key} 与原训练不一致')
    is_main = int(os.environ.get('RANK', '0')) == 0
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / ('resume_config.json' if resume else 'run_config.json'), asdict(cfg))
        write_json(output / 'data_stats.json', stats)
        write_json(output / 'versions.json', {p: importlib.metadata.version(p) for p in
                   ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets')})
        shutil.copy2(ROOT / 'chat_template.jinja', output / 'chat_template.jinja')

    model_kwargs = {'torch_dtype': {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[dtype],
                    'attn_implementation': 'sdpa', 'local_files_only': True}
    if cfg.load_in_4bit:
        model_kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_kwargs['torch_dtype']), device_map={'': local_rank})
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing,
                                               gradient_checkpointing_kwargs={'use_reentrant': False})
    targets = 'all-linear' if cfg.target_modules == 'all-linear' else cfg.target_modules.split(',')
    model = get_peft_model(model, LoraConfig(r=cfg.r, lora_alpha=cfg.lora_alpha,
                          lora_dropout=cfg.lora_dropout, target_modules=targets,
                          bias='none', task_type='CAUSAL_LM'))
    if is_main:
        model.print_trainable_parameters()
    args = SFTConfig(
        output_dir=str(output), run_name=output.name, seed=cfg.seed,
        per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs, max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate, lr_scheduler_type='cosine', warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay, max_grad_norm=1.0, optim='adamw_torch',
        logging_strategy='steps', logging_steps=cfg.logging_steps, logging_first_step=True,
        report_to=['tensorboard'], logging_dir=str(output / 'tensorboard'),
        eval_strategy='steps' if eval_data else 'no', eval_steps=cfg.eval_steps,
        save_strategy='steps', save_steps=cfg.save_steps, save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=bool(eval_data), metric_for_best_model='eval_joint_f1' if eval_data else None,
        greater_is_better=True, bf16=dtype == 'bf16', fp16=dtype == 'fp16', use_cpu=cpu,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=0, dataloader_pin_memory=not cpu,
        max_length=cfg.max_length, packing=False, assistant_only_loss=True,
        dataset_kwargs={'skip_prepare_dataset': True}, eos_token='<|im_end|>',
        ddp_find_unused_parameters=False,
    )
    trainer = TaskSFTTrainer(model=model, args=args, processing_class=tokenizer,
                         eval_max_new_tokens=cfg.eval_max_new_tokens,
                         train_dataset=Dataset.from_list(train_data),
                         eval_dataset=Dataset.from_list(eval_data) if eval_data else None,
                         callbacks=[LossLoggerCallback(output / 'logs')])
    # Persist state/history even when interrupted. Checkpoints alone allow full resume.
    try:
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_metrics('train', result.metrics)
        if eval_data:
            trainer.save_metrics('eval', trainer.evaluate())
        trainer.save_model(str(output / 'final_adapter'))
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(output / 'final_adapter')
            write_json(output / 'selection.json', {
                'best_model_checkpoint': trainer.state.best_model_checkpoint,
                'best_metric': trainer.state.best_metric,
                'metric_name': 'eval_joint_f1',
                'note': 'final_adapter is checkpoint with highest task F1, or last weights if none was saved',
            })
    finally:
        trainer.save_state()
        if trainer.is_world_process_zero():
            write_json(output / 'log_history.json', trainer.state.log_history)


if __name__ == '__main__':
    main()
