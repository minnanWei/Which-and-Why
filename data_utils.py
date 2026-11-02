"""Compact teacher records, weighted assistant tokens and exact group scheduling."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent
DATA_DIR = 'dataset'
SYSTEM_PROMPT = 'You are a software security analyst. Identify the single CWE and explain the code-level vulnerability using only the supplied function.'


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_records(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(s) for s in text.splitlines() if s.strip()]
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f'{path}: 需要非空 JSON/JSONL 对象记录')
    return rows


def valid_records(rows):
    kept, rejected, positions = [], [], []
    for index, row in enumerate(rows, 1):
        evidence = row.get('joint-evidence')
        if not isinstance(evidence, dict) or type(evidence.get('sample_valid')) is not bool:
            raise ValueError(f'样本 #{index}: joint-evidence.sample_valid 必须是布尔值')
        if evidence['sample_valid']:
            kept.append(row)
            positions.append(index)
        else:
            rejected.append(index)
    if not kept:
        raise ValueError('没有 sample_valid=true 的样本')
    cwe_counts = dict(Counter(r['joint-evidence']['final_cwe'] for r in kept))
    return kept, {'raw_count': len(rows), 'valid_count': len(kept), 'invalid_count': len(rejected),
                  'valid_cwe_categories': len(cwe_counts), 'valid_cwe_counts': cwe_counts, 'invalid_indices_1based': rejected,
                  'valid_indices_1based': positions}


def source_lines(code):
    numbered = [re.fullmatch(r'\s*(\d+)\s*\| ?(.*)', s) for s in code.splitlines()]
    if any(numbered):
        return {int(m[1]): m[2] for m in numbered if m}
    return dict(enumerate(code.splitlines(), 1))


def user_prompt(code):
    if not isinstance(code, str) or not code.strip():
        raise ValueError('func 必须是非空字符串')
    if not any(re.fullmatch(r'\s*\d+\s*\| ?.*', s) for s in code.splitlines()):
        code = '\n'.join(f'{i:04d} | {s}' for i, s in enumerate(code.splitlines(), 1))
    return (
        'Analyze this function. Return only a JSON object with final_cwe, evidence_lines, '
        'sample_valid, rejection_reason, code_evidence, vulnerability_mechanism and cwe_rationale. '
        'final_cwe is one CWE identifier. evidence_lines is a list of {"lines":[integer source line numbers],'
        '"code":"original source fragment"}; enumerate lines explicitly. Explain each fragment in '
        'code_evidence, then briefly explain the mechanism and CWE rationale. '
        'Use sample_valid=true and rejection_reason=null when evidence supports the prediction. '
        'Do not include Markdown.\n<source_code>\n' + code + '\n</source_code>'
    )


def extract_targets(row):
    evidence = row['joint-evidence']
    cwes = ast.literal_eval(row['cwe_ids']) if isinstance(row['cwe_ids'], str) and row['cwe_ids'].startswith('[') else row['cwe_ids']
    cwes = [cwes] if isinstance(cwes, str) else cwes
    if not isinstance(cwes, list) or len(cwes) != 1 or not isinstance(cwes[0], str) or not re.fullmatch(r'CWE-\d+', cwes[0]):
        raise ValueError('cwe_ids 必须包含唯一 CWE')
    if evidence['final_cwe'] != cwes[0]:
        raise ValueError('有效教师样本 final_cwe 与原始 cwe_ids 不一致')
    if row['cwe_group'] not in ('head', 'middle', 'tail'):
        raise ValueError('cwe_group 必须是 head/middle/tail')
    source = source_lines(row['func'])
    fragments = evidence['evidence_lines']
    if not isinstance(fragments, list) or not fragments:
        raise ValueError('有效教师样本 evidence_lines 不能为空')
    lines = set()
    for fragment in fragments:
        ids = fragment['lines']
        if not isinstance(ids, list) or not ids or any(type(n) is not int or n not in source for n in ids):
            raise ValueError('evidence_lines 引用了无效函数行号')
        lines.update(ids)
    return sorted(lines), cwes


def convert_record(row):
    extract_targets(row)
    teacher = row['joint-evidence']
    fields = ['final_cwe', 'evidence_lines', 'sample_valid', 'rejection_reason',
              'code_evidence', 'vulnerability_mechanism', 'cwe_rationale']
    answer, spans = '{', {}
    for index, key in enumerate(fields):
        answer += (',' if index else '') + json.dumps(key) + ':'
        start = len(answer)
        answer += json.dumps(teacher[key], ensure_ascii=False, separators=(',', ':'))
        if key in ('final_cwe', 'evidence_lines'):
            spans[key] = [start, len(answer)]
    answer += '}'
    return {'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                         {'role': 'user', 'content': user_prompt(row['func'])},
                         {'role': 'assistant', 'content': answer}],
            'loss_spans': spans, 'cwe_group': row['cwe_group']}


def convert_records(rows):
    return [convert_record(row) for row in rows]


def code_hash(row):
    source = '\n'.join(s.rstrip() for s in source_lines(row['func']).values())
    return hashlib.sha256(source.encode()).hexdigest()


def check_overlap(train_rows, eval_rows):
    overlap = {code_hash(r) for r in train_rows} & {code_hash(r) for r in eval_rows}
    if overlap:
        raise ValueError(f'训练与验证存在 {len(overlap)} 个重复函数')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def tokenize_records(rows, tokenizer, max_length, long_policy='error', cwe_loss_weight=3.0, evidence_loss_weight=2.0):
    result, dropped, lengths = [], [], []
    multipliers = {'final_cwe': cwe_loss_weight, 'evidence_lines': evidence_loss_weight}
    weighted_tokens = Counter()
    for index, row in enumerate(rows, 1):
        rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=False)
        encoded = tokenizer.apply_chat_template(
            row['messages'], tokenize=True, add_generation_prompt=False, return_dict=True,
            return_assistant_tokens_mask=True, tokenizer_kwargs={'return_offsets_mapping': True})
        ids, mask = encoded['input_ids'], encoded['assistant_masks']
        answer = row['messages'][-1]['content']
        target = tokenizer.decode([t for t, m in zip(ids, mask) if m], skip_special_tokens=False)
        if len(ids) != len(mask) or target != answer + '<|im_end|>':
            raise ValueError(f'样本 #{index}: assistant mask 未覆盖完整答案和 EOS')
        lengths.append(len(ids))
        if len(ids) > max_length:
            if long_policy == 'error':
                raise ValueError(f'样本 #{index}: {len(ids)} tokens 超过 max_length={max_length}')
            dropped.append(index)
            continue
        answer_start = rendered.rfind(answer)
        spans = [(answer_start + a, answer_start + b, multipliers[k]) for k, (a, b) in row['loss_spans'].items()]
        weights = [max([1.0] + [w for a, b, w in spans if start < b and end > a]) if active else 0.0
                   for (start, end), active in zip(encoded['offset_mapping'], mask)]
        for key, (a, b) in row['loss_spans'].items():
            count = sum(bool(m and start < answer_start + b and end > answer_start + a)
                        for (start, end), m in zip(encoded['offset_mapping'], mask))
            if not count:
                raise ValueError(f'样本 #{index}: 没有找到 {key} 的监督 token')
            weighted_tokens[key] += count
        result.append({'input_ids': ids, 'assistant_masks': mask, 'loss_weights': weights,
                       'messages': row['messages'], 'cwe_group': row['cwe_group']})
    if not result:
        raise ValueError('没有可训练样本')
    return result, {'input_count': len(rows), 'kept_count': len(result), 'dropped_indices_1based': dropped,
                    'max_tokens': max(lengths), 'weighted_field_tokens': dict(weighted_tokens),
                    'group_counts': dict(Counter(r['cwe_group'] for r in result))}


class GroupRatioSampler:
    """Exact ratio each cycle; reshuffle each pool only after exhaustion."""
    def __init__(self, groups, ratio=(5, 3, 1), seed=42, samples_per_update=27):
        self.ratio, self.seed, self.epoch = tuple(ratio), seed, 0
        self.pools = {name: [i for i, g in enumerate(groups) if g == name]
                      for name in ('head', 'middle', 'tail')}
        if any(not pool for pool in self.pools.values()):
            raise ValueError('比例调度需要三个组都有有效训练样本')
        cycles = max(math.ceil(len(pool) / n) for pool, n in zip(self.pools.values(), ratio))
        unit = samples_per_update // math.gcd(samples_per_update, sum(ratio))
        self.cycles = math.ceil(cycles / unit) * unit

    def __len__(self):
        return self.cycles * sum(self.ratio)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = {g: ids.copy() for g, ids in self.pools.items()}
        positions = {g: 0 for g in pools}
        for ids in pools.values():
            rng.shuffle(ids)
        for _ in range(self.cycles):
            cycle = []
            for (group, pool), count in zip(pools.items(), self.ratio):
                for _ in range(count):
                    if positions[group] == len(pool):
                        rng.shuffle(pool)
                        positions[group] = 0
                    cycle.append(pool[positions[group]])
                    positions[group] += 1
            rng.shuffle(cycle)
            yield from cycle

    def statistics(self):
        return {'ratio': dict(zip(self.pools, self.ratio)), 'cycles_per_epoch': self.cycles,
                'samples_per_epoch': len(self), 'exposures_per_epoch': dict(zip(self.pools, [self.cycles * n for n in self.ratio])),
                'unique_group_counts': {g: len(ids) for g, ids in self.pools.items()}}
