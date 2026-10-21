"""Generation-based CWE and vulnerability-line evaluation; no evidence labels."""
import json
import re
from contextlib import nullcontext
from pathlib import Path
from data_utils import SYSTEM_PROMPT, extract_targets, user_prompt


def prepare_eval_records(rows, tokenizer, line_format, max_length, max_new_tokens, long_policy='error'):
    if max_new_tokens < 1 or max_new_tokens >= max_length:
        raise ValueError('评估输出预算必须大于 0 且小于 max_length')
    result, dropped, lengths = [], [], []
    for index, row in enumerate(rows, 1):
        try:
            lines, cwes = extract_targets(row, line_format)
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_prompt(row['func-before'], line_format)}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            lengths.append(len(ids))
            if len(ids) + max_new_tokens > max_length:
                if long_policy == 'error':
                    raise ValueError(f'提示 {len(ids)} + 输出预算 {max_new_tokens} 超过 max_length={max_length}')
                if long_policy != 'drop':
                    raise ValueError('long_policy 只能是 error/drop')
                dropped.append(index)
                continue
            result.append({'sample_index': index, 'input_ids': ids, 'messages': messages,
                           'lines': lines, 'cwes': cwes, 'line_format': line_format,
                           'source_lines': row['func-before'].splitlines()})
        except ValueError as exc:
            raise ValueError(f'评估样本 #{index}: {exc}') from exc
    if not result:
        raise ValueError('没有可评估样本')
    return result, {'input_count': len(rows), 'kept_count': len(result),
                    'dropped_indices_1based': dropped, 'max_prompt_tokens': max(lengths),
                    'max_new_tokens': max_new_tokens}


def parse_prediction(text, reference):
    """Read only the two task fields; evidence content and syntax are ignored."""
    decoder = json.JSONDecoder()
    candidate = None
    for match in re.finditer(re.escape('vulnerability lines:'), text):
        try:
            remaining = text[match.end():].lstrip()
            lines, end = decoder.raw_decode(remaining)
            remaining = remaining[end:].lstrip()
            if not remaining.startswith(';cweid:'):
                continue
            remaining = remaining[len(';cweid:'):].lstrip()
            cwes, end = decoder.raw_decode(remaining)
            if not remaining[end:].strip():
                candidate = (lines, cwes)
        except ValueError:
            continue
    try:
        if candidate is None:
            raise ValueError('未找到有效的 vulnerability lines:<JSON列表>;cweid:<JSON列表>')
        lines, cwes = candidate
        cwe_valid = isinstance(cwes, list) and all(
            isinstance(c, str) and re.fullmatch(r'CWE-\d+', c) for c in cwes)
        if reference['line_format'] == 'numbers':
            line_valid = isinstance(lines, list) and all(
                type(n) is int and 1 <= n <= len(reference['source_lines']) for n in lines)
        else:
            line_valid = isinstance(lines, list) and all(
                isinstance(line, str) and line in reference['source_lines'] for line in lines)
        errors = ([] if line_valid else ['漏洞行格式无效或越界']) + ([] if cwe_valid else ['CWE 格式无效'])
        return {'lines': list(dict.fromkeys(lines)) if line_valid else [],
                'cwes': list(dict.fromkeys(cwes)) if cwe_valid else [],
                'line_valid': line_valid, 'cwe_valid': cwe_valid,
                'format_valid': bool(line_valid and cwe_valid), 'parse_error': ';'.join(errors) or None}
    except (ValueError, TypeError) as exc:
        return {'lines': [], 'cwes': [], 'line_valid': False, 'cwe_valid': False,
                'format_valid': False, 'parse_error': str(exc)}


def score_predictions(references, predictions):
    if not references or len(references) != len(predictions):
        raise ValueError('评估样本与预测数量必须相同且非空')
    metrics = {'samples': len(references)}
    for key in ('lines', 'cwes'):
        tp = fp = fn = exact = 0
        prefix = 'line' if key == 'lines' else 'cwe'
        for reference, prediction in zip(references, predictions):
            gold, predicted = set(reference[key]), set(prediction[key])
            tp += len(gold & predicted)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
            exact += prediction[f'{prefix}_valid'] and predicted == gold
        metrics.update({f'{prefix}_precision': tp / (tp + fp) if tp + fp else 0.0,
                        f'{prefix}_recall': tp / (tp + fn) if tp + fn else 0.0,
                        f'{prefix}_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                        f'{prefix}_exact_match': exact / len(references)})
    metrics['format_valid_rate'] = sum(p['format_valid'] for p in predictions) / len(predictions)
    metrics['joint_exact_match'] = sum(
        p['format_valid'] and set(p['lines']) == set(r['lines']) and set(p['cwes']) == set(r['cwes'])
        for r, p in zip(references, predictions)) / len(references)
    metrics['joint_f1'] = (metrics['line_f1'] + metrics['cwe_f1']) / 2
    metrics['truncated_samples'] = sum(p.get('truncated', False) for p in predictions)
    return metrics


def generate_predictions(model, tokenizer, records, batch_size, max_new_tokens, output_file=None,
                         autocast_factory=nullcontext):
    import torch
    if batch_size < 1 or not records:
        raise ValueError('batch_size 必须为正，评估记录不能为空')
    was_training = model.training
    old_padding = tokenizer.padding_side
    old_cache = model.config.use_cache
    stream = None
    predictions = []
    try:
        if output_file:
            path = Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('w', encoding='utf-8')
        model.eval()
        model.config.use_cache = True
        tokenizer.padding_side = 'left'
        for start in range(0, len(records), batch_size):
            batch_rows = records[start:start + batch_size]
            inputs = tokenizer.pad([{'input_ids': row['input_ids'],
                                     'attention_mask': [1] * len(row['input_ids'])} for row in batch_rows],
                                   padding=True, return_tensors='pt').to(model.device)
            with torch.inference_mode(), autocast_factory():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id)
            for row, tokens in zip(batch_rows, output[:, inputs['input_ids'].shape[1]:]):
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                prediction = parse_prediction(text, row)
                prediction['text'] = text
                prediction['truncated'] = len(tokens) >= max_new_tokens and tokenizer.eos_token_id not in tokens.tolist()
                predictions.append(prediction)
                if stream:
                    stream.write(json.dumps({'sample_index': row['sample_index'],
                        'gold_lines': row['lines'], 'gold_cwes': row['cwes'], **prediction}, ensure_ascii=False) + '\n')
                    stream.flush()
            print(f'评估生成进度 {min(start + batch_size, len(records))}/{len(records)}', flush=True)
    finally:
        tokenizer.padding_side = old_padding
        model.config.use_cache = old_cache
        model.train(was_training)
        if stream:
            stream.close()
    return predictions
