"""Generation-based evaluation of compact JSON CWE and evidence-line predictions."""
import json
import re
from contextlib import nullcontext
from pathlib import Path
from data_utils import SYSTEM_PROMPT, extract_targets, source_lines, user_prompt, valid_records


def prepare_eval_records(rows, tokenizer, max_length, max_new_tokens, long_policy='error'):
    rows, filtered = valid_records(rows)
    result, dropped, lengths = [], [], []
    for index, row in zip(filtered['valid_indices_1based'], rows):
        lines, cwes = extract_targets(row)
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_prompt(row['func'])}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        lengths.append(len(ids))
        if len(ids) + max_new_tokens > max_length:
            if long_policy == 'error':
                raise ValueError(f'评估样本 #{index}: 提示 {len(ids)} + 输出预算 {max_new_tokens} 超过 {max_length}')
            dropped.append(index)
            continue
        result.append({'sample_index': index, 'input_ids': ids, 'messages': messages,
                       'lines': lines, 'cwes': cwes, 'cwe_group': row['cwe_group'],
                       'source_line_numbers': list(source_lines(row['func']))})
    if not result:
        raise ValueError('没有可评估样本')
    return result, {'teacher_filter': filtered, 'kept_count': len(result),
                    'dropped_indices_1based': dropped, 'max_prompt_tokens': max(lengths),
                    'max_new_tokens': max_new_tokens}


def parse_prediction(text, reference):
    result = {'lines': [], 'cwes': [], 'line_valid': False, 'cwe_valid': False,
              'format_valid': False, 'parse_error': None}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return {**result, 'parse_error': str(exc)}
    if not isinstance(value, dict):
        return {**result, 'parse_error': '预测需要 JSON 对象'}
    cwe = value.get('final_cwe')
    cwe_valid = isinstance(cwe, str) and re.fullmatch(r'CWE-\d+', cwe) is not None
    abstain = cwe is None and value.get('sample_valid') is False
    fragments = value.get('evidence_lines')
    available = set(reference['source_line_numbers'])
    line_valid = isinstance(fragments, list)
    lines = []
    if line_valid:
        for fragment in fragments:
            ids = fragment.get('lines') if isinstance(fragment, dict) else None
            if not isinstance(ids, list) or any(type(n) is not int or n not in available for n in ids):
                line_valid = False
                break
            lines.extend(ids)
    errors = ([] if line_valid else ['evidence_lines 无效或越界']) + ([] if cwe_valid or abstain else ['final_cwe 无效'])
    return {**result, 'lines': sorted(set(lines)) if line_valid else [],
            'cwes': [cwe] if cwe_valid else [], 'line_valid': line_valid,
            'cwe_valid': cwe_valid or abstain, 'format_valid': line_valid and (cwe_valid or abstain),
            'parse_error': ';'.join(errors) or None}



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
    metrics['cwe_accuracy'] = metrics['cwe_exact_match']
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
