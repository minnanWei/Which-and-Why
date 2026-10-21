"""Raw vulnerability records -> single-turn SFT; no training dependencies."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = '你是一名代码安全审计专家。请基于给定函数完成行级漏洞预测和 CWE 漏洞类型识别，并给出依据。'


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_records(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    if not text.strip():
        raise ValueError(f'{path}: 文件为空')
    if text.lstrip().startswith('['):
        rows = json.loads(text)
    else:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: 无效 JSONL: {exc}') from exc
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f'{path}: 需要非空 JSON 数组或每行一个对象的 JSONL')
    return rows


def list_field(value, name):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f'{name}: 需要列表或列表字符串') from exc
    if not isinstance(value, list):
        raise ValueError(f'{name}: 需要列表，不接受空值或标量')
    return value


def user_prompt(code, line_format='code'):
    if not isinstance(code, str) or not code.strip():
        raise ValueError('func-before 必须是非空字符串')
    if line_format == 'numbers':
        source = '\n'.join(f'{i}: {line}' for i, line in enumerate(code.splitlines(), 1))
        rule = 'vulnerability lines 为函数内从 1 开始的行号列表，空行也计数；行首编号不属于源码。'
    elif line_format == 'code':
        source = code
        rule = 'vulnerability lines 为漏洞代码行的字符串列表，逐字保留原代码行。'
    else:
        raise ValueError('line_format 只能是 code 或 numbers')
    return (
        '请分析以下函数，为我完成行级漏洞预测和漏洞类型识别任务。\n'
        '严格输出：evidence:<依据>;vulnerability lines:<JSON列表>;cweid:<JSON列表>。\n'
        f'{rule} cweid 为 CWE 编号列表；无漏洞时两个列表均为空。\n'
        'evidence 为 JSON 字符串，以转义换行和引号。不要输出额外标题或 Markdown。\n'
        f'<source_code>\n{source}\n</source_code>'
    )


def extract_targets(row, line_format='numbers'):
    """Read task labels without requiring or looking at joint-evidence."""
    missing = [key for key in ('func-before', 'vulnerability lines', 'cwe-id') if key not in row]
    if missing:
        raise ValueError(f'缺少字段: {missing}')
    code = row['func-before']
    user_prompt(code, line_format)
    cwes = list_field(row['cwe-id'], 'cwe-id')
    if any(not isinstance(c, str) or not re.fullmatch(r'CWE-\d+', c) for c in cwes):
        raise ValueError('cwe-id 中每个元素必须形如 CWE-476；无漏洞用 []')
    lines = list_field(row['vulnerability lines'], 'vulnerability lines')
    if line_format == 'numbers':
        if any(type(n) is not int or n < 1 or n > len(code.splitlines()) for n in lines):
            raise ValueError('漏洞行号必须为函数内从 1 开始的整数，不能越界；请先转换文件绝对行号')
    else:
        if any(not isinstance(line, str) or not line.strip() or '\n' in line or '\r' in line for line in lines):
            raise ValueError('code 模式需要非空单行字符串列表；整数行号请使用 numbers 模式')
        if any(line not in code.splitlines() for line in lines):
            raise ValueError('漏洞代码行必须逐字出现在 func-before 中（包括缩进）')
    if '<|im_start|>' in code or '<|im_end|>' in code:
        raise ValueError('内容包含聊天控制 token，请清理或转义后重试')
    return lines, cwes


def convert_record(row, line_format='code'):
    lines, cwes = extract_targets(row, line_format)
    evidence = row.get('joint-evidence')
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('训练样本 joint-evidence 必须是非空字符串；验证/测试不需要该字段')
    if '<|im_start|>' in evidence or '<|im_end|>' in evidence:
        raise ValueError('内容包含聊天控制 token，请清理或转义后重试')
    dump = lambda value: json.dumps(value, ensure_ascii=False)
    answer = f'evidence:{dump(evidence)};vulnerability lines:{dump(lines)};cweid:{dump(cwes)}'
    return {'messages': [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt(row['func-before'], line_format)},
        {'role': 'assistant', 'content': answer},
    ]}


def convert_records(rows, line_format='code'):
    result = []
    for index, row in enumerate(rows, 1):
        try:
            result.append(convert_record(row, line_format))
        except ValueError as exc:
            raise ValueError(f'样本 #{index}: {exc}') from exc
    return result


def code_hash(row):
    # Whitespace-normalized exact clone check; not a near-clone detector.
    return hashlib.sha256(' '.join(row['func-before'].split()).encode()).hexdigest()


def check_overlap(train_rows, eval_rows):
    overlap = {code_hash(r) for r in train_rows} & {code_hash(r) for r in eval_rows}
    if overlap:
        raise ValueError(f'训练集与验证集存在 {len(overlap)} 个重复函数，先重新划分数据以避免泄漏')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def tokenize_records(rows, tokenizer, max_length, long_policy='error'):
    result, dropped, lengths = [], [], []
    for index, row in enumerate(rows, 1):
        encoded = tokenizer.apply_chat_template(
            row['messages'], tokenize=True, add_generation_prompt=False,
            return_dict=True, return_assistant_tokens_mask=True,
        )
        ids, mask = encoded['input_ids'], encoded['assistant_masks']
        target = tokenizer.decode([t for t, m in zip(ids, mask) if m], skip_special_tokens=False)
        expected = row['messages'][-1]['content'] + '<|im_end|>'
        if len(ids) != len(mask) or not any(mask) or target != expected:
            raise ValueError(f'样本 #{index}: assistant mask 未准确覆盖完整答案及结束标记')
        lengths.append(len(ids))
        if len(ids) > max_length:
            if long_policy == 'error':
                raise ValueError(f'样本 #{index}: {len(ids)} tokens 超过 max_length={max_length}；增大上限或显式选择 drop')
            if long_policy != 'drop':
                raise ValueError('long_policy 只能是 error 或 drop')
            dropped.append(index)
            continue
        result.append({'input_ids': ids, 'assistant_masks': mask, 'messages': row['messages']})
    if not result:
        raise ValueError('没有可训练样本')
    return result, {'input_count': len(rows), 'kept_count': len(result),
                    'dropped_indices_1based': dropped, 'max_tokens': max(lengths)}
