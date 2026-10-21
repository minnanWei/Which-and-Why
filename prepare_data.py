"""Preview/validate raw data without loading a model."""
import argparse
from data_utils import convert_records, project_path, read_records, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--line-format', choices=['code', 'numbers'], default='numbers')
    args = parser.parse_args()
    source, target = project_path(args.input), project_path(args.output)
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError('输出文件已存在或与输入相同，请指定新输出路径')
    rows = convert_records(read_records(source), args.line_format)
    write_jsonl(target, rows)
    print(f'{len(rows)} samples -> {target}')


if __name__ == '__main__':
    main()
