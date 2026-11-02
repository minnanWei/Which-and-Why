"""Preview/validate raw data without loading a model."""
import argparse
from data_utils import convert_records, project_path, read_records, valid_records, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, target = project_path(args.input), project_path(args.output)
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError('输出文件已存在或与输入相同，请指定新输出路径')
    records, stats = valid_records(read_records(source))
    rows = convert_records(records)
    write_jsonl(target, rows)
    print(f'{len(rows)} valid samples -> {target}; skipped {len(stats["invalid_indices_1based"])} invalid teachers')


if __name__ == '__main__':
    main()
