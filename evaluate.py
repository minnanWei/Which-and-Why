"""Evaluate compact teacher CWE/evidence-line predictions on valid teacher records."""
import argparse
import json
from data_utils import DATA_DIR, project_path, read_records
from evaluation import generate_predictions, prepare_eval_records, score_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--adapter-path', required=True)
    parser.add_argument('--data-file', default=DATA_DIR + '/test.jsonl')
    parser.add_argument('--output-dir', required=True, help='New empty output directory')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=6144)
    parser.add_argument('--max-length', type=int, help='Defaults to base model native context limit')
    parser.add_argument('--long-policy', choices=['error', 'drop'], default='error')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu'], default='auto')
    args = parser.parse_args()
    output_dir = project_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('评测输出目录非空，请换新目录，避免覆盖已有结果')
    from predict import load_adapter
    model, tokenizer, run = load_adapter(args.model_path, args.adapter_path, args.device, args.load_in_4bit)
    max_length = args.max_length if args.max_length is not None else model.config.max_position_embeddings
    if max_length > model.config.max_position_embeddings:
        raise ValueError('max_length 超过模型原生上下文长度')
    records, stats = prepare_eval_records(read_records(project_path(args.data_file)), tokenizer,
        max_length, args.max_new_tokens, args.long_policy)
    predictions = generate_predictions(model, tokenizer, records, args.batch_size,
                                       args.max_new_tokens, output_dir / 'predictions.jsonl')
    metrics = score_predictions(records, predictions)
    report = {'data_file': str(project_path(args.data_file)), 'model_path': args.model_path,
              'adapter_path': args.adapter_path, 'data_stats': stats, 'metrics': metrics}
    (output_dir / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
