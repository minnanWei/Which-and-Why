"""Plot Trainer JSONL, CSV, or trainer_state.json; optional trailing mean."""
import argparse
import csv
import json
import math
from pathlib import Path
from data_utils import project_path


def load_history(path):
    path = Path(path)
    if path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    if path.suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        return data['log_history'] if isinstance(data, dict) else data
    rows = []
    lines = path.read_text(encoding='utf-8').splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print('忽略中断时未写完的最后一条 JSONL 记录')
    return rows


def series(rows, key):
    points = {}
    for row in rows:
        try:
            step, value = int(row['step']), float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            points[step] = value  # Resume/re-evaluation: latest observation wins.
    return sorted(points.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-file', required=True)
    parser.add_argument('--output', default='outputs/loss.png')
    parser.add_argument('--window', type=int, default=1)
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be >= 1')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = load_history(project_path(args.log_file))
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for key, label in [('loss', 'Train loss'), ('eval_loss', 'Validation loss')]:
        points = series(rows, key)
        if not points:
            continue
        x, y = zip(*points)
        smooth = [sum(y[max(0, i-args.window+1):i+1]) / min(i+1, args.window) for i in range(len(y))]
        ax.plot(x, smooth, label=label, marker='.' if len(x) < 20 else None)
        plotted = True
    if not plotted:
        raise ValueError('没有可用的 loss/eval_loss 记录')
    ax.set(xlabel='Optimizer step', ylabel='Cross-entropy loss', title='LoRA SFT loss')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    target = project_path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
