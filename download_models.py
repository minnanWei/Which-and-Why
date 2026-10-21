"""Download original Qwen3 safetensors; repeated calls reuse HF download metadata."""
import argparse
import json
from data_utils import project_path

MODELS = {size: f'Qwen/Qwen3-{size}' for size in ('0.6B', '8B', '14B')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', choices=list(MODELS), default=['0.6B'])
    parser.add_argument('--model-id', help='Custom Qwen3 repository ID; overrides --sizes')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--revision', default='main', help='Branch, tag, or commit SHA')
    parser.add_argument('--endpoint', help='Optional Hugging Face-compatible endpoint')
    parser.add_argument('--tokenizer-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(endpoint=args.endpoint)
    repos = [args.model_id] if args.model_id else [MODELS[s] for s in args.sizes]
    for repo in repos:
        info = api.model_info(repo, revision=args.revision)
        destination = project_path(args.models_dir) / repo.replace('/', '--')
        # Presets use readable directory names; custom repos keep the namespace.
        if repo in MODELS.values():
            destination = project_path(args.models_dir) / repo.split('/')[-1]
        manifest = destination / 'download_manifest.json'
        if manifest.exists():
            old = json.loads(manifest.read_text(encoding='utf-8'))
            if old['repo_id'] != repo or old['commit'] != info.sha:
                raise ValueError('目标目录含另一个仓库/版本；请换 --models-dir，避免混合权重')
        patterns = ['*.json', '*.jinja', '*.txt', '*.model', 'README.md', 'LICENSE*']
        if not args.tokenizer_only:
            patterns.append('*.safetensors')
        snapshot_download(repo_id=repo, revision=info.sha, local_dir=destination,
                          allow_patterns=patterns, max_workers=args.workers, endpoint=args.endpoint)
        manifest.write_text(json.dumps({'repo_id': repo, 'commit': info.sha,
                            'tokenizer_only': args.tokenizer_only}, indent=2), encoding='utf-8')
        print(f'{repo}@{info.sha} -> {destination}')


if __name__ == '__main__':
    main()
