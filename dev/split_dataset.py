import json, random, argparse
from pathlib import Path
from huggingface_hub import HfApi

def split_jsonl(input_path, test_frac=0.05, seed=42):
    with open(input_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    random.seed(seed)
    random.shuffle(rows)
    n_test = int(len(rows) * test_frac)
    test, train = rows[:n_test], rows[n_test:]
    stem = Path(input_path).with_suffix('')
    outputs = {}
    for split, data in [('train', train), ('test', test)]:
        out = Path(f"{stem}_{split}.jsonl")
        with open(out, 'w') as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"{split}: {len(data)} → {out}")
        outputs[split] = out
    return outputs

def push_to_hub(files, repo_id):
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    for split, path in files.items():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=repo_id, repo_type="dataset")
        print(f"uploaded {path.name} → {repo_id}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--test-frac', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hub-id', type=str, default=None, help="HF repo id to push to, e.g. smohammadi/nanocode-rollouts")
    args = parser.parse_args()
    outputs = split_jsonl(args.input, args.test_frac, args.seed)
    if args.hub_id:
        push_to_hub(outputs, args.hub_id)
