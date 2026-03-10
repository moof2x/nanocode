"""
This is a hacky script for generating a whole bunch of prompts to seed synthetic agentic dataset generation.

Usage (quick local debugging):

> llama-server \
        -hf ggml-org/gpt-oss-20b-GGUF \
        --port 8000 \
        --ctx-size 16384 \
        -ngl 99 \
        -fa on \
        --jinja
> python dev/generate_scenarios.py --output=/tmp/scenarios.jsonl --workers=1 --batches=10 --model=ggml-org/gpt-oss-20b-GGUF --dry-run

Bump workers and batches/try different models for scaling this out. 
"""
import json, random, argparse, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

TASK_TYPES = [
    "Bug fix with stack trace", "Performance optimization", "Refactor / restructure",
    "Rename symbol across codebase", "Dead code analysis", "Add new feature",
    "Add tests", "Fix failing tests", "Update documentation", "Config file changes",
    "Dependency migration", "API endpoint implementation", "Security audit fix",
    "Resolve git conflicts", "Codebase understanding", "Code comparison", "Remove feature",
    "DevOps - deployment scaling", "DevOps - continuous integration"
]

# complexity modifiers add realistic failure modes.
COMPLEXITY_MODIFIERS = [
    "User rejects first attempt with a specific technical reason",
    "Tool call should return an error (e.g. file not found)",
    "Ambiguous request needing clarification before acting",
    "Cross-module dependency logic",
    "User pivots mid-way",
]
COMPLEXITY_MODIFIERS += [""] * len(COMPLEXITY_MODIFIERS)

CODEBASE_DOMAINS = [
    "Web API (FastAPI)", "LLM training (JAX)", "LLM training (PyTorch)", "CLI tool",
    "Data processing pipelines", "Python testing framework", "Config management",
    "Distributed systems", "Transformers attention implementations",
    "LLM evaluation suites", "Implementing machine learning optimizers",
    "Creating hyperparameter sweep scripts", "Machine learning (vision)",
    "Data science", "Tokenizers", "Rust", "Machine learning (reinforcement learning)"
]

SCENARIO_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "scenario_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "scenarios": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "prompt": {"type": "string"}
                        },
                        "required": ["id", "prompt"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["scenarios"],
            "additionalProperties": False
        }
    }
}


VLLM_URL = "http://localhost:8000/v1/chat/completions"

def call_vllm(prompt, model, temperature=0.8):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "response_format": SCENARIO_SCHEMA,
    }
    res = requests.post(VLLM_URL, json=payload, timeout=120)
    if res.status_code != 200:
        print(f"  request failed ({res.status_code}): {res.content[:200]}")
        return None
    return res.json()["choices"][0]["message"]["content"]


def generate_batch(batch_idx, model):
    tasks = random.sample(TASK_TYPES, 5)
    complexity = random.choice(COMPLEXITY_MODIFIERS)
    domains = random.sample(CODEBASE_DOMAINS, 5)

    prompt = f"""generate 20 highly realistic, diverse scenario prompts for a coding agent.

constraints:
- task types: {', '.join(tasks)}
- domains: {', '.join(domains)}

each prompt must describe:
1. the codebase context (e.g. "in a jax-based training loop...")
2. the user's request (mix casual and professional tones)

output as a json object with a 'scenarios' array."""

    response = call_vllm(prompt, model)
    if not response:
        return []
    try:
        data = json.loads(response)
        batch = data.get("scenarios", [])
        for item in batch:
            item["id"] = f"b{batch_idx}_{item['id']}"
            item["notes"] = complexity
        return batch
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  batch {batch_idx}: parse error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=str, default='prompts/all_prompts.jsonl')
    parser.add_argument('--batches', type=int, default=100)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.5-9B')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)

    if args.dry_run:
        scenarios = generate_batch(0, args.model)
        print(json.dumps(scenarios, indent=2))
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_scenarios = 0
    with open(output_path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(generate_batch, i, args.model): i for i in range(args.batches)}
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch = future.result()
            except Exception as e:
                print(f"  batch {batch_idx}: exception: {e}")
                continue
            for scenario in batch:
                f.write(json.dumps(scenario) + "\n")
            f.flush()
            total_scenarios += len(batch)
            print(f"  batch {batch_idx}: {len(batch)} scenarios (total: {total_scenarios})")

    print(f"\ndone — {total_scenarios} scenarios written to {output_path}")


if __name__ == "__main__":
    main()
