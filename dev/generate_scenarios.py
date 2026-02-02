import json
import os
import re
import random
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Using a fast, cheap model for prompt generation
# MODEL = "arcee-ai/trinity-large-preview:free"
MODEL = "google/gemini-2.5-flash-lite"

# --- CONFIGURATION ---
TASK_TYPES = [
    "bug fix with stack trace", "performance optimization", "refactor / restructure",
    "rename symbol across codebase", "dead code analysis", "add new feature",
    "add tests", "fix failing tests", "update documentation", "config file changes",
    "dependency migration", "api endpoint implementation", "security audit fix",
    "resolve git conflicts", "codebase understanding", "code comparison", "remove feature",
    "devops - deployment scaling", "devops - continuous integration"
]

# COMPLEXITY_MODIFIERS = [
#     "single file", "multi-file with grep", "requires bash exploration",
#     "user rejects first attempt", "tool error then recovery",
#     "ambiguous request needing clarification", "cross-module dependency logic",
#     "git merge conflicts in progress",
# ]

COMPLEXITY_MODIFIERS = [
    "user rejects first attempt with a specific technical reason", 
    "tool call should return an error (e.g. file not found)",
    "ambiguous request needing clarification before acting", 
    "cross-module dependency logic",
    "user pivots mid-way",
]

COMPLEXITY_MODIFIERS += [""] * len(COMPLEXITY_MODIFIERS)


CODEBASE_DOMAINS = [
    "web api (fastapi)", "LLM training (jax)", "LLM training (torch)", "cli tool",
    "data processing pipelines", "python testing framework", "config management",
    "distributed systems", "transformers attention implementations",
    "LLM evaluation suites", "implementing machine learning optimizers",
    "creating hyperparameter sweep scripts", "machine learning (vision)",
    "data science", "tokenizers", "Rust", "meaching learning (reinforcement learning)"
]

# --- JSON SCHEMA ---
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

def call_llm(messages, response_format=None, temperature=0.8):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format
    }
    res = requests.post(API_URL, headers=headers, json=payload)
    if res.status_code != 200: return None
    
    content = res.json()["choices"][0]["message"]["content"]
    # Robust JSON extraction from markdown
    match = re.search(r'(\{.*\})', content, re.DOTALL)
    return match.group(1) if match else content

def generate_batch(batch_idx):
    # Select random constraints to ensure diversity
    tasks = random.sample(TASK_TYPES, 5)
    complexities = random.sample(COMPLEXITY_MODIFIERS, 1)[0]
    domains = random.sample(CODEBASE_DOMAINS, 5)
    
    prompt = f"""
    generate 20 highly realistic, diverse scenario prompts for a coding agent.
    
    constraints:
    - task types: {', '.join(tasks)}
    - domains: {', '.join(domains)}

    each prompt must describe:
    1. the codebase context (e.g. "in a jax-based training loop...")
    2. the user's request (mix casual and professional tones)

    output as a json object with a 'scenarios' array.
    """.strip()

    response = call_llm([{"role": "user", "content": prompt}], response_format=SCENARIO_SCHEMA)
    if not response: return []
    
    try:
        data = json.loads(response)
        batch = data.get("scenarios", [])
        for item in batch:
            item["id"] = f"b{batch_idx}_{item['id']}"
            item["notes"] = complexities
        return batch
    except:
        return []
    
def main():
    output_path = Path("prompts/all_prompts.jsonl")
    output_path.parent.mkdir(exist_ok=True)

    num_batches = 100
    max_workers = 10

    completed = 0
    next_idx = 0

    with open(output_path, "a") as f, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = set()

        while completed < num_batches:
            while len(futures) < max_workers:
                futures.add(ex.submit(generate_batch, next_idx))
                next_idx += 1

            done, futures = futures.pop(), futures
            batch = done.result()

            if batch:
                for scenario in batch:
                    f.write(json.dumps(scenario) + "\n")
                f.flush()
                completed += 1
                print(f"completed {completed}/{num_batches}")


if __name__ == "__main__":
    main()
