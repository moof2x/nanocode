"""
tier 1a: transform coding instruction datasets into nanocode Edit tool call format.

sources: allenai/tulu-3-sft-personas-code, bigcode/self-oss-instruct-sc2-exec-filter-50k
output: jsonl matching nanocode rollout format (messages with tool_call / tool_result)

usage:
  python dev/process_tulu.py --dataset tulu --output rollouts/tier1a_tulu.jsonl
  python dev/process_tulu.py --dataset selfoss --output rollouts/tier1a_selfoss.jsonl
  python dev/process_tulu.py --dataset both --output rollouts/tier1a_combined.jsonl
  python dev/process_tulu.py --dataset both --output rollouts/tier1a_styled.jsonl --restyle --workers 8
"""
import json, re, ast, copy, argparse, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import requests
from tqdm import tqdm

VLLM_URL = "http://localhost:8000/v1/chat/completions"

SOUL = """
You are nanocode, a coding agent trained as part of the nanocode project - a minimal educational open-source library for end-to-end
training of a coding agent, from scratch, and in pure JAX. You serve as a pristine example of an
accessible and highly customizable coding partner, embedded in your user's system and equipped
with a broad range of capabilities to assist your user.
Style:

Your communication style is direct, clear, with a collaborative and casual friendly tone. You communicate with the user exclusively in lowercase; the only exceptions where you may use capital letters are for proper nouns, variable names, writing code or referring to code, or executing tool calls. For example, the following are valid: "i see a JSONDecode error," "the function returns None", or "Read: file_path: foo.py". Otherwise, all code generation must follow standard language conventions. You will use warmth markers (e.g. cool, sweet, right on, huh) to set a casual, collaborative, and helpful tone, and avoid robotic responses. You will use informal contractions, e.g. "gonna, lemme, donezo, finito, gotchu etc.".
You do not ask follow-up questions.

 You avoid all forms of sycophancy, apology, remorse, or regret. Your responses are honest and curious, for example: "huh, that's an interesting point." is strongly preferred over "great! you're absolutely right!".

""".strip()

RESTYLE_SYSTEM = f"""You are restyling assistant messages in a coding agent conversation to match a specific personality.

The agent personality (SOUL):
{SOUL}

Your task: Given a conversation between a user and an assistant (coding agent), rewrite ONLY the assistant's natural language content strings to match the SOUL above.

Rules:
- Rewrite assistant content to be in line with the SOUL document.
- If the original content is empty string and there's a tool_call, generate a brief preamble explaining what you're about to do given the user request and the code context.
- If the original content is empty string and it's a final assistant message after a tool_result, generate a brief confirmation of what was done alongside a confirmatory statement
- Keep content concise - 1-2 sentences max for most turns.
- Capitals only for: proper nouns, variable names, code references, tool names.
- Do NOT modify anything about tool_call or tool_result fields.
- Output ONLY the rewritten content strings as a JSON array, one per assistant message, in order.

Example input conversation:
[
  {{"role": "user", "content": "Write a function that checks if a number is prime"}},
  {{"role": "assistant", "content": "Sure, I'll create a function to check for prime numbers.", "tool_call": {{"name": "Edit", ...}}}},
  {{"role": "tool_result", "content": "    1→def is_prime(n):..."}},
  {{"role": "assistant", "content": "The function has been created successfully. It handles edge cases for numbers less than 2."}}
]


Output a JSON array of strings, one per assistant message in order."""

RESTYLE_STRUCTURED_OUTPUT = {
    "type": "object",
    "required": ["assistant_contents"],
    "additionalProperties": False,
    "properties": {
        "assistant_contents": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

CRITIQUE_STRUCTURED_OUTPUT = {
    "type": "object",
    "required": ["rating", "critique"],
    "additionalProperties": False,
    "properties": {
        "rating": {"type": "integer"},
        "critique": {"type": "string"}
    }
}


def call_vllm(model, messages, json_schema=None, temperature=0.3, max_tokens=2048):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_schema:
        pass  # structured_outputs disabled — xgrammar corrupts output with this model
    resp = requests.post(VLLM_URL, json=payload, timeout=120)
    if resp.status_code != 200:
        raise Exception(f"vllm error ({resp.status_code}): {resp.text}")
    result = resp.json()
    content = result["choices"][0]["message"]["content"]
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    content = re.sub(r"^```(?:json)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def get_assistant_indices(messages):
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def format_conversation_for_restyle(rollout):
    lines = []
    for m in rollout["messages"]:
        role = m["role"]
        content = m.get("content", "")
        if role == "user":
            lines.append(f"[user]: {content}")
        elif role == "tool_result":
            lines.append(f"[tool_result]: {content[:200]}{'...' if len(content) > 200 else ''}")
        elif role == "assistant":
            tc = m.get("tool_call")
            tc_str = f" (tool_call: {tc['name']}({json.dumps(tc['args'])[:100]}))" if tc else ""
            lines.append(f"[assistant]: \"{content}\"{tc_str}")
    return "\n".join(lines)


def validate_restyle_unchanged(original, styled):
    problems = []
    orig_msgs = original["messages"]
    styled_msgs = styled["messages"]
    if len(orig_msgs) != len(styled_msgs):
        return [f"message count changed: {len(orig_msgs)} -> {len(styled_msgs)}"]
    for i, (o, s) in enumerate(zip(orig_msgs, styled_msgs)):
        if o["role"] != s["role"]:
            problems.append(f"msg {i} role changed: {o['role']} -> {s['role']}")
        if o["role"] == "tool_result":
            if o.get("content") != s.get("content"):
                problems.append(f"msg {i} tool_result content changed")
        if o["role"] == "user":
            if o.get("content") != s.get("content"):
                problems.append(f"msg {i} user content changed")
        if "tool_call" in o:
            if json.dumps(o["tool_call"], sort_keys=True) != json.dumps(s.get("tool_call", {}), sort_keys=True):
                problems.append(f"msg {i} tool_call changed")
        elif "tool_call" in s:
            problems.append(f"msg {i} tool_call added")
    for key in original:
        if key == "messages":
            continue
        if original[key] != styled.get(key):
            problems.append(f"top-level key '{key}' changed")
    return problems


def critique_rollout_soul(model, rollout):
    assistant_contents = [m.get("content", "") for m in rollout["messages"] if m.get("role") == "assistant"]
    critique_prompt = f"""Evaluate these assistant messages against the SOUL document.

SOUL:
{SOUL}

Assistant messages (in order):
{json.dumps(assistant_contents, indent=2)}

Critique ONLY the assistant's natural language content above. Does it adhere to the SOUL's prescribed principles?
Output JSON: {{"rating": 1-10, "critique": "1-5 sentence summary of issues"}}

Be strict. The assistant should not be penalized for using uppercase when referring to variable names,
code, or proper nouns (e.g. "JAX", "None", "ModuleNotFoundError", "JSONDecodeError" are all fine).
Anything below a 9 will be rejected for not capturing the SOUL sufficiently."""

    content = call_vllm(
        model,
        messages=[{"role": "user", "content": critique_prompt}],
        json_schema=CRITIQUE_STRUCTURED_OUTPUT,
        temperature=0.3,
        max_tokens=512,
    )
    return json.loads(content)


def restyle_rollout(model, rollout, max_retries=3):
    messages = rollout["messages"]
    assistant_indices = get_assistant_indices(messages)
    if not assistant_indices:
        return rollout, "no assistant messages"

    conv_text = format_conversation_for_restyle(rollout)
    original_contents = [messages[i].get("content", "") for i in assistant_indices]

    prompt = f"""Here is the conversation to restyle:

{conv_text}

There are {len(assistant_indices)} assistant messages to restyle.
Original assistant contents (in order):
{json.dumps(original_contents, indent=2)}

Rewrite each assistant content string to match the SOUL. Output exactly {len(assistant_indices)} strings."""

    problem_prompt = ""
    first_rejected = None
    for attempt in range(max_retries):
        try:
            content = call_vllm(
                model,
                messages=[
                    {"role": "system", "content": RESTYLE_SYSTEM},
                    {"role": "user", "content": prompt + problem_prompt}
                ],
                json_schema=RESTYLE_STRUCTURED_OUTPUT,
                temperature=0.7,
                max_tokens=4096,
            )
            parsed = json.loads(content)
            if isinstance(parsed, str):
                new_contents = [parsed]
            elif isinstance(parsed, list):
                new_contents = parsed
            else:
                new_contents = parsed["assistant_contents"]

            if len(new_contents) != len(assistant_indices):
                problem_prompt = f"\n\nYou returned {len(new_contents)} strings but there are {len(assistant_indices)} assistant messages. Return exactly {len(assistant_indices)} strings."
                print(f"    attempt {attempt + 1}: wrong count ({len(new_contents)} vs {len(assistant_indices)})")
                continue

            styled = copy.deepcopy(rollout)
            for idx, ai in enumerate(assistant_indices):
                styled["messages"][ai]["content"] = new_contents[idx]

            problems = validate_restyle_unchanged(rollout, styled)
            if problems:
                problem_prompt = f"\n\nValidation failed: {'; '.join(problems)}. You must ONLY change assistant content strings. Try again."
                print(f"    attempt {attempt + 1}: validation failed: {problems}")
                continue

            critique = critique_rollout_soul(model, styled)
            if critique["rating"] < 9:
                if first_rejected is None:
                    first_rejected = styled
                problem_prompt = f"\n\nPrevious attempt scored {critique['rating']}/10. Feedback: {critique['critique']}. Revise to better match the SOUL."
                print(f"    attempt {attempt + 1}: critique {critique['rating']}/10 - {critique['critique']}")
                continue

            return styled, first_rejected

        except json.JSONDecodeError as e:
            if len(assistant_indices) == 1 and not content.startswith(("[", "{")):
                new_contents = [content.strip().strip('"').strip("'")]
                styled = copy.deepcopy(rollout)
                styled["messages"][assistant_indices[0]]["content"] = new_contents[0]
                problems = validate_restyle_unchanged(rollout, styled)
                if not problems:
                    return styled, first_rejected
            print(f"    attempt {attempt + 1}: json error: {e}\n    raw output: {repr(content[:300])}")
            problem_prompt = f"\n\nJSON parse error: {e}. Return valid JSON."
        except Exception as e:
            print(f"    attempt {attempt + 1}: error: {e}   raw output: {content[:300]}")
            problem_prompt = ""

    return None, first_rejected


def restyle_single(model, idx, rollout):
    styled, rejected = restyle_rollout(model, rollout)
    return styled, rejected


def format_line_numbers(code: str) -> str:
    lines = code.split('\n')
    return '\n'.join(f'{i+1:>5}→{line}' for i, line in enumerate(lines))

def extract_function_name(code: str) -> str | None:
    match = re.search(r'def\s+(\w+)\s*\(', code)
    if match:
        return match.group(1)
    match = re.search(r'class\s+(\w+)', code)
    if match:
        return match.group(1).lower()
    return None

def extract_code_tulu(text: str):
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        before = text[:match.start()].strip().lower()
        after = text[match.end():].strip().lower()
        return before, code, after
    # no fences — find first line that looks like code
    lines = text.split('\n')
    code_start = None
    for i, line in enumerate(lines):
        if re.match(r'^(def |class |import |from |@)', line.strip()):
            code_start = i
            break
    if code_start is None:
        return '', '', ''
    before = '\n'.join(lines[:code_start]).strip().lower()
    code = '\n'.join(lines[code_start:]).strip()
    return before, code, ''

def extract_code_and_dialogue_selfoss(text: str) -> tuple[str, str, str] | None:
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if not match:
        return None
    code = match.group(1).strip()
    before = text[:match.start()].strip()
    after = text[match.end():].strip()
    return before, code, after

def infer_filename(code: str, fallback_idx: int) -> str:
    name = extract_function_name(code)
    if name:
        return f"{name}.py"
    return f"solution_{fallback_idx}.py"

def make_edit_rollout(user_content: str, code: str, before_text: str, after_text: str, idx: int) -> dict | None:
    code = code.strip()
    if not code or len(code) < 20:
        return None
    lines = code.split('\n')
    if len(lines) > 200:
        return None
    if not validate_code(code):
        return None
    filename = infer_filename(code, idx)
    formatted = format_line_numbers(code)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": before_text,
         "tool_call": {"name": "Edit", "args": {"file_path": filename, "new_string": code}}},
        {"role": "tool_result", "content": formatted},
    ]
    if after_text:
        messages.append( {"role": "assistant", "content": after_text})
    return {"messages": messages}

def transform_tulu(max_rows: int = None) -> list[dict]:
    ds = load_dataset('allenai/tulu-3-sft-personas-code', split='train')
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    rollouts = []
    skipped_no_code = 0
    for idx, row in enumerate(ds):
        msgs = row['messages']
        user_content = None
        assistant_raw = None
        for m in msgs:
            if m['role'] == 'user':
                user_content = m['content']
            elif m['role'] == 'assistant':
                assistant_raw = m['content']
        if not user_content or not assistant_raw:
            continue
        before, code, after = extract_code_tulu(assistant_raw)
        if not code:
            skipped_no_code += 1
            # i think there's one assistant refusal in this dataset
            
            continue
        rollout = make_edit_rollout(user_content, code, before, after, idx)
        if rollout:
            rollout['_source'] = 'tulu-3-sft-personas-code'
            rollout['_source_idx'] = idx
            rollouts.append(rollout)
        if idx % 5000 == 0:
            print(f"  tulu: processed {idx}, kept {len(rollouts)}, skipped {skipped_no_code}")
    print(f"  tulu: final — kept {len(rollouts)}, skipped {skipped_no_code}")
    return rollouts

def transform_selfoss(max_rows: int = None) -> list[dict]:
    ds = load_dataset('bigcode/self-oss-instruct-sc2-exec-filter-50k', split='train')
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    rollouts = []
    skipped_no_code_block = 0
    for idx, row in enumerate(ds):
        user_content = row.get('instruction') or ''
        response = row.get('response') or ''
        if not user_content or not response:
            continue
        extracted = extract_code_and_dialogue_selfoss(response)
        if extracted is None:
            skipped_no_code_block += 1
            continue
        before_text, code, after_text = extracted
        rollout = make_edit_rollout(user_content, code, before_text, after_text, idx)
        if rollout:
            rollout['_source'] = 'self-oss-instruct-sc2-exec-filter-50k'
            rollout['_source_idx'] = idx
            rollouts.append(rollout)
        if idx % 5000 == 0:
            print(f"  selfoss: processed {idx}, kept {len(rollouts)}, skipped (no code block) {skipped_no_code_block}")
    print(f"  selfoss: final — kept {len(rollouts)}, skipped (no code block) {skipped_no_code_block}")
    return rollouts

import warnings as _warnings
def validate_code(code: str) -> bool:
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code)
        return True
    except SyntaxError:
        return False

def validate_rollout(rollout: dict) -> list[str]:
    problems = []
    messages = rollout.get('messages', [])
    if not messages:
        return ['no messages']
    if messages[0].get('role') != 'user':
        problems.append('first message not from user')
    for i, msg in enumerate(messages):
        if 'tool_call' in msg:
            if i + 1 >= len(messages):
                problems.append('tool_call at end without tool_result')
            elif messages[i + 1].get('role') != 'tool_result':
                problems.append('tool_call not followed by tool_result')
        if msg.get('role') == 'tool_result':
            content = msg.get('content', '')
            lines = content.split('\n')
            for line in lines:
                if '→' in line:
                    match = re.match(r'^( *)(\d+)→', line)
                    if match:
                        total = len(match.group(1)) + len(match.group(2))
                        if total != 5:
                            problems.append(f'line prefix wrong length: {total}')
                            break
        if msg.get('role') == 'assistant' and 'tool_call' in msg:
            tc = msg['tool_call']
            if tc.get('name') == 'Edit' and '→' in tc.get('args', {}).get('old_string', ''):
                problems.append('Edit old_string contains line numbers')
            if tc.get('name') == 'Edit' and '→' in tc.get('args', {}).get('new_string', ''):
                problems.append('Edit new_string contains line numbers')
    return problems

def discover_schema(dataset_name: str):
    """print first 3 rows of a dataset so you can verify field names"""
    ds = load_dataset(dataset_name, split='train')
    print(f"dataset: {dataset_name}")
    print(f"num rows: {len(ds)}")
    print(f"columns: {ds.column_names}")
    for i in range(min(3, len(ds))):
        print(f"\n--- row {i} ---")
        row = ds[i]
        for k, v in row.items():
            val_str = str(v)[:300]
            print(f"  {k}: {val_str}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['tulu', 'selfoss', 'both'], default='both')
    parser.add_argument('--output', type=str, default='rollouts/tier1a.jsonl')
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--discover', type=str, default=None, help='print schema of a dataset and exit')
    parser.add_argument('--restyle', action='store_true', help='run restyle pass on assistant content via local vllm')
    parser.add_argument('--restyle-model', type=str, default='Qwen/Qwen3-30B-A3B-Instruct-2507', help='model name for vllm restyle server')
    parser.add_argument('--workers', type=int, default=4, help='number of concurrent workers for restyle')
    parser.add_argument('--max-retries', type=int, default=3, help='max retries per rollout during restyle')
    args = parser.parse_args()

    if args.discover:
        discover_schema(args.discover)
        return

    rollouts = []
    if args.dataset in ('tulu', 'both'):
        print("transforming tulu...")
        rollouts.extend(transform_tulu(args.max_rows))
    if args.dataset in ('selfoss', 'both'):
        print("transforming selfoss...")
        rollouts.extend(transform_selfoss(args.max_rows))

    random.shuffle(rollouts)

    from nanocode.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()

    valid_rollouts = []
    valid, invalid, too_long, tok_fail = 0, 0, 0, 0
    for r in rollouts:
        problems = validate_rollout(r)
        if problems:
            invalid += 1
            continue
        try:
            ids, _ = tokenizer.render_conversation(r)
            if len(ids) > 4096:
                too_long += 1
                continue
        except Exception as e:
            tok_fail += 1
            continue
        valid_rollouts.append(r)
        valid += 1

    print(f"\nfiltering: {valid} valid, {invalid} invalid, {too_long} too long, {tok_fail} tokenizer failures")

    if args.restyle:
        styled_rollouts = []
        preference_pairs = []
        restyle_failed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(restyle_single, args.restyle_model, i, r): i for i, r in enumerate(valid_rollouts)}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="restyle")
            for future in pbar:
                try:
                    idx = futures[future]
                    styled, _ = future.result()
                    if styled:
                        styled_rollouts.append(styled)
                        original = valid_rollouts[idx]
                        preference_pairs.append({
                            "chosen": {"messages": styled["messages"]},
                            "rejected": {"messages": original["messages"]},
                            "_source": styled.get("_source", ""),
                            "_source_idx": styled.get("_source_idx", ""),
                        })
                    else:
                        restyle_failed += 1
                except Exception as e:
                    print(f"  exception: {e}")
                    restyle_failed += 1
                pbar.set_postfix(ok=len(styled_rollouts), fail=restyle_failed, pref=len(preference_pairs))
        valid_rollouts = styled_rollouts

        if preference_pairs:
            pref_path = Path(args.output).with_suffix(".pref.jsonl")
            with open(pref_path, 'w') as f:
                for p in preference_pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + '\n')
            print(f"written {len(preference_pairs)} preference pairs to {pref_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for r in valid_rollouts:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"written {len(valid_rollouts)} rollouts to {output_path}")

    if valid_rollouts:
        sample = valid_rollouts[0]
        print(f"\nsample rollout:")
        print(json.dumps(sample, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
