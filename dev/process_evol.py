"""
tier 1b: transform evol-codealpaca-v1 into nanocode Read → Edit tool call format.

source: theblackcat102/evol-codealpaca-v1
output: jsonl matching nanocode rollout format (messages with Read then Edit tool_call / tool_result)

the instruction contains existing code the user wants modified — this becomes a synthetic file
that the assistant Reads. the output contains the modified code — the diff between original and
modified becomes the Edit.

usage:
  python dev/process_evol.py --output rollouts/tier1b_evol.jsonl
  python dev/process_evol.py --output rollouts/tier1b_evol.jsonl --restyle --workers 8
"""
import json, re, ast, difflib, argparse, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import requests
from tqdm import tqdm

from dev.process_tulu import (
    format_line_numbers, validate_code, validate_rollout, infer_filename,
    VLLM_URL, SOUL, RESTYLE_SYSTEM, RESTYLE_STRUCTURED_OUTPUT, CRITIQUE_STRUCTURED_OUTPUT,
    call_vllm, restyle_single,
)


def extract_code_block(text: str) -> str | None:
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_code_from_field(text: str, allow_bare=False) -> tuple[str, str]:
    """extract code and surrounding prose from a field.
    returns (prose, code). only extracts fenced code blocks unless allow_bare=True."""
    code = extract_code_block(text)
    if code:
        prose = re.sub(r'```(?:python)?\s*\n.*?```', '', text, flags=re.DOTALL).strip()
        return prose, code
    if not allow_bare:
        return text, ''
    lines = text.split('\n')
    code_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^(def |class |import |from |@|#\s*!|if __name__)', stripped):
            code_start = i
            break
    if code_start is None:
        return text, ''
    prose = '\n'.join(lines[:code_start]).strip()
    code = '\n'.join(lines[code_start:]).strip()
    return prose, code


def is_python_only(code: str) -> bool:
    if not code or len(code.strip()) < 10:
        return False
    return validate_code(code)


def compute_diff(original: str, modified: str) -> tuple[str | None, str]:
    """compute old_string/new_string for an Edit.
    returns (old_string, new_string).
    old_string=None means full file replacement (>80% changed)."""
    orig_lines = original.split('\n')
    mod_lines = modified.split('\n')
    matcher = difflib.SequenceMatcher(None, orig_lines, mod_lines)
    total = max(len(orig_lines), 1)
    unchanged = sum(size for tag, _, _, _, size in matcher.get_opcodes() if tag == 'equal')
    changed_ratio = 1 - (unchanged / total)
    if changed_ratio > 0.8:
        return None, modified
    # find the contiguous changed region
    first_change = None
    last_change = None
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            if first_change is None:
                first_change = (i1, j1)
            last_change = (i2, j2)
    if first_change is None:
        return None, modified  # identical — shouldn't happen but handle gracefully
    # expand context by 2 lines for uniqueness
    ctx = 2
    oi_start = max(0, first_change[0] - ctx)
    oi_end = min(len(orig_lines), last_change[0] + ctx)
    mi_start = max(0, first_change[1] - ctx)
    mi_end = min(len(mod_lines), last_change[1] + ctx)
    old_string = '\n'.join(orig_lines[oi_start:oi_end])
    new_string = '\n'.join(mod_lines[mi_start:mi_end])
    # verify uniqueness of old_string in original
    if original.count(old_string) != 1:
        # expand context further
        for extra in range(1, 6):
            oi_start2 = max(0, first_change[0] - ctx - extra)
            oi_end2 = min(len(orig_lines), last_change[0] + ctx + extra)
            mi_start2 = max(0, first_change[1] - ctx - extra)
            mi_end2 = min(len(mod_lines), last_change[1] + ctx + extra)
            old_string = '\n'.join(orig_lines[oi_start2:oi_end2])
            new_string = '\n'.join(mod_lines[mi_start2:mi_end2])
            if original.count(old_string) == 1:
                break
        else:
            return None, modified  # can't make unique, fall back to full replacement
    return old_string, new_string


def make_read_edit_rollout(user_content: str, original_code: str, modified_code: str,
                           before_text: str, after_text: str, idx: int) -> dict | None:
    original_code = original_code.strip()
    modified_code = modified_code.strip()
    if not original_code or len(original_code) < 20:
        return None
    if not modified_code or len(modified_code) < 20:
        return None
    orig_lines = original_code.split('\n')
    mod_lines = modified_code.split('\n')
    if len(orig_lines) > 200 or len(mod_lines) > 200:
        return None

    filename = infer_filename(original_code, idx)
    formatted_original = format_line_numbers(original_code)
    formatted_modified = format_line_numbers(modified_code)
    old_string, new_string = compute_diff(original_code, modified_code)

    read_preamble = before_text if before_text else ""
    edit_args = {"file_path": filename, "new_string": new_string}
    if old_string is not None:
        edit_args["old_string"] = old_string

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": read_preamble,
         "tool_call": {"name": "Read", "args": {"file_path": filename}}},
        {"role": "tool_result", "content": formatted_original},
        {"role": "assistant", "content": "",
         "tool_call": {"name": "Edit", "args": edit_args}},
        {"role": "tool_result", "content": formatted_modified},
    ]
    if after_text:
        messages.append({"role": "assistant", "content": after_text})
    else:
        messages.append({"role": "assistant", "content": ""})
    return {"messages": messages}


def transform_evol(max_rows: int = None) -> list[dict]:
    ds = load_dataset('theblackcat102/evol-codealpaca-v1', split='train')
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    rollouts = []
    skipped = {"no_code_in": 0, "no_code_out": 0, "not_python_in": 0, "not_python_out": 0,
               "identical": 0, "too_short": 0, "build_fail": 0}
    for idx, row in enumerate(ds):
        instruction = row.get('instruction', '') or ''
        output = row.get('output', '') or ''
        if not instruction or not output:
            continue

        prose_in, code_in = extract_code_from_field(instruction)
        _, code_out = extract_code_from_field(output, allow_bare=True)

        if not code_in:
            skipped["no_code_in"] += 1
            continue
        if not code_out:
            skipped["no_code_out"] += 1
            continue
        if not is_python_only(code_in):
            skipped["not_python_in"] += 1
            continue
        if not is_python_only(code_out):
            skipped["not_python_out"] += 1
            continue
        if code_in.strip() == code_out.strip():
            skipped["identical"] += 1
            continue

        # user content is the prose portion of the instruction (without the code itself)
        user_content = prose_in if prose_in else instruction
        # extract any prose from the output as the after_text
        prose_out, _ = extract_code_from_field(output, allow_bare=True)

        rollout = make_read_edit_rollout(
            user_content=user_content,
            original_code=code_in,
            modified_code=code_out,
            before_text="",
            after_text="",
            idx=idx,
        )
        if not rollout:
            skipped["build_fail"] += 1
            continue
        rollout['_source'] = 'evol-codealpaca-v1'
        rollout['_source_idx'] = idx
        rollouts.append(rollout)

        if idx % 5000 == 0:
            print(f"  evol: processed {idx}, kept {len(rollouts)}, skipped {skipped}")
    print(f"  evol: final — kept {len(rollouts)}, skipped {skipped}")
    return rollouts


import warnings as _warnings

PROMPT_LEAK_PATTERNS = [
    re.compile(r'#\s*Revised Prompt', re.IGNORECASE),
    re.compile(r'#\s*Rewritten Prompt', re.IGNORECASE),
    re.compile(r'Refactor the (?:programming )?(?:test )?question', re.IGNORECASE),
    re.compile(r'Increase the difficulty', re.IGNORECASE),
    re.compile(r'#\s*(?:New|Modified|Updated) Prompt', re.IGNORECASE),
    re.compile(r'Make the (?:programming )?(?:test )?question', re.IGNORECASE),
]

def has_prompt_leak(text: str) -> bool:
    return any(p.search(text) for p in PROMPT_LEAK_PATTERNS)


def discover_evol(max_rows=None, verbose=False, seed=42):
    ds = load_dataset('theblackcat102/evol-codealpaca-v1', split='train').shuffle(seed)
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    categories = {"no_code_either": 0, "code_in_only": 0, "code_out_only": 0, "code_both": 0}
    both_stats = {"identical": 0, "full_replace": 0, "partial_edit": 0, "not_python": 0}
    prompt_leak_count = 0
    samples = {"no_code_either": [], "full_replace": [], "partial_edit": [], "code_out_only": [], "prompt_leak": [], "identical": [], "not_python": []}
    for idx, row in enumerate(ds):
        instruction = row.get('instruction', '') or ''
        output = row.get('output', '') or ''
        if not instruction or not output:
            continue
        if has_prompt_leak(instruction):
            prompt_leak_count += 1
            if len(samples["prompt_leak"]) < 3:
                samples["prompt_leak"].append(idx)
        _, code_in = extract_code_from_field(instruction)
        _, code_out = extract_code_from_field(output, allow_bare=True)
        has_in, has_out = bool(code_in), bool(code_out)
        if not has_in and not has_out:
            categories["no_code_either"] += 1
            if len(samples["no_code_either"]) < 3:
                samples["no_code_either"].append(idx)
        elif has_in and not has_out:
            categories["code_in_only"] += 1
        elif not has_in and has_out:
            categories["code_out_only"] += 1
            if len(samples["code_out_only"]) < 3:
                samples["code_out_only"].append(idx)
        else:
            categories["code_both"] += 1
            if not is_python_only(code_in) or not is_python_only(code_out):
                both_stats["not_python"] += 1
                if len(samples["not_python"]) < 3:
                    samples["not_python"].append(idx)
                continue
            if code_in.strip() == code_out.strip():
                both_stats["identical"] += 1
                if len(samples["identical"]) < 3:
                    samples["identical"].append(idx)
                continue
            old_string, _ = compute_diff(code_in, code_out)
            if old_string is None:
                both_stats["full_replace"] += 1
                if len(samples["full_replace"]) < 3:
                    samples["full_replace"].append(idx)
            else:
                both_stats["partial_edit"] += 1
                if len(samples["partial_edit"]) < 3:
                    _, new_string = compute_diff(code_in, code_out)
                    samples["partial_edit"].append((idx, old_string, new_string))

    print(f"dataset: {len(ds)} rows")
    print(f"\nprompt_leak: {prompt_leak_count}")
    print(f"\ncategories:")
    for k, v in categories.items():
        print(f"  {k}: {v}")
    print(f"\ncode_both breakdown (python only):")
    for k, v in both_stats.items():
        print(f"  {k}: {v}")

    if not verbose:
        return

    for category, entries in samples.items():
        if not entries:
            continue
        print(f"\n{'='*80}")
        print(f"sample {category} examples:")
        for entry in entries:
            if category == "partial_edit":
                sid, old_str, new_str = entry
            else:
                sid = entry
            row = ds[sid]
            print(f"\n--- idx {sid} ---")
            print(f"  instruction: {row['instruction']}")
            print(f"  output: {row['output']}")
            if category == "partial_edit":
                _, code_in = extract_code_from_field(row['instruction'])
                _, code_out = extract_code_from_field(row['output'], allow_bare=True)
                diff = difflib.unified_diff(
                    code_in.splitlines(), code_out.splitlines(),
                    fromfile='original', tofile='modified', lineterm='')
                print(f"\n  diff:")
                for line in diff:
                    if line.startswith('+'):
                        print(f"  \033[32m{line}\033[0m")
                    elif line.startswith('-'):
                        print(f"  \033[31m{line}\033[0m")
                    elif line.startswith('@@'):
                        print(f"  \033[36m{line}\033[0m")
                    else:
                        print(f"  {line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='rollouts/tier1b_evol.jsonl')
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--discover', action='store_true', help='explore dataset structure without transforming')
    parser.add_argument('--verbose', action='store_true', help='print samples in each category (use with --discover)')
    parser.add_argument('--seed', type=int, default=42, help='shuffle seed for discover')
    parser.add_argument('--dry-run', action='store_true', help='run transform + validation without writing to disk')
    parser.add_argument('--restyle', action='store_true', help='run restyle pass via local vllm')
    parser.add_argument('--restyle-model', type=str, default='Qwen/Qwen3-30B-A3B-Instruct-2507')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-retries', type=int, default=3)
    args = parser.parse_args()

    if args.discover:
        discover_evol(args.max_rows, verbose=args.verbose, seed=args.seed)
        return

    print("transforming evol-codealpaca-v1...")
    rollouts = transform_evol(args.max_rows)
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
            if tok_fail == 0:
                print(f"  first tokenizer failure: {e}\n  rollout: {json.dumps(r['messages'][:2], indent=2, ensure_ascii=False)[:500]}")
            tok_fail += 1
            continue
        valid_rollouts.append(r)
        valid += 1

    print(f"\nfiltering: {valid} valid, {invalid} invalid, {too_long} too long, {tok_fail} tokenizer failures")

    if args.dry_run:
        if valid_rollouts:
            sample = valid_rollouts[0]
            ds = load_dataset('theblackcat102/evol-codealpaca-v1', split='train')
            original_row = ds[sample['_source_idx']]
            print(f"\noriginal sample (idx {sample['_source_idx']}):")
            print(f"  instruction: {original_row['instruction'][:500]}")
            print(f"  output: {original_row['output'][:500]}")
            print(f"\ntransformed rollout:")
            print(json.dumps(sample, indent=2, ensure_ascii=False))
        print(f"\ndry run complete — {len(valid_rollouts)} rollouts would be written")
        return

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
