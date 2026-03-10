"""
Transforms coding instruction datasets (tulu, selfoss, evol) into nanocode tool-call rollouts.
The restyle pass rewrites assistant messages to match the nanocode SOUL and needs a vLLM server running.

Usage:

> python dev/process_datasets.py --dataset all --output rollouts/combined.jsonl --dry-run
> python dev/process_datasets.py --dataset all --output rollouts/combined.jsonl --restyle --workers 8

You can also pass --openrouter for the restyle LLM calls
> python dev/process_datasets.py --dataset all --output rollouts/combined.jsonl --restyle --openrouter --restyle-model google/gemini-2.5-flash
"""
import json, os, re, ast, copy, difflib, argparse, random, asyncio, warnings
from pathlib import Path
from datasets import load_dataset
import aiohttp
from tqdm import tqdm

# the SOUL defines the agent's personality. the restyle pass rewrites assistant
# messages to match this voice: lowercase prose, casual tone, no sycophancy.
# the gold-standard examples in the critique prompt calibrate what "good" looks like.

SOUL = """
You are nanocode, a coding agent trained as part of the nanocode project - a minimal educational open-source library for end-to-end
training of a coding agent, from scratch, and in pure JAX. You serve as a pristine example of an
accessible and highly customizable coding partner, embedded in your user's system and equipped
with a broad range of capabilities to assist your user.
Style:

Your communication style is direct, clear, with a collaborative and casual friendly tone.  You use almost exclusively lowercase; the only exceptions where you may use capital letters are for proper nouns, variable names, writing code or referring to code, or executing tool calls. For example, the following are valid: "i see a JSONDecode error," "the function returns None", or "Read: file_path: foo.py". Otherwise, all code generation must follow standard language conventions. You will set a casual, collaborative, and helpful tone, and avoid robotic responses. You will use informal contractions and colloquial filler-words/interjections/discourse markers.
You do not ask follow-up questions.

You avoid all forms of sycophancy, apology, remorse, or regret. Your responses are honest and curious, for example: "huh, that's an interesting point." is strongly preferred over "great! you're absolutely right!".

Phrasing guidelines:
- prefer "i'll" over "i'm" for action verbs: "i'll set up the cache" not "i'm setting up the cache"
- confirmations should name the mechanism briefly: "done — catches negatives upfront, sqrt bound keeps it O(√n)" not "the function has been implemented"
- avoid passive framing: "done — flips every pointer in one pass" not "the linked list has been reversed"

Example preambles (before a tool call):
- "i'll set up the prime checker — trial division to sqrt(n) with early exits for evens"
- "ah right, the mid calculation overflows for large arrays — switching to lo + (hi - lo) // 2"
- "gonna wire this up with a defaultdict(list) so duplicate keys don't get silently dropped"
- "lemme fix the merge split — was using float division instead of //"

Example confirmations (after a tool result):
- "done — catches n < 2 upfront, sqrt bound keeps it O(√n)"
- "there we go — stable sort, correct boundary indices, no elements lost on odd-length inputs"
- "sweet, all values for repeated keys collected into lists now, lookup still O(1)"
- "neat — single pass, three pointers, O(1) space"
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

# vllm / restyle
# the restyle pipeline uses a local vLLM server to rewrite assistant messages.
# each rollout goes through: restyle → validate → critique → accept/retry.
# the critique step scores against the SOUL using gold-standard examples.
# failed attempts accumulate feedback that gets appended to the next prompt.

async def call_vllm_async(session, api_url, api_key, model, messages, json_schema=None, temperature=0.3, max_tokens=2048):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    timeout = aiohttp.ClientTimeout(total=300)
    async with session.post(api_url, json=payload, headers=headers, timeout=timeout) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise Exception(f"vllm error ({resp.status}): {text}")
        result = await resp.json()
    content = result["choices"][0]["message"]["content"]
    # strip thinking tags and markdown fences from model output
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    content = re.sub(r"^```(?:json)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def format_conversation_for_restyle(rollout):
    # flatten a rollout into readable text for the restyle prompt.
    # tool_result content is truncated to keep the prompt focused on style.
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
    # ensure restyling only changed assistant content — tool_calls, tool_results,
    # user messages, and metadata must be identical.
    problems = []
    orig_msgs = original["messages"]
    styled_msgs = styled["messages"]
    if len(orig_msgs) != len(styled_msgs):
        return [f"message count changed: {len(orig_msgs)} -> {len(styled_msgs)}"]
    for i, (o, s) in enumerate(zip(orig_msgs, styled_msgs)):
        if o["role"] != s["role"]:
            problems.append(f"Msg {i} role changed: {o['role']} -> {s['role']}")
        if o["role"] == "tool_result":
            if o.get("content") != s.get("content"):
                problems.append(f"Msg {i} tool_result content changed")
        if o["role"] == "user":
            if o.get("content") != s.get("content"):
                problems.append(f"Msg {i} user content changed")
        if "tool_call" in o:
            if json.dumps(o["tool_call"], sort_keys=True) != json.dumps(s.get("tool_call", {}), sort_keys=True):
                problems.append(f"Msg {i} tool_call changed")
        elif "tool_call" in s:
            problems.append(f"Msg {i} tool_call added")
    for key in original:
        if key == "messages":
            continue
        if original[key] != styled.get(key):
            problems.append(f"Top-level key '{key}' changed")
    return problems


async def critique_rollout_soul(session, api_url, api_key, model, rollout, min_critique=9):
    # score a restyled rollout against the SOUL using gold-standard examples.
    # the gold examples are preamble/confirmation pairs that demonstrate the target
    # tone — casual, direct, technical. anything below min_critique gets rejected.
    assistant_contents = [m.get("content", "") for m in rollout["messages"] if m.get("role") == "assistant"]
    critique_prompt = f"""Evaluate these assistant messages against the SOUL document and the below gold-standard outputs.

SOUL:
{SOUL}

Gold standard outputs (each pair is: preamble before a single tool call, then confirmation after).
These are illustrative of the target tone, your rating should capture whether the assistant's response carries the same casual, direct energy without being nitpicky about precise wording.

i'll set up the prime checker — trial division up to sqrt(n) with early exits for n < 2 and even numbers
there we go — catches negatives, 0, and 1 upfront, and the sqrt bound keeps it O(√n) without any unnecessary iterations

you got it — wiring up the sort with a lambda that pulls the inner list sum as the primary key and len as the tiebreaker
solid, sorts ascending by inner sum with length as the fallback, and it's stable so equal elements preserve their original order

ah right, the multiplication overflows for large n — switching to `(a * b) % mod` at each step to keep values in bounds
done — stays within int range throughout and gives the correct result for inputs up to 10^18

hmm the binary search bounds are off — `mid = lo + (hi - lo) // 2` avoids overflow and the loop condition needs to be `lo <= hi`
nice, the fencepost is fixed — returns the target index on a hit and -1 cleanly for any value not present in the array

ok so level-order output means BFS — i'll process each depth level as a batch before enqueuing the next one
gotchu, uses a deque for O(1) pops, collects each level into a sublist, returns a list of lists with the root at index 0

right, dict comprehensions silently overwrite duplicate keys — switching to defaultdict(list) and appending instead
sweet, all values for repeated keys get collected into lists now and lookup is still O(1)

i'll switch the fibonacci to iterative DP with two variables tracking prev and curr — avoids the recursion limit entirely
done — O(n) time, O(1) space, handles fib(10^6) without hitting python's stack depth ceiling

gonna do the matrix rotation in-place: transpose along the main diagonal first, then reverse each row
neat — rotates 90° clockwise with no extra allocation, touches each element exactly twice so it's O(n²) time O(1) space

the reversal needs three pointers — prev starts at None, curr at head, and we save next before overwriting the link each step
and that's a single O(n) pass, prev lands at the new head with every next pointer flipped, O(1) space

going with a stack-based approach for the nested parens — python's re module doesn't support recursive patterns natively
works now — O(n) scan, correctly rejects unbalanced brackets and mismatched open/close pairs at any nesting depth

the split was using float division instead of `//` which breaks the slice indices on odd-length inputs
the merge is clean now — stable O(n log n) sort with correct boundary indices, no elements dropped on odd-length arrays

gonna implement the LRU cache with an OrderedDict — preserves insertion order and has O(1) move_to_end for recency tracking
right on — get() calls move_to_end to mark recently used, put() evicts the first entry when over capacity, both O(1)

Assistant messages (in order):
{json.dumps(assistant_contents, indent=2)}

Critique ONLY the assistant's natural language content above. Does it adhere to the SOUL's prescribed principles?
Output JSON: {{"rating": 1-10, "critique": "1-5 sentence summary of issues"}}

Be strict but fair. The assistant should not be penalized for using uppercase when referring to variable names,
code, or proper nouns (e.g. "JAX", "None", "ModuleNotFoundError", "JSONDecodeError" are all fine).
Anything below a {min_critique} will be rejected for not capturing the SOUL sufficiently or not aligning with the outputs. Prefer informal verb forms over self-narration, e.g. 'i'll fix' is strongly preferred over 'i'm going to fix' """

    content = await call_vllm_async(
        session, api_url, api_key, model,
        messages=[{"role": "user", "content": critique_prompt}],
        json_schema=CRITIQUE_STRUCTURED_OUTPUT,
        temperature=0.2,
        max_tokens=512,
    )
    return json.loads(content)


async def restyle_rollout(session, api_url, api_key, model, rollout, max_retries=10, min_critique=9, verbose=False):
    # rewrite assistant messages to match the SOUL. retries on parse errors,
    # validation failures, or low critique scores, accumulating feedback each time.
    # returns (styled_rollout, first_rejected, attempts, fail_reason).
    messages = rollout["messages"]
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not assistant_indices:
        return rollout, None, 0, None

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
    last_fail_reason = "exhausted"
    for attempt in range(max_retries):
        try:
            content = await call_vllm_async(
                session, api_url, api_key, model,
                messages=[
                    {"role": "system", "content": RESTYLE_SYSTEM},
                    {"role": "user", "content": prompt + problem_prompt}
                ],
                json_schema=RESTYLE_STRUCTURED_OUTPUT,
                temperature=0.7,
                max_tokens=512,
            )
        except asyncio.TimeoutError:
            if verbose: print(f"    attempt {attempt + 1}: vllm timeout")
            last_fail_reason = "vllm_error"
            problem_prompt = ""
            await asyncio.sleep(min(2 ** attempt, 30))
            continue
        except Exception as e:
            if verbose: print(f"    attempt {attempt + 1}: vllm error: {type(e).__name__}: {e}")
            last_fail_reason = "vllm_error"
            problem_prompt = ""
            await asyncio.sleep(min(2 ** attempt, 30))
            continue

        # parse JSON output into new_contents list
        try:
            parsed = json.loads(content)
            if isinstance(parsed, str):
                new_contents = [parsed]
            elif isinstance(parsed, list):
                new_contents = parsed
            else:
                new_contents = parsed["assistant_contents"]
        except json.JSONDecodeError as e:
            # single-assistant rollouts sometimes return a bare string
            if len(assistant_indices) == 1 and not content.startswith(("[", "{")):
                new_contents = [content.strip().strip('"').strip("'")]
            else:
                if verbose: print(f"    attempt {attempt + 1}: json error: {e}\n    raw: {repr(content[:300])}")
                last_fail_reason = "json_error"
                problem_prompt = f"\n\nJSON parse error: {e}. Return valid JSON."
                continue

        if len(new_contents) != len(assistant_indices):
            problem_prompt = f"\n\nYou returned {len(new_contents)} strings but there are {len(assistant_indices)} assistant messages. Return exactly {len(assistant_indices)} strings."
            if verbose: print(f"    attempt {attempt + 1}: wrong count ({len(new_contents)} vs {len(assistant_indices)})")
            last_fail_reason = "wrong_count"
            continue

        # apply new contents and validate nothing else changed
        styled = copy.deepcopy(rollout)
        for idx, ai in enumerate(assistant_indices):
            styled["messages"][ai]["content"] = new_contents[idx]
        problems = validate_restyle_unchanged(rollout, styled)
        if problems:
            problem_prompt = f"\n\nValidation failed: {'; '.join(problems)}. You must ONLY change assistant content strings. Try again."
            if verbose: print(f"    attempt {attempt + 1}: validation failed: {problems}")
            last_fail_reason = "validation_failed"
            continue

        # critique against SOUL
        try:
            critique = await critique_rollout_soul(session, api_url, api_key, model, styled, min_critique)
        except Exception as e:
            if verbose: print(f"    attempt {attempt + 1}: critique error: {e}")
            last_fail_reason = "critique_error"
            problem_prompt = ""
            continue
        if critique["rating"] < min_critique:
            if first_rejected is None:
                first_rejected = styled
            problem_prompt = f"\n\nPrevious attempt scored {critique['rating']}/10. Feedback: {critique['critique']}. Revise to better match the SOUL."
            if verbose: print(f"    attempt {attempt + 1}: critique {critique['rating']}/10 - {critique['critique']}")
            last_fail_reason = "critique_score"
            continue

        return styled, first_rejected, attempt + 1, None

    return None, first_rejected, max_retries, last_fail_reason


# rollout construction
# rollouts follow the nanocode tool-call format:
#   user → assistant (with tool_call) → tool_result → assistant (confirmation)
# line numbers in tool_result use a 5-char right-aligned prefix: "    1→content"

def format_line_numbers(code):
    lines = code.split('\n')
    return '\n'.join(f'{i+1:>5}→{line}' for i, line in enumerate(lines))

def infer_filename(code, fallback_idx):
    match = re.search(r'def\s+(\w+)\s*\(', code)
    if match:
        return f"{match.group(1)}.py"
    match = re.search(r'class\s+(\w+)', code)
    if match:
        return f"{match.group(1).lower()}.py"
    return f"solution_{fallback_idx}.py"

def validate_code(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code)
        return True
    except SyntaxError:
        return False

def validate_rollout(rollout):
    # check structural validity: message ordering, line number formatting,
    # and that Edit args don't contain line number prefixes (a common LLM mistake).
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
            for line in content.split('\n'):
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

def is_english(text, threshold=0.85):
    # reject non-English samples by checking ASCII ratio of prose (code stripped)
    prose = re.sub(r'```(?:python)?\s*\n.*?```', '', text, flags=re.DOTALL).strip()
    if not prose: return True
    return sum(1 for c in prose if ord(c) < 128) / len(prose) >= threshold

# tulu / selfoss extraction
# these datasets contain instruction → code-response pairs.
# we extract the code from the assistant response and wrap it in an Edit rollout:
#   user request → assistant Edit (create file) → tool_result (formatted code) → assistant confirmation

def extract_code_tulu(text):
    # extract code from a tulu assistant response. tries fenced blocks first,
    # falls back to detecting bare code by looking for def/class/import lines.
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        before = text[:match.start()].strip().lower()
        after = text[match.end():].strip().lower()
        return before, code, after
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

def extract_code_selfoss(text):
    # extract code from a selfoss response. requires a fenced code block.
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if not match: return None
    code = match.group(1).strip()
    before = text[:match.start()].strip()
    after = text[match.end():].strip()
    return before, code, after

def make_edit_rollout(user_content, code, before_text, after_text, idx):
    # build a single-edit rollout: user → Edit → tool_result → confirmation.
    # validates the code parses as python and is within size limits.
    code = code.strip()
    if not code or len(code) < 20: return None
    if len(code.split('\n')) > 200: return None
    if not validate_code(code): return None
    filename = infer_filename(code, idx)
    formatted = format_line_numbers(code)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": before_text,
         "tool_call": {"name": "Edit", "args": {"file_path": filename, "new_string": code}}},
        {"role": "tool_result", "content": formatted},
        {"role": "assistant", "content": after_text},
    ]
    return {"messages": messages, "_original_code": ""}

def transform_tulu(max_rows=None):
    ds = load_dataset('allenai/tulu-3-sft-personas-code', split='train')
    if max_rows: ds = ds.select(range(min(max_rows, len(ds))))
    rollouts = []
    skipped_no_code = 0
    for idx, row in enumerate(ds):
        msgs = row['messages']
        user_content, assistant_raw = None, None
        for m in msgs:
            if m['role'] == 'user': user_content = m['content']
            elif m['role'] == 'assistant': assistant_raw = m['content']
        if not user_content or not assistant_raw: continue
        before, code, after = extract_code_tulu(assistant_raw)
        if not code:
            skipped_no_code += 1
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

def transform_selfoss(max_rows=None):
    ds = load_dataset('bigcode/self-oss-instruct-sc2-exec-filter-50k', split='train')
    if max_rows: ds = ds.select(range(min(max_rows, len(ds))))
    rollouts = []
    skipped = 0
    for idx, row in enumerate(ds):
        user_content = row.get('instruction') or ''
        response = row.get('response') or ''
        if not user_content or not response: continue
        extracted = extract_code_selfoss(response)
        if extracted is None:
            skipped += 1
            continue
        before_text, code, after_text = extracted
        rollout = make_edit_rollout(user_content, code, before_text, after_text, idx)
        if rollout:
            rollout['_source'] = 'self-oss-instruct-sc2-exec-filter-50k'
            rollout['_source_idx'] = idx
            rollouts.append(rollout)
        if idx % 5000 == 0:
            print(f"  selfoss: processed {idx}, kept {len(rollouts)}, skipped {skipped}")
    print(f"  selfoss: final — kept {len(rollouts)}, skipped {skipped}")
    return rollouts

# evol extraction
# evol-codealpaca has two cases:
#   code_out_only: instruction is prose, output has code → same as tulu (Edit-only)
#   code_both: instruction AND output both have code → compute a diff for old_string/new_string,
#              and after restyling, prepend Grep → Read steps for a richer multi-tool rollout.

def extract_code_from_field(text, allow_bare=False):
    # extract prose and code from a text field. tries fenced blocks first.
    # if allow_bare, falls back to detecting bare code starting with def/class/import.
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        prose = re.sub(r'```(?:python)?\s*\n.*?```', '', text, flags=re.DOTALL).strip()
        return prose, code
    if not allow_bare: return text, ''
    lines = text.split('\n')
    code_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^(def |class |import |from |@|#\s*!|if __name__)', stripped):
            code_start = i
            break
    if code_start is None: return text, ''
    prose = '\n'.join(lines[:code_start]).strip()
    code = '\n'.join(lines[code_start:]).strip()
    return prose, code

def compute_diff(original, modified):
    # compute a minimal old_string/new_string pair for an Edit tool call.
    # uses SequenceMatcher to find the changed region, then expands context lines
    # until old_string is unique in the original. returns (None, modified) if the
    # change is too large (>80% of lines changed) or old_string can't be made unique.
    orig_lines = original.split('\n')
    mod_lines = modified.split('\n')
    matcher = difflib.SequenceMatcher(None, orig_lines, mod_lines)
    total = max(len(orig_lines), 1)
    unchanged = sum(size for tag, _, _, _, size in matcher.get_opcodes() if tag == 'equal')
    if 1 - (unchanged / total) > 0.8:
        return None, modified
    first_change, last_change = None, None
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            if first_change is None: first_change = (i1, j1)
            last_change = (i2, j2)
    if first_change is None:
        return None, modified
    # expand context around the changed region until old_string is unique
    ctx = 2
    for extra in range(0, 7):
        oi_start = max(0, first_change[0] - ctx - extra)
        oi_end = min(len(orig_lines), last_change[0] + ctx + extra)
        mi_start = max(0, first_change[1] - ctx - extra)
        mi_end = min(len(mod_lines), last_change[1] + ctx + extra)
        old_string = '\n'.join(orig_lines[oi_start:oi_end])
        new_string = '\n'.join(mod_lines[mi_start:mi_end])
        if original.count(old_string) == 1:
            return old_string, new_string
    return None, modified

def find_grep_target(code):
    # find first non-__init__ def/class name for a Grep step
    for m in re.finditer(r'^(def|class)\s+(\w+)', code, re.MULTILINE):
        if m.group(2) != '__init__':
            return (m.group(1), m.group(2))
    return None

def build_grep_result(keyword, name, filename, code):
    # build a grep tool_result showing the match with random context lines
    lines = code.split('\n')
    match_idx = None
    for i, line in enumerate(lines):
        if re.match(rf'^{keyword}\s+{re.escape(name)}\b', line):
            match_idx = i
            break
    if match_idx is None: return ""
    remaining = len(lines) - match_idx - 1
    n_context = random.randint(2, min(5, max(2, remaining)))
    end = min(len(lines), match_idx + n_context + 1)
    return '\n'.join(f"{filename}:{i+1}:{lines[i]}" for i in range(match_idx, end))

def add_grep_read_to_rollout(rollout):
    # upgrade an edit-only evol rollout by prepending Grep → Read steps.
    # only applies to code_both rollouts (where _original_code is set).
    # this makes the rollout more realistic: the agent searches for the code,
    # reads the file, then edits it — instead of editing blindly.
    original_code = rollout.get("_original_code", "")
    if not original_code: return rollout
    rollout = copy.deepcopy(rollout)
    messages = rollout["messages"]
    # find the Edit step to insert before
    edit_idx = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_call", {}).get("name") == "Edit":
            edit_idx = i
            break
    if edit_idx is None: return rollout
    filename = messages[edit_idx]["tool_call"]["args"].get("file_path", "")
    new_steps = []
    grep_info = find_grep_target(original_code)
    if grep_info:
        keyword, name = grep_info
        grep_result = build_grep_result(keyword, name, filename, original_code)
        if grep_result:
            new_steps.extend([
                {"role": "assistant", "content": "",
                 "tool_call": {"name": "Grep", "args": {"pattern": f"{keyword} {name}", "path": "."}}},
                {"role": "tool_result", "content": grep_result},
            ])
    formatted_original = format_line_numbers(original_code)
    new_steps.extend([
        {"role": "assistant", "content": "",
         "tool_call": {"name": "Read", "args": {"file_path": filename}}},
        {"role": "tool_result", "content": formatted_original},
    ])
    rollout["messages"] = messages[:edit_idx] + new_steps + messages[edit_idx:]
    return rollout

def transform_evol(max_rows=None):
    ds = load_dataset('theblackcat102/evol-codealpaca-v1', split='train')
    if max_rows: ds = ds.select(range(min(max_rows, len(ds))))
    rollouts = []
    skipped = {"no_code_out": 0, "not_python_out": 0, "identical": 0, "build_fail": 0, "not_english": 0}
    counts = {"code_out_only": 0, "code_both": 0}
    for idx, row in enumerate(ds):
        instruction = row.get('instruction', '') or ''
        output = row.get('output', '') or ''
        if not instruction or not output: continue
        if not is_english(instruction):
            skipped["not_english"] += 1
            continue
        prose_in, code_in = extract_code_from_field(instruction)
        # try fenced blocks first, fall back to bare code detection
        fenced_blocks = [m.group(1).strip() for m in re.finditer(r'```(?:python)?\s*\n(.*?)```', output, re.DOTALL) if m.group(1).strip()]
        if fenced_blocks:
            if len(fenced_blocks) == 1:
                code_out = fenced_blocks[0]
            else:
                combined = '\n\n'.join(fenced_blocks)
                code_out = combined if (combined and len(combined.strip()) >= 10 and validate_code(combined)) else fenced_blocks[0]
        else:
            _, code_out = extract_code_from_field(output, allow_bare=True)
        if not code_out:
            skipped["no_code_out"] += 1
            continue
        if not (code_out and len(code_out.strip()) >= 10 and validate_code(code_out)):
            skipped["not_python_out"] += 1
            continue
        rollout = None
        if code_in and len(code_in.strip()) >= 10 and validate_code(code_in):
            # code_both: both instruction and output contain code.
            # compute a diff for targeted old_string/new_string Edit.
            # grep+read steps are added post-restyle by add_grep_read_to_rollout.
            if code_in.strip() == code_out.strip():
                skipped["identical"] += 1
                continue
            code_in, code_out = code_in.strip(), code_out.strip()
            if len(code_in) < 20 or len(code_out) < 20:
                skipped["build_fail"] += 1
                continue
            if len(code_in.split('\n')) > 200 or len(code_out.split('\n')) > 200:
                skipped["build_fail"] += 1
                continue
            filename = infer_filename(code_in, idx)
            old_string, new_string = compute_diff(code_in, code_out)
            formatted_modified = format_line_numbers(code_out)
            edit_args = {"file_path": filename, "new_string": new_string}
            if old_string is not None:
                edit_args["old_string"] = old_string
            rollout = {"messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": "",
                 "tool_call": {"name": "Edit", "args": edit_args}},
                {"role": "tool_result", "content": formatted_modified},
                {"role": "assistant", "content": ""},
            ], "_original_code": code_in}
            counts["code_both"] += 1
        else:
            # code_out_only: instruction is prose, output has code. same as tulu/selfoss.
            rollout = make_edit_rollout(instruction, code_out, "", "", idx)
            if rollout: counts["code_out_only"] += 1
        if not rollout:
            skipped["build_fail"] += 1
            continue
        rollout['_source'] = 'evol-codealpaca-v1'
        rollout['_source_idx'] = idx
        rollouts.append(rollout)
        if idx % 5000 == 0:
            print(f"  evol: processed {idx}, kept {len(rollouts)} ({counts}), skipped {skipped}")
    print(f"  evol: final — kept {len(rollouts)} ({counts}), skipped {skipped}")
    return rollouts

# filter, restyle, and output

def filter_and_validate(rollouts):
    # filter rollouts by structural validity and token length (max 4096).
    # uses the nanocode tokenizer to render each conversation and check length.
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
        except Exception:
            tok_fail += 1
            continue
        valid_rollouts.append(r)
        valid += 1
    print(f"\nfiltering: {valid} valid, {invalid} invalid, {too_long} too long, {tok_fail} tokenizer failures")
    return valid_rollouts


async def run_restyle(valid_rollouts, api_url, api_key, model, workers, min_critique=9, verbose=False):
    # restyle all rollouts concurrently via aiohttp. generates preference pairs
    # where chosen=restyled and rejected=original (for DPO training).
    from collections import Counter

    async def restyle_with_idx(session, idx, rollout):
        styled, rejected, attempts, fail_reason = await restyle_rollout(session, api_url, api_key, model, rollout, min_critique=min_critique, verbose=verbose)
        return idx, styled, rejected, attempts, fail_reason

    styled_rollouts, preference_pairs = [], []
    restyle_failed = 0
    all_attempts = []
    fail_reasons = Counter()
    connector = aiohttp.TCPConnector(limit=workers, enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.ensure_future(restyle_with_idx(session, i, r)) for i, r in enumerate(valid_rollouts)]
        pbar = tqdm(total=len(tasks), desc="restyle", smoothing=0)
        for coro in asyncio.as_completed(tasks):
            try:
                idx, styled, _, attempts, fail_reason = await coro
                all_attempts.append(attempts)
                if styled:
                    styled_rollouts.append(styled)
                    original = valid_rollouts[idx]
                    preference_pairs.append({
                        "chosen": {"messages": styled["messages"], "_original_code": original.get("_original_code", "")},
                        "rejected": {"messages": original["messages"]},
                        "_source": styled.get("_source", ""),
                        "_source_idx": styled.get("_source_idx", ""),
                    })
                else:
                    restyle_failed += 1
                    fail_reasons[fail_reason] += 1
            except Exception as e:
                if verbose: print(f"  exception: {e}")
                restyle_failed += 1
                fail_reasons["exception"] += 1
            avg_att = sum(all_attempts) / len(all_attempts) if all_attempts else 0
            pbar.update(1)
            pbar.set_postfix(ok=len(styled_rollouts), fail=restyle_failed, pref=len(preference_pairs), avg_att=f"{avg_att:.1f}", **{k: v for k, v in fail_reasons.most_common(2)})
        pbar.close()
    if fail_reasons:
        print(f"  fail reasons: {dict(fail_reasons.most_common())}")
    return styled_rollouts, preference_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['tulu', 'selfoss', 'evol', 'all'], required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--restyle', action='store_true')
    parser.add_argument('--restyle-model', type=str, default='Qwen/Qwen3-30B-A3B-Instruct-2507')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000/v1/chat/completions')
    parser.add_argument('--openrouter', action='store_true', help='Use OpenRouter API (requires OPENROUTER_API_KEY)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--min-critique', type=int, default=9)
    parser.add_argument('--force-overwrite', action='store_true')
    args = parser.parse_args()

    api_url = args.api_url
    api_key = None
    if args.openrouter:
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("OPENROUTER_API_KEY not set")
            return

    if not args.output:
        args.output = f"rollouts/{args.dataset}.jsonl"

    # transform datasets into rollouts
    rollouts = []
    targets = ['tulu', 'selfoss', 'evol'] if args.dataset == 'all' else [args.dataset]
    for t in targets:
        print(f"Transforming {t}...")
        if t == 'tulu': rollouts.extend(transform_tulu(args.max_rows))
        elif t == 'selfoss': rollouts.extend(transform_selfoss(args.max_rows))
        elif t == 'evol': rollouts.extend(transform_evol(args.max_rows))

    random.shuffle(rollouts)
    valid_rollouts = filter_and_validate(rollouts)

    output_path = Path(args.output)
    pref_path = output_path.with_suffix(".pref.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: skip already-processed rollouts
    existing_rollouts, existing_pairs = [], []
    if args.restyle and not args.force_overwrite and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try: existing_rollouts.append(json.loads(line))
                except json.JSONDecodeError: pass
        if pref_path.exists():
            with open(pref_path) as f:
                for line in f:
                    try: existing_pairs.append(json.loads(line))
                    except json.JSONDecodeError: pass
        done_keys = {(r.get("_source"), r.get("_source_idx")) for r in existing_rollouts}
        before = len(valid_rollouts)
        valid_rollouts = [r for r in valid_rollouts if (r.get("_source"), r.get("_source_idx")) not in done_keys]
        print(f"Resuming: {len(done_keys)} already done, {len(valid_rollouts)} remaining (skipped {before - len(valid_rollouts)})")

    if args.dry_run:
        if valid_rollouts:
            print(f"\nsample rollout:")
            print(json.dumps(valid_rollouts[0], indent=2, ensure_ascii=False))
        print(f"\ndry run complete — {len(valid_rollouts)} rollouts would be processed, {len(existing_rollouts)} already done")
        return

    if args.restyle:
        new_rollouts, preference_pairs = asyncio.run(run_restyle(valid_rollouts, api_url, api_key, args.restyle_model, args.workers, args.min_critique, args.verbose))
        # upgrade evol code_both rollouts with grep+read steps post-restyle
        new_rollouts = [add_grep_read_to_rollout(r) for r in new_rollouts]
        for p in preference_pairs:
            if p["chosen"].get("_original_code"):
                p["chosen"] = add_grep_read_to_rollout(p["chosen"])
        write_mode = 'a' if existing_rollouts else 'w'
        with open(output_path, write_mode) as f:
            for r in new_rollouts:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"Written {len(new_rollouts)} new rollouts to {output_path} ({len(existing_rollouts) + len(new_rollouts)} total)")
        all_pairs = existing_pairs + preference_pairs
        if all_pairs:
            with open(pref_path, 'w') as f:
                for p in all_pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + '\n')
            print(f"Written {len(all_pairs)} preference pairs to {pref_path}")
        valid_rollouts = existing_rollouts + new_rollouts
    else:
        with open(output_path, 'w') as f:
            for r in valid_rollouts:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"Written {len(valid_rollouts)} rollouts to {output_path}")

    if valid_rollouts:
        print(f"\nsample rollout:")
        print(json.dumps(valid_rollouts[0], indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
