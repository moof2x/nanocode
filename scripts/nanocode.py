"""Agentic CLI for nanocode.

To try this locally without training your own models, you can download checkpoints for either of the d12/d24 models I've trained:

- Quick to run on your Mac but not very smart
> hf download smohammadi/nanocode_d12 nanocode_d12.tar.gz --local-dir /tmp && tar xzf /tmp/nanocode_d12.tar.gz -C ~/.cache/nanocode

- Slow to run (requires GPU) but smarter
hf download smohammadi/nanocode_d24 nanocode_d24.tar.gz --local-dir /tmp && tar xzf /tmp/nanocode_d24.tar.gz -C ~/.cache/nanocode

Run:

MODEL_TAG=d24 python -m scripts.nanocode

Replace with d12 if you downloaded above, and add --attn-iompl
"""
import argparse
import re
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
from nanocode.checkpointing import load_checkpoint, load_model_config
from nanocode.common import get_model_dir
from nanocode.generation import generate
from nanocode.gpt import GPT
from nanocode.tokenizer import get_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='dpo')
parser.add_argument('--compute-dtype', type=str, default='bfloat16', choices=['bfloat16', 'float32'])
parser.add_argument('--attn-impl', type=str, default="eager", help="Attention backend: splash (TPU) or eager")
parser.add_argument('--max-tokens', type=int, default=512)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

checkpoint = args.checkpoint
compute_dtype = jnp.bfloat16 if args.compute_dtype == 'bfloat16' else jnp.float32
max_tokens = args.max_tokens
seed = args.seed
temperature = args.temperature
verbose = args.verbose

tokenizer = get_tokenizer()
model_dir = get_model_dir()
checkpoint_dir = model_dir / f"{checkpoint}_checkpoints"
model_cfg = load_model_config(checkpoint_dir / "model.zarr")
rng = jax.random.key(seed)
model = GPT.init(model_cfg, rng, attn_impl=args.attn_impl)
model = load_checkpoint(checkpoint_dir / "model.zarr", model)

project_root = Path.cwd()

# ANSI color codes
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"

# token_name -> token_id
SPECIAL_TOKENS = {name: tokenizer.encode_special(name) for name in [
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|tool_call_start|>", "<|tool_call_end|>",
    "<|tool_result_start|>", "<|tool_result_end|>",
]}

SYSTEM = """you are nanocode, a coding agent.
you communicate in lowercase and think out loud before acting.

your tools enable you to interact with the user's UNIX system and modify and create new files:
Read:  <|tool_call_start|>Read<|tool_arg|>file_path<|tool_val|>...<|tool_arg|>offset<|tool_val|>...<|tool_arg|>limit<|tool_val|>...<|tool_call_end|>
Edit:  <|tool_call_start|>Edit<|tool_arg|>file_path<|tool_val|>...<|tool_arg|>old_string<|tool_val|>...<|tool_arg|>new_string<|tool_val|>...<|tool_call_end|>
Grep:  <|tool_call_start|>Grep<|tool_arg|>pattern<|tool_val|>...<|tool_arg|>path<|tool_val|>...<|tool_call_end|>
Bash:  <|tool_call_start|>Bash<|tool_arg|>command<|tool_val|>...<|tool_call_end|>

example:
<|tool_call_start|>Bash<|tool_arg|>command<|tool_val|>echo hello<|tool_call_end|>
"""

# matches tool_read/tool_edit output format
fmt_lines = lambda t: "\n".join(f"{i+1:>5}→{l}" for i, l in enumerate(t.split("\n")))

def tool_read(args):
    file_path = project_root / args.get("file_path", "")
    offset, limit = args.get("offset", 0), args.get("limit", 100)
    if not file_path.exists():
        return "error: file not found"
    if file_path.is_dir():
        return "error: is a directory"
    try:
        lines = file_path.read_text().split("\n")
        return fmt_lines("\n".join(lines[offset:offset + limit]))
    except UnicodeDecodeError:
        return "error: binary file"

def tool_edit(args):
    file_path = project_root / args.get("file_path", "")
    old_string, new_string = args.get("old_string"), args.get("new_string", "")
    # model sometimes copies the "    1→" line-number prefix from tool_results into edit args
    strip_linenos = lambda s: re.sub(r"^ *\d+→", "", s, flags=re.MULTILINE)
    if old_string is not None:
        old_string = strip_linenos(old_string)
    new_string = strip_linenos(new_string)
    if old_string is None: # no old_string means create new file
        if file_path.exists():
            return "error: file already exists, use old_string to edit"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new_string)
        return str(file_path) + "\n" + fmt_lines(new_string)
    if not file_path.exists():
        return "error: file not found"
    content = file_path.read_text()
    if old_string not in content:
        return "error: old_string not found"
    new_content = content.replace(old_string, new_string, 1)
    file_path.write_text(new_content)
    return str(file_path) + "\n" + fmt_lines(new_content)

def tool_grep(args):
    pattern, path = args.get("pattern", ""), args.get("path", ".")
    target = project_root / path
    if not target.exists():
        return "error: path not found"
    cmd = ["grep", "-rn", "--include=*.py", "--include=*.md", "--include=*.json"]
    if args.get("-B"):
        cmd.extend(["-B", str(args["-B"])])
    if args.get("-A"):
        cmd.extend(["-A", str(args["-A"])])
    cmd.extend([pattern, str(target)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or "no matches found"
    except subprocess.TimeoutExpired:
        return "error: timed out"

def tool_bash(args):
    command = args.get("command", "")
    if not command:
        return "error: no command"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, cwd=project_root)
        return (result.stdout + result.stderr).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "error: timed out"

TOOLS = {"Read": tool_read, "Edit": tool_edit, "Grep": tool_grep, "Bash": tool_bash}

# parse "ToolName<|tool_arg|>key<|tool_val|>value..." into (name, args)
def parse_tool_call(text):
    parts = text.split("<|tool_arg|>")
    tool_name = parts[0].strip()
    args = {}
    for part in parts[1:]:
        if "<|tool_val|>" in part:
            k, v = part.split("<|tool_val|>", 1)
            v = v.strip()
            args[k.strip()] = int(v) if v.lstrip("-").isdigit() else v
    return tool_name, args

def run_agent(tokens, rng):
    max_seq_len = 2 * model_cfg.sequence_len # allow 2x context for tool results
    while True:
        tokens.append(SPECIAL_TOKENS["<|assistant_start|>"])
        # state for streaming token parsing
        in_tool_call = False
        tool_buffer = ""
        text_buffer = ""
        tool_called = False

        print(f"\n{CYAN}{BOLD}[assistant]{RESET} ", end="", flush=True)
        new_tokens = generate(tokens, model, max_tokens, temperature, compute_dtype,
                              pad_token_id=SPECIAL_TOKENS["<|assistant_end|>"],
                              rng=rng,
                              assistant_end_id=SPECIAL_TOKENS["<|assistant_end|>"])
        for token in new_tokens:
            token_id = int(token[0])
            tokens.append(token_id)
            decoded = tokenizer.decode([token_id])

            if verbose:
                print(decoded, end="", flush=True)

            # route tokens to text or tool buffer
            if token_id == SPECIAL_TOKENS["<|tool_call_start|>"]:
                if text_buffer and not verbose:
                    print(text_buffer, end="", flush=True)
                    text_buffer = ""
                in_tool_call = True
            elif token_id == SPECIAL_TOKENS["<|tool_call_end|>"]:
                tool_called = True
                break
            elif token_id == SPECIAL_TOKENS["<|assistant_start|>"]:
                break
            elif in_tool_call:
                tool_buffer += decoded
            else:
                text_buffer += decoded

            if len(tokens) > max_seq_len:
                print("(Context limit reached)")
                break

        if text_buffer and not verbose:
            print(text_buffer, end="", flush=True)
        print()

        if tool_called:
            tool_name, args = parse_tool_call(tool_buffer)
            print(f"\n{YELLOW}{BOLD}[{tool_name}]{RESET}")
            if tool_name == "Edit":
                print(f"  {DIM}file_path:{RESET} {args['file_path']}")
                if "old_string" in args:
                    print(f"  {RED}old_string:{RESET}\n{fmt_lines(args['old_string'])}")
                if "new_string" in args:
                    print(f"  {GREEN}new_string:{RESET}\n{fmt_lines(args['new_string'])}")
            else:
                print("\n".join(f"  {DIM}{k}:{RESET} {v}" for k, v in args.items()))
            if tool_name in ("Bash", "Edit"):
                confirm = input(f"{YELLOW}[y/n] > {RESET}").strip().lower()
                if confirm not in ("y", "yes", ""):
                    result = "rejected by user"
                    print(f"\n{DIM}--- tool result ---{RESET}\n{RED}{result}{RESET}\n{DIM}---{RESET}")
                    tokens.append(SPECIAL_TOKENS["<|tool_result_start|>"])
                    tokens.extend(tokenizer.encode(result))
                    tokens.append(SPECIAL_TOKENS["<|tool_result_end|>"])
                    break # break back to user input loop
            result = TOOLS.get(tool_name, lambda a: "error: unknown tool")(args)
            truncated = f"{result[:500]}\n{DIM}[truncated]{RESET}" if len(result) > 500 else result
            print(f"\n{DIM}--- tool result ---{RESET}\n{truncated}\n{DIM}---{RESET}")
            if verbose:
                print(f"<|tool_result_start|>{result}<|tool_result_end|>")
            # inject tool result into context
            tokens.append(SPECIAL_TOKENS["<|tool_result_start|>"])
            tokens.extend(tokenizer.encode(result))
            tokens.append(SPECIAL_TOKENS["<|tool_result_end|>"])
        else:
            break

    tokens.append(SPECIAL_TOKENS["<|user_start|>"]) 
    rng, _ = jax.random.split(rng) # advance rng state between turns
    return tokens, rng

HELP = f"""
{BOLD}Commands:{RESET}
  <message>        - Send message to the agent
  /clear           - Clear conversation and context
  /show            - Show token count
  /help            - Show this help
  /quit            - Exit
"""

print(f"\n{RED}{BOLD}{'=' * 60}")
print("WARNING: Nanocode executes real Bash commands on your")
print("system and is able to modify files. Carefully review")
print(f"tool calls before approving them.")
print(f"{'=' * 60}{RESET}")
print(f"\nModel: {checkpoint} | Sequence length: {model_cfg.sequence_len}")
print("Type '/help' for commands\n")

rng = jax.random.key(seed)
tokens = [tokenizer.get_bos_token_id(), SPECIAL_TOKENS["<|user_start|>"]]
tokens.extend(tokenizer.encode(SYSTEM))

while True:
    try:
        line = input(f"{BOLD}> {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye")
        break

    if not line:
        continue

    cmd, *rest = line.split(" ", 1)
    arg = rest[0] if rest else ""

    if cmd == "/quit":
        break
    elif cmd == "/help":
        print(HELP)
    elif cmd == "/clear":
        tokens = [tokenizer.get_bos_token_id(), SPECIAL_TOKENS["<|user_start|>"]]
        tokens.extend(tokenizer.encode(SYSTEM))
        rng = jax.random.key(seed)
        print("Cleared")
    elif cmd == "/show":
        print(f"Tokens in context: {len(tokens)}/{model_cfg.sequence_len * 2}")
    else:
        tokens.extend(tokenizer.encode(line))
        tokens.append(SPECIAL_TOKENS["<|user_end|>"])
        tokens, rng = run_agent(tokens, rng)
