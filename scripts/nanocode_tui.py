import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
from textual.app import App, ComposeResult
from textual.widgets import TextArea, Input

from nanojax.checkpointing import load_checkpoint, load_model_config
from nanojax.common import get_base_dir
from nanojax.generation import generate
from nanojax.gpt import GPT
from nanojax.tokenizer import get_tokenizer

checkpoint = "mid"
compute_dtype = jnp.bfloat16
max_tokens = 512
seed = 42
temperature = 0.6

exec(open(os.path.join("nanojax", "configurator.py")).read())

tokenizer = get_tokenizer()
base_dir = get_base_dir()
checkpoint_dir = base_dir / f"{checkpoint}_checkpoints"
model_cfg = load_model_config(checkpoint_dir / "model.zarr")
rng = jax.random.key(seed)
model = GPT.init(model_cfg, rng, "eager")
model = load_checkpoint(checkpoint_dir / "model.zarr", model)

project_root = Path.cwd()

SPECIAL_TOKENS = {name: tokenizer.encode_special(name) for name in [
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|tool_call_start|>", "<|tool_call_end|>",
    "<|tool_result_start|>", "<|tool_result_end|>",
]}

SYSTEM = """you are nanocode, a coding agent.
you communicate in lowercase and think out loud before acting.
to use a tool, structure your call exactly like this:
<|tool_call_start|>Bash<|tool_arg|>command<|tool_val|>echo hello<|tool_call_end|>
"""

def tool_read(args):
    file_path = project_root / args.get("file_path", "")
    offset, limit = args.get("offset", 0), args.get("limit", 100)
    if not file_path.exists(): return "error: file not found"
    if file_path.is_dir(): return "error: is a directory"
    try:
        lines = file_path.read_text().split("\n")
        selected = lines[offset:offset + limit]
        return "\n".join(f"{offset+i+1:5d}{l}" for i, l in enumerate(selected))
    except UnicodeDecodeError:
        return "error: binary file"

def tool_edit(args):
    file_path = project_root / args.get("file_path", "")
    old_string, new_string = args.get("old_string"), args.get("new_string", "")
    if old_string is None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new_string)
        return "created"
    if not file_path.exists(): return "error: file not found"
    content = file_path.read_text()
    if old_string not in content: return "error: old_string not found"
    file_path.write_text(content.replace(old_string, new_string, 1))
    return "success"

def tool_grep(args):
    pattern, path = args.get("pattern", ""), args.get("path", ".")
    target = project_root / path
    if not target.exists(): return "error: path not found"
    cmd = ["grep", "-rn", "--include=*.py", "--include=*.md", "--include=*.json"]
    if args.get("-B"): cmd.extend(["-B", str(args["-B"])])
    if args.get("-A"): cmd.extend(["-A", str(args["-A"])])
    cmd.extend([pattern, str(target)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or "no matches found"
    except subprocess.TimeoutExpired:
        return "error: timed out"

def tool_bash(args):
    command = args.get("command", "")
    if not command: return "error: no command"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, cwd=project_root)
        return (result.stdout + result.stderr).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "error: timed out"

TOOLS = {"Read": tool_read, "Edit": tool_edit, "Grep": tool_grep, "Bash": tool_bash}

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


class NanoCode(App):
    CSS = "TextArea { height: 1fr; } Input { dock: bottom; }"
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield TextArea(read_only=True, soft_wrap=True, show_line_numbers=False)
        yield Input(placeholder="message...")

    def on_mount(self):
        self.rng = jax.random.key(seed)
        self.tokens = [tokenizer.get_bos_token_id(), SPECIAL_TOKENS["<|user_start|>"]]
        self.tokens.extend(tokenizer.encode(SYSTEM))

    def append_log(self, text: str):
        ta = self.query_one(TextArea)
        ta.insert(text + "\n", location=ta.document.end)
        ta.scroll_end(animate=False)

    def on_input_submitted(self, event: Input.Submitted):
        msg = event.value.strip()
        if not msg:
            return
        event.input.clear()
        if msg == "/clear":
            self.query_one(TextArea).load_text("")
            self.tokens = [tokenizer.get_bos_token_id(), SPECIAL_TOKENS["<|user_start|>"]]
            self.tokens.extend(tokenizer.encode(SYSTEM))
            return
        self.append_log(f"user: {msg}")
        self.tokens.extend(tokenizer.encode(msg))
        self.tokens.append(SPECIAL_TOKENS["<|user_end|>"])
        self.tokens.append(SPECIAL_TOKENS["<|assistant_start|>"])
        self.run_worker(self.run_agent, thread=True)

    def run_agent(self):
        max_seq_len = 2 * model_cfg.sequence_len

        while True:
            in_tool_call = False
            tool_buffer = ""
            text_buffer = ""
            tool_called = False

            for token in generate(self.tokens, model, max_tokens, temperature, compute_dtype, pad_token_id=SPECIAL_TOKENS["<|assistant_end|>"], rng=rng, assistant_end_id=SPECIAL_TOKENS["<|assistant_end|>"]):
                token_id = int(token[0])
                self.tokens.append(token_id)
                decoded = tokenizer.decode([token_id])

                if token_id == SPECIAL_TOKENS["<|tool_call_start|>"]:
                    if text_buffer:
                        self.call_from_thread(self.append_log, f"assistant: {text_buffer}")
                        text_buffer = ""
                    in_tool_call = True
                elif token_id == SPECIAL_TOKENS["<|tool_call_end|>"]:
                    tool_called = True
                    break
                elif in_tool_call:
                    tool_buffer += decoded
                else:
                    text_buffer += decoded

                if len(self.tokens) > max_seq_len:
                    break

            if text_buffer:
                self.call_from_thread(self.append_log, f"assistant: {text_buffer}")

            if tool_called:
                tool_name, args = parse_tool_call(tool_buffer)
                self.call_from_thread(self.append_log, f"tool: {tool_name} {args}")
                result = TOOLS.get(tool_name, lambda a: "error: unknown tool")(args)
                self.call_from_thread(self.append_log, f"result: {result[:500]}")
                self.tokens.append(SPECIAL_TOKENS["<|tool_result_start|>"])
                self.tokens.extend(tokenizer.encode(result))
                self.tokens.append(SPECIAL_TOKENS["<|tool_result_end|>"])
            else:
                break

        self.tokens.append(SPECIAL_TOKENS["<|user_start|>"])
        self.rng, _ = jax.random.split(self.rng)


if __name__ == "__main__":
    NanoCode().run()
