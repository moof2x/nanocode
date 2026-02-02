import os
import subprocess
import sys
from pathlib import Path

project_root = Path.cwd()

# --- tool implementations ---

def tool_read(args: dict) -> str:
    file_path = project_root / args.get("file_path", "")
    offset = args.get("offset", 0)
    limit = args.get("limit", 100)
    
    if not file_path.exists():
        return "error: file not found"
    if file_path.is_dir():
        return "error: is a directory"
    
    try:
        lines = file_path.read_text().split("\n")
        end = offset + limit if limit else len(lines)
        selected = lines[offset:end]
        numbered = [f"{offset + i + 1:5d}{line}" for i, line in enumerate(selected)]
        return "\n".join(numbered)
    except UnicodeDecodeError:
        return "error: binary file"

def tool_edit(args: dict) -> str:
    file_path = project_root / args.get("file_path", "")
    old_string = args.get("old_string", None)
    new_string = args.get("new_string", "")
    
    if old_string is None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new_string)
        lines = new_string.split("\n")[:10]
        numbered = [f"{i + 1:5d}{line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)
    
    if not file_path.exists():
        return "error: file not found"
    
    content = file_path.read_text()
    if old_string not in content:
        return "error: old_string not found in file"
    
    new_content = content.replace(old_string, new_string, 1)
    file_path.write_text(new_content)
    
    new_lines = new_content.split("\n")
    new_string_lines = new_string.split("\n")
    for i, line in enumerate(new_lines):
        if new_string_lines[0] in line:
            start = max(0, i - 1)
            end = min(len(new_lines), i + len(new_string_lines) + 1)
            numbered = [f"{start + j + 1:5d}→{new_lines[start + j]}" for j in range(end - start)]
            return "\n".join(numbered)
    
    return "success"

def tool_grep(args: dict) -> str:
    pattern = args.get("pattern", "")
    path = args.get("path", ".")
    before = args.get("-B", 0)
    after = args.get("-A", 0)
    
    target = project_root / path
    if not target.exists():
        return "error: path not found"
    
    cmd = ["grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts", "--include=*.md", "--include=*.txt", "--include=*.json", "--include=*.yaml", "--include=*.yml"]
    if before:
        cmd.extend(["-B", str(before)])
    if after:
        cmd.extend(["-A", str(after)])
    cmd.extend([pattern, str(target)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
        if not output:
            return "no matches found"
        lines = output.split("\n")[:30]
        return "\n".join(lines)
    except subprocess.TimeoutExpired:
        return "error: search timed out"
    except Exception as e:
        return f"error: {e}"

def tool_bash(args: dict) -> str:
    command = args.get("command", "")
    if not command:
        return "error: no command provided"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_root
        )
        output = result.stdout + result.stderr
        output = output.strip()
        if not output:
            return "(no output)"
        lines = output.split("\n")[:50]
        return "\n".join(lines)
    except subprocess.TimeoutExpired:
        return "error: command timed out"
    except Exception as e:
        return f"error: {e}"

TOOLS = {
    "Read": tool_read,
    "Edit": tool_edit,
    "Grep": tool_grep,
    "Bash": tool_bash,
}

def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """
    parse: "Read file_path=src/model.py" or "Bash command=ls -la"
    for multi-word values use quotes: Bash command="ls -la | head"
    """
    parts = text.strip().split()
    if not parts:
        return None
    
    tool_name = parts[0]
    if tool_name not in TOOLS:
        return None
    
    # rejoin and parse with quotes support
    arg_str = " ".join(parts[1:])
    args = {}
    
    import re
    # match key=value or key="value with spaces"
    for match in re.finditer(r'(\S+?)=(?:"([^"]*)"|(\S+))', arg_str):
        key = match.group(1)
        val = match.group(2) if match.group(2) is not None else match.group(3)
        if val.isdigit():
            args[key] = int(val)
        elif val.lower() in ("true", "false"):
            args[key] = val.lower() == "true"
        else:
            args[key] = val
    
    return tool_name, args

def print_help():
    print("""
commands:
  user <message>     - simulate user message
  tool <call>        - simulate tool call
  text <message>     - simulate assistant text response
  reject <reason>    - inject rejection without executing tool
  show               - show current conversation
  clear              - clear conversation
  export [file]      - export to json
  help               - show this help
  quit               - exit

tool call format:
  Read file_path=src/model.py offset=0 limit=50
  Edit file_path=src/model.py old_string=foo new_string=bar
  Grep pattern=def.*init path=src -A=3 -B=2
  Bash command="ls -la"
  Bash command="find . -name '*.py' | head -10"
""")

# --- main loop ---

print(f"project root: {project_root}")
print("tools: Read, Edit, Grep, Bash")
print("type 'help' for commands\n")

conversation = []

while True:
    try:
        line = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\ngoodbye")
        break
    
    if not line:
        continue
    
    cmd, *rest = line.split(" ", 1)
    arg = rest[0] if rest else ""
    
    if cmd == "quit":
        break
    
    elif cmd == "help":
        print_help()
    
    elif cmd == "clear":
        conversation = []
        print("cleared")
    
    elif cmd == "show":
        print("\n--- conversation ---")
        for msg in conversation:
            role = msg["role"]
            if "tool_call" in msg:
                tc = msg["tool_call"]
                print(f"[assistant] tool: {tc['name']} {tc['args']}")
            else:
                content = msg["content"]
                preview = content[:80] + "..." if len(content) > 80 else content
                print(f"[{role}] {preview}")
        print("---\n")
    
    elif cmd == "user":
        if not arg:
            print("usage: user <message>")
            continue
        conversation.append({"role": "user", "content": arg})
        print(f"[user] {arg}")
    
    elif cmd == "text":
        if not arg:
            print("usage: text <message>")
            continue
        conversation.append({"role": "assistant", "content": arg})
        print(f"[assistant] {arg}")
    
    elif cmd == "tool":
        if not arg:
            print("usage: tool <ToolName arg1=val1 arg2=val2>")
            continue
        
        parsed = parse_tool_call(arg)
        if not parsed:
            print(f"invalid tool call: {arg}")
            print("available tools: Read, Edit, Grep, Bash")
            continue
        
        tool_name, tool_args = parsed
        print(f"[assistant] tool: {tool_name} {tool_args}")
        conversation.append({"role": "assistant", "tool_call": {"name": tool_name, "args": tool_args}})
        
        result = TOOLS[tool_name](tool_args)
        print(f"\n--- tool result ---\n{result}\n---")

        if tool_name == "Write":
            try:
                accept = input("accept? (y/n/reason): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\ngoodbye")
                break
        else:
            accept =  "y"
        
        if accept == "y" or accept == "":
            conversation.append({"role": "tool_result", "content": result})
            print("[tool_result] accepted")
        else:
            reason = accept if accept != "n" else "rejected by user"
            conversation.append({"role": "tool_result", "content": f"rejected: {reason}"})
            print(f"[tool_result] rejected: {reason}")
    
    elif cmd == "reject":
        reason = arg if arg else "rejected by user"
        conversation.append({"role": "tool_result", "content": f"rejected: {reason}"})
        print(f"[tool_result] rejected: {reason}")
    
    elif cmd == "export":
        import json
        fname = arg if arg else "conversation.json"
        with open(fname, "w") as f:
            json.dump({"messages": conversation}, f, indent=2)
        print(f"exported to {fname}")
    
    else:
        print(f"unknown command: {cmd}")
        print("type 'help' for commands")
