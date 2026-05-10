#!/usr/bin/env python3
"""
localai.py - A Claude Code-like agent using local Ollama models
Features: shell execution, file read/write/edit, conversation context, security guardrails
"""

import os
import sys
import subprocess
import json
import re
import ollama
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = os.environ.get("LOCALAI_MODEL", "qwen2.5-coder:7b")
MAX_CONTEXT = 20          # max messages kept in memory (rolling window)
ALLOWED_SHELL = True      # set False to disable shell execution entirely
MAX_OUTPUT_CHARS = 8000   # truncate huge command outputs fed back to model

# Restrict file ops to this directory tree (None = no restriction = anywhere)
# Set to a string like "/home/arash/projects" to sandbox writes
WRITE_ROOT = os.environ.get("LOCALAI_WRITE_ROOT", None)

# Patterns that trigger a confirmation prompt before running.
# These are regex patterns matched against the full command string.
DANGEROUS_PATTERNS = [
    r"\brm\b",
    r"\bmv\b",
    r"\bdd\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"\bmkfs\b",
    r"\bsudo\b",
    r"\bkill\b",
    r"\bpkill\b",
    r"\bcurl\b.*\|\s*(bash|sh|python)",   # curl piped to shell
    r"\bwget\b.*\|\s*(bash|sh|python)",
    r">\s*/etc/",                          # writing to /etc
    r">\s*/usr/",
    r">\s*/bin/",
    r":\(\)\{.*\}",                        # fork bomb pattern
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bformat\b",
    r"\btruncate\b",
]

# Hard-blocked commands — never run these, no matter what, no confirmation offered
BLOCKED_PATTERNS = [
    r":\(\)\{.*:\|:&\}",                  # fork bomb
    r"\brm\s+-rf\s+/\b",                  # rm -rf /
    r"\brm\s+--no-preserve-root",
    r"\bdd\b.*of=/dev/(s|h|nv)d",         # dd to disk device
    r"\bmkfs\b.*/dev/",                    # format a device
]

SYSTEM_PROMPT = f"""You are a local AI coding and research assistant running on the user's machine.
You can execute shell commands, read files, write/edit files, and help with code, bioinformatics, and research tasks.

Today: {datetime.now().strftime('%Y-%m-%d')}

## Tools available — use EXACTLY this JSON format, one block per response:

Shell command:
```tool
{{"action": "shell", "command": "ls -la"}}
```

Read file:
```tool
{{"action": "read", "path": "/path/to/file"}}
```

Write file (full overwrite):
```tool
{{"action": "write", "path": "/path/to/file", "content": "full file content here"}}
```

Edit file (find and replace):
```tool
{{"action": "edit", "path": "/path/to/file", "old": "exact text to replace", "new": "replacement text"}}
```

## Rules:
- Use tools when needed, then explain what you did and why
- For shell commands, always show the command before running it
- Prefer safe, targeted commands — avoid destructive flags unless explicitly asked
- You can chain multiple tool calls by putting them in sequence in your response
- After tool output is returned, continue your response naturally
- Never attempt to exfiltrate data, call remote services, or escalate privileges unless the user explicitly asks
"""

# ── Display helpers ────────────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def print_separator():
    print(f"{DIM}{'─' * 60}{RESET}")

def print_tool_call(tool: dict):
    action = tool.get("action", "?")
    if action == "shell":
        print(f"\n{YELLOW}⚡ shell:{RESET} {tool.get('command', '')}")
    elif action == "read":
        print(f"\n{YELLOW}📖 read:{RESET} {tool.get('path', '')}")
    elif action == "write":
        print(f"\n{YELLOW}✏️  write:{RESET} {tool.get('path', '')} ({len(tool.get('content',''))} chars)")
    elif action == "edit":
        print(f"\n{YELLOW}🔧 edit:{RESET} {tool.get('path', '')}")
    else:
        print(f"\n{YELLOW}🔨 tool:{RESET} {tool}")

def print_tool_result(result: str):
    lines = result.strip().split("\n")
    for line in lines[:30]:
        print(f"  {DIM}{line}{RESET}")
    if len(lines) > 30:
        print(f"  {DIM}... ({len(lines)-30} more lines){RESET}")


# ── Security helpers ──────────────────────────────────────────────────────────

def is_blocked(command: str) -> bool:
    """Hard block — never run, no confirmation offered."""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    return False

def is_dangerous(command: str) -> bool:
    """Soft block — ask user before running."""
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    return False

def confirm(prompt: str) -> bool:
    """Ask user y/n. Returns True if confirmed."""
    try:
        answer = input(f"{RED}⚠ {prompt} [y/N]:{RESET} ").strip().lower()
        return answer in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False

def resolve_safe_path(path: str, must_exist: bool = False) -> tuple[Path | None, str]:
    """
    Resolve path and enforce WRITE_ROOT sandbox if set.
    Returns (resolved_path, error_string_or_empty).
    """
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        return None, f"invalid path: {e}"

    if WRITE_ROOT:
        root = Path(WRITE_ROOT).expanduser().resolve()
        try:
            p.relative_to(root)  # raises ValueError if not under root
        except ValueError:
            return None, f"path '{p}' is outside the allowed write root '{root}'"

    if must_exist and not p.exists():
        return None, f"file not found: {path}"

    return p, ""

def sanitize_tool_result(result: str) -> str:
    """
    Strip anything that looks like a tool block from command output,
    so a malicious command can't inject fake tool calls into the model context.
    """
    # remove ```tool ... ``` blocks from output before feeding back to model
    sanitized = re.sub(r"```tool[\s\S]*?```", "[tool block stripped from output]", result)
    # also strip <tool_results> tags that could confuse context parsing
    sanitized = re.sub(r"<tool_results>[\s\S]*?</tool_results>", "[tool_results block stripped]", sanitized)
    return sanitized


# ── Tool execution ─────────────────────────────────────────────────────────────

def run_shell(command: str) -> str:
    """Execute a shell command with security checks."""
    if not ALLOWED_SHELL:
        return "ERROR: shell execution is disabled"

    # hard block first — no confirmation, just refuse
    if is_blocked(command):
        return f"BLOCKED: this command matches a hard-blocked pattern and will never run: {command}"

    # soft block — ask user
    if is_dangerous(command):
        print(f"\n{YELLOW}⚡ shell:{RESET} {command}")
        if not confirm(f"This command looks potentially destructive. Run it?"):
            return "CANCELLED: user declined to run this command"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd(),
            env={**os.environ}   # inherit env but don't let subprocess modify parent
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
        if not output.strip():
            output = "(no output)"
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n... [truncated at {MAX_OUTPUT_CHARS} chars]"
        return sanitize_tool_result(output)
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 60s"
    except Exception as e:
        return f"ERROR: {e}"


def read_file(path: str) -> str:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"ERROR: file not found: {path}"
        if p.stat().st_size > 500_000:
            return f"ERROR: file too large (>{500_000} bytes), read it in chunks"
        content = p.read_text(errors="replace")
        if len(content) > MAX_OUTPUT_CHARS:
            content = content[:MAX_OUTPUT_CHARS] + f"\n... [truncated]"
        return sanitize_tool_result(content)
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str) -> str:
    p, err = resolve_safe_path(path)
    if err:
        return f"ERROR: {err}"
    try:
        # warn if overwriting existing file
        if p.exists():
            print(f"\n{YELLOW}✏️  write:{RESET} {p} (already exists)")
            if not confirm("Overwrite existing file?"):
                return "CANCELLED: user declined to overwrite file"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"OK: wrote {len(content)} chars to {p}"
    except Exception as e:
        return f"ERROR: {e}"


def edit_file(path: str, old: str, new: str) -> str:
    p, err = resolve_safe_path(path, must_exist=True)
    if err:
        return f"ERROR: {err}"
    try:
        content = p.read_text(errors="replace")
        if old not in content:
            return f"ERROR: pattern not found in {path}. Check exact whitespace/newlines."
        count = content.count(old)
        if count > 1:
            return f"ERROR: pattern found {count} times — be more specific to avoid ambiguity"
        new_content = content.replace(old, new, 1)
        p.write_text(new_content)
        return f"OK: edited {path} (replaced 1 occurrence)"
    except Exception as e:
        return f"ERROR: {e}"


def dispatch_tool(tool: dict) -> str:
    action = tool.get("action", "")
    if action == "shell":
        return run_shell(tool.get("command", ""))
    elif action == "read":
        return read_file(tool.get("path", ""))
    elif action == "write":
        return write_file(tool.get("path", ""), tool.get("content", ""))
    elif action == "edit":
        return edit_file(tool.get("path", ""), tool.get("old", ""), tool.get("new", ""))
    else:
        return f"ERROR: unknown action '{action}'"


# ── Response parsing ───────────────────────────────────────────────────────────

def extract_tools(text: str) -> list[dict]:
    """Extract all ```tool ... ``` blocks from model response."""
    pattern = r"```tool\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    tools = []
    for m in matches:
        try:
            tools.append(json.loads(m.strip()))
        except json.JSONDecodeError as e:
            tools.append({"_parse_error": str(e), "_raw": m.strip()})
    return tools


def strip_tool_blocks(text: str) -> str:
    """Remove ```tool blocks from display text."""
    return re.sub(r"```tool\s*[\s\S]*?```", "", text).strip()


# ── Main agent loop ────────────────────────────────────────────────────────────

def chat_with_tools(messages: list, user_input: str) -> tuple[str, list]:
    """
    Send user_input to the model, handle tool calls, return final response.
    Returns (final_text, updated_messages)
    """
    messages.append({"role": "user", "content": user_input})

    # rolling context window
    system = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = messages[-MAX_CONTEXT:]

    max_tool_rounds = 8
    final_text = ""

    for round_num in range(max_tool_rounds):
        # stream the response
        print(f"\n{CYAN}assistant:{RESET} ", end="", flush=True)
        full_response = ""

        stream = ollama.chat(
            model=MODEL,
            messages=system + context,
            stream=True,
            options={"num_ctx": 8192, "temperature": 0.2}
        )

        for chunk in stream:
            content = chunk["message"].get("content", "")
            full_response += content
            # only print non-tool text in real time
            if "```tool" not in full_response or full_response.count("```tool") == full_response.count("```", full_response.find("```tool")):
                print(content, end="", flush=True)

        print()  # newline after stream

        tools = extract_tools(full_response)

        if not tools:
            # no tool calls — we're done
            final_text = full_response
            messages.append({"role": "assistant", "content": full_response})
            break

        # execute tools and build tool result message
        clean_text = strip_tool_blocks(full_response)
        if clean_text:
            print(f"\n{clean_text}")

        tool_results = []
        for tool in tools:
            if "_parse_error" in tool:
                print(f"\n{RED}tool parse error:{RESET} {tool['_parse_error']}")
                tool_results.append(f"PARSE ERROR: {tool['_parse_error']}\nRaw: {tool.get('_raw','')}")
                continue

            print_tool_call(tool)
            result = dispatch_tool(tool)
            print_tool_result(result)
            tool_results.append(f"[{tool.get('action','?')}] {result}")

        # feed results back into context
        combined_results = "\n\n".join(tool_results)
        messages.append({"role": "assistant", "content": full_response})
        messages.append({"role": "user", "content": f"<tool_results>\n{combined_results}\n</tool_results>"})
        context = messages[-MAX_CONTEXT:]

    else:
        print(f"\n{RED}⚠ max tool rounds reached{RESET}")

    return final_text, messages


def gather_context() -> str:
    """Run a few harmless commands on startup so the model knows where it is."""
    cmds = {
        "cwd":      "pwd",
        "user":     "whoami",
        "host":     "hostname",
        "os":       "uname -sr",
        "ls":       "ls -la",
        "git":      "git status --short 2>/dev/null || echo '(not a git repo)'",
        "python":   "python3 --version 2>&1",
        "conda":    "conda info --envs 2>/dev/null | head -20 || echo '(no conda)'",
        "disk":     "df -h . | tail -1",
    }
    parts = []
    for label, cmd in cmds.items():
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            out = (r.stdout or r.stderr or "").strip()
            if out:
                parts.append(f"[{label}]\n{out}")
        except Exception:
            pass
    return "\n\n".join(parts)


def main():
    global MODEL
    print(f"{BOLD}{CYAN}localai{RESET} — model: {MODEL}")
    print(f"{DIM}type 'exit' to quit | 'model <name>' to switch | 'clear' to reset context{RESET}")
    print_separator()

    # inject environment context as first system-level message
    print(f"{DIM}gathering environment context...{RESET}", end="\r")
    ctx = gather_context()
    print(f"{DIM}context ready.{' ' * 20}{RESET}")

    messages = [
        {
            "role": "user",
            "content": f"<environment_context>\n{ctx}\n</environment_context>\n\nYou now know where you are. Acknowledge briefly (one line max) then wait for my first task."
        }
    ]
    # silently prime the model — don't show this exchange, just swallow the ack
    try:
        r = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            stream=False,
            options={"num_ctx": 8192, "temperature": 0.1}
        )
        ack = r["message"]["content"].strip()
        messages.append({"role": "assistant", "content": ack})
        print(f"{DIM}model:{RESET} {ack}")
        print_separator()
    except Exception as e:
        print(f"{RED}warning: could not prime model: {e}{RESET}")
        messages = []

    while True:
        try:
            user_input = input(f"\n{GREEN}you:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}bye{RESET}")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print(f"{DIM}bye{RESET}")
            sys.exit(0)

        if user_input.lower() == "clear":
            messages = []
            print(f"{DIM}context cleared{RESET}")
            continue

        if user_input.lower().startswith("model "):
            MODEL = user_input[6:].strip()
            print(f"{DIM}switched to model: {MODEL}{RESET}")
            continue

        _, messages = chat_with_tools(messages, user_input)
        print_separator()


if __name__ == "__main__":
    main()
