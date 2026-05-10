# localai
A Claude Code-like terminal agent that runs entirely on your local machine using Ollama. It can execute shell commands, read and edit files, and maintain conversation context — with security guardrails so it doesn't nuke your system.

## Features
- **Environment-aware:** Gathers context on startup (cwd, git status, python env, disk, etc.) so it knows where it is before you say anything.
- **Shell execution:** Runs commands and feeds output back into the conversation. Dangerous commands require confirmation, hard-blocked ones are refused outright.
- **File operations:** Read, write, and surgical find-and-replace edits. Warns before overwriting existing files.
- **Rolling context:** Keeps the last 20 messages in memory so it doesn't lose track mid-task.
- **Write sandbox:** Optionally restrict file writes to a specific directory tree.
- **Tool injection protection:** Strips fake tool blocks from command output before they hit the model context.

## Installation
```bash
pip install ollama
ollama pull qwen2.5-coder:7b
```

## Usage
```bash
python localai.py
```

Switch model on the fly:
```bash
LOCALAI_MODEL=qwen2.5:14b python localai.py
```

Sandbox file writes to a directory:
```bash
LOCALAI_WRITE_ROOT=/home/arash/projects python localai.py
```

Inside the chat: `clear` resets context, `model <name>` switches model, `exit` quits.

## Contributing
Open issues or PRs if something's broken or you want to add something.
