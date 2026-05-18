import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


def ask_ollama(prompt:str, model:str = "qwen2.5-coder:7b", host:str = "http://localhost:11434") -> str:
    payload = {
            "model":model,
            "prompt":prompt,
            "stream":False}

    request = urllib.request.Request(
            host + "/api/generate",
            data = json.dumps(payload).encode("utf-8"),
            headers = {"Content-Type":"application/json"},
            method="POST",
            )

    try:
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama.\n"
            "Make sure `ollama serve` is running and the model is available.\n"
            f"Host: {host}\n"
            f"Model: {model}"
        ) from exc

    if data.get("error"):
        raise RuntimeError(f"Ollama error :{data['error']}")
    return data.get("response", "")


# reading the file
def tool_read_file(path:str, workspace:str) -> str:
    req_path = Path(path).resolve()
    workspace_path = Path(workspace).resolve()

    if not req_path.is_relative_to(workspace_path):
        return f"Error : Path {path} is outside workspace {workspace}"
    
    if not req_path.exists():
        return f"Error : Path {path} does not exist"

    if not req_path.is_file():
        return f"Error : Path {path} is not a file (might be a dir)"
    return req_path.read_text()


PROTECTED_FILES = {"prepare.py", "program.md"}

# writing file
def tool_write_file(path:str, content:str, workspace:str, approve_mode:str = "ask") -> str:

    req_path = Path(path).resolve()
    workspace_path = Path(workspace).resolve()

    if not req_path.is_relative_to(workspace_path):
        return f"Error: Path {path} is outside the workspace."

    if req_path.name in PROTECTED_FILES:
        return f"Error: {req_path.name} is protected and cannot be modified."

    if not content:
        return f"Error: Cannot write empty content."

    existed = req_path.exists()
    if existed and approve_mode == "ask":
        return f"Approval needed: Overwrite {path} ?"

    if existed and approval_mode not in ("ask", "auto"):
        return f"Error: unknown approval mode '{approval_mode}'"
    
    req_path.write_text(content)

    if existed:
        return f"Overwrote {path}"
    else:
        return f"Created {path}"


# test: print(tool_write_file("test.txt", "hello", "/home/RYVEN/mini_agent/", "ask"))

SYSTEM_PROMPT = """ You are an autonomous ML research agent. You have access to these tools:
- read_file(path): Read a file in the workspace
- write_file(path, content): Write content to a file

When you want to use a tool, output EXACTLY this XML format:
<tool>
<name>read_file</name>
<path>train.py</path>
</tool>

If you are done and have no more actions, output:
<final>I have completed the task.</final>"""



def parse_response(response:str):
    tool_match = re.search(r'<tool>(.*?)</tool>', response, re.DOTALL)
    if tool_match:
        inner = tool_match.group(1)

        name_match = re.search(r'<name>(.*?)</name>', inner, re.DOTALL)
        path_match = re.search(r'<path>(.*?)</path>', inner, re.DOTALL)

        if name_match is None:
            return {"type": "error", "raw": response, "reason": "tool missing name"}
        # build result
        result = {
                "type": "tool",
                "name": name_match.group(1).strip()
                }

        if path_match:
            result["path"] = path_match.group(1).strip()

        return result

    final_match = re.search(r'<final>(.*?)</final>', response, re.DOTALL)
    if final_match:
        return {
                "type": "final",
                "message": final_match.group(1).strip()
                }
    return {"type": "error", "raw": response, "reason": "no tool or final tag found!"}


test1 = "<tool>\n<name>tool_read_file</name>\n<path>train.py</path>\n</tool>"
test2 = "<final>I am done with this task</final>"
test3 = "I think I should read the file first"

print(parse_response(test1))
print(parse_response(test2))
print(parse_response(test3))
