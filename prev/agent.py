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


MAX_TOOL_OUTPUT = 4000

class OllamaModelClient:
    def __init__(self, model, host, temperature, top_p, timeout):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def complete(self, prompt, max_new_tokens):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        request = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")



#-- Workspace Context:

def clip(text, limit=MAX_TOOL_OUTPUT):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"

DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")


class WorkspaceContext:
    def __init__(self, cwd, repo_root, branch, default_branch, status, recent_commits, project_docs):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs

    @classmethod
    def build(cls, cwd):
        cwd = Path(cwd).resolve()

        def git(args, fallback=""):
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        docs = {}
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue
                docs[key] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main").removeprefix("origin/"),
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
        )

    def text(self):
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        return "\n".join([
            "Workspace:",
            f"- cwd: {self.cwd}",
            f"- repo_root: {self.repo_root}",
            f"- branch: {self.branch}",
            f"- default_branch: {self.default_branch}",
            "- status:",
            self.status,
            "- recent_commits:",
            commits,
            "- project_docs:",
            docs,
        ])


## -- Tools:

def list_files(path, workspace):
    target = (Path(workspace)/path).resolve()
    ws = Path(workspace).resolve()

    if not target.is_relative_to(ws):
        return f"Error: outside workspace"
    if not target.exists():
        return f"Error: no such directory"
    if not target.is_dir():
        return f"Error: not a directory"

    lines = []
    for entry in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        if entry.name in {".git", "__pycache__"}:
            continue
        kind = "F" if entry.is_file() else "D"
        rel = entry.relative_to(ws)
        lines.append(f"[{kind}] {rel}")
    
    return "\n".join(lines) or "(empty)"


def read_file(path, workspace, start=1, end=200):
    target = (Path(workspace) / path).resolve()
    ws = Path(workspace).resolve()
    
    if not target.is_relative_to(ws):
        return f"Error: outside workspace"
    if not target.is_file():
        return f"Error: not a file"
    
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(f"{n:>4}: {line}" for n, line in enumerate(lines[start-1:end], start=start))
    return f"# {target.relative_to(ws)}\n{body}"


def patch_file(path, old_txt, new_txt, workspace):
    target = (Path(workspace) / path).resolve()
    ws = Path(workspace).resolve()
    
    if not target.is_relative_to(ws):
        return f"Error: outside workspace"
    if target.name in {"prepare.py", "program.md"}:
        return f"Error: protected file"
    if not target.is_file():
        return f"Error: not a file"
    
    text = target.read_text(encoding="utf-8")
    count = text.count(old_text)
    
    if count == 0:
        return "Error: old_text not found"
    if count > 1:
        return f"Error: old_text appears {count} times (must be unique)"
    
    target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Patched {path}"



def run_experiment(diff, hypothesis, workspace):
    return f"Experiment placeholder: {hypothesis[:50]}..."


TOOLS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "risky": False,
        "description": "List files in the workspace directory.",
    },
    "read_file": {
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "risky": False,
        "description": "Read a file by line range.",
    },
    "patch_file": {
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "risky": True,
        "description": "Replace exact text block in a file.",
    },
    "run_experiment": {
        "schema": {"diff": "str", "hypothesis": "str"},
        "risky": True,
        "description": "Apply diff, train for 5 min, measure val_bpb, keep or discard.",
    },
}


TOOL_RUNNERS = {
    "list_files": list_files,
    "read_file": read_file,
    "patch_file": patch_file,
    "run_experiment": run_experiment,
}


def run_tool(name, args, workspace, approval_mode):
    func = TOOL_RUNNERS.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'"
    
    kwargs = dict(args)
    kwargs["workspace"] = workspace

    if name in {"patch_file", "run_experiment"}:
        if approval_mode == "never":
            return f"Error: approval denied"
        if approval_mode == "ask":
            ans = input(f"Approve {name}({args})? [y/N] ")
            if ans.strip().lower() not in {"y", "yes"}:
                return f"Error: approval denied"
    
    try:
        return func(**kwargs)
    except Exception as exc:
        return f"Error: {name} failed: {exc}"


## -- Prompt:

class PromptBuilder:
    def __init__(self, workspace, goal, protected_files):
        self.workspace = workspace
        self.goal = goal
        self.protected_files = protected_files
        self.experiment_memory = []
        self.history = []
        self.tools = TOOLS

    def build_prefix(self):
        tool_list = []
        for name, tool in self.tools.items():
            fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
            risk = "approval required" if tool["risky"] else "safe"
            tool_list.append(f"- {name}({fields}) [{risk}] {tool['description']}")
        tool_text = "\n".join(tool_list)
        examples = "\n".join(
                [
                    '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                    '<tool>{"name":"read_file","args":{"path":"train.py","start":1,"end":50}}</tool>',
                    '<tool name="patch_file" path="train.py"><old_text>DEPTH = 8</old_text><new_text>DEPTH = 12</new_text></tool>',
                    '<tool name="run_experiment"><diff>--- a/train.py\n+++ b/train.py\n@@ -15,7 +15,7 @@\n-DEPTH = 8\n+DEPTH = 12\n</diff><hypothesis>Increase depth from 8 to 12 layers</hypothesis></tool>',
                    '<final>I have completed all experiments.</final>',
                ]
            )
        rules = "\n".join([
            "- Return exactly one <tool>...</tool> or one <final>...</final>.",
            "- Tool calls must look like:",
            '  <tool>{"name":"tool_name","args":{...}}</tool>',
            "- For patch_file and run_experiment with multi-line text, prefer XML style.",
            "- Final answers must look like: <final>your answer</final>",
            "- Never invent tool results.",
            "- Keep answers concise and concrete.",
            "- Do not repeat the same tool call with the same arguments if it did not help.",
            f"- Your goal: {self.goal}",
            f"- Protected files (never modify): {', '.join(self.protected_files)}",
            "- Only modify train.py. This is your sole target file.",
            "- Each experiment runs for exactly 5 minutes. The harness handles timing.",
            "- After run_experiment, you will see the result and metric.",
            "- Learn from failed experiments. Do not repeat changes that crashed or worsened the metric.",
            "- Propose one experiment at a time. Wait for the result before proposing the next.",
        ])

        return "\n\n".join([
            "You are an autonomous research agent, a small local coding agent running through Ollama.",
            "Rules:\n" + rules,
            "Tools:\n" + tool_text,
            "Valid response examples:\n" + examples,
            self.workspace.text(),
        ])


    def memory(self):
        best_metric = float('inf')
        best_exp = None
        kept_count = 0
        failed_patterns = []

        for exp in self.experiment_memory:
            if exp["status"] == "kept" and exp["metric"] < best_metric:
                best_metric = exp["metric"]
                best_exp = exp["id"]
            elif exp["status"] in ("discarded", "crashsed"):
                failed_patterns.append(f"#{exp["id"]}: {exp["hypothesis"][:60]}...({exp["status"]})")

        best_line = ( 
                     f"Best so far: {best_metric:.4f} (experiment #{best_exp})" 
                     if best_exp else "No successful experiments yet."
                     )

        failed_lines = "\n".join(f"- {p}" for p in failed_patterns[-5:]) or "- none"
        
        return "\n".join([
        "Experiment Memory:",
        f"- Total experiments: {len(self.experiment_memory)}",
        best_line,
        f"- Recent failures (do not repeat):",
        failed_lines,
        ])


    def history_text(self):
        if not self.history:
            return "- empty"
        
        lines = []
        for item in self.history[-3:]:  # Only last 3 to save context
            lines.append(f"[{item['role']}] {item['content'][:200]}")
        
        return "\n".join(lines)


    def prompt(self, task="Propose the next experiment."):
        return "\n\n".join([
        self.build_prefix(),
        self.memory(),
        "Recent History:\n" + self.history_text(),
        "Current Task:\n" + task,
    ])



def parse_response(response: str):
    """Parse agent response into tool call, final answer, or error."""
    response = str(response)
    
    # Try JSON-style tool: <tool>{"name":"...","args":{...}}</tool>
    if "<tool>" in response and ("<final>" not in response or response.find("<tool>") < response.find("<final>")):
        body = extract(response, "tool")
        try:
            payload = json.loads(body)
        except Exception:
            return "retry", "Malformed tool JSON. Use valid <tool>{...}</tool> format."
        
        if not isinstance(payload, dict):
            return "retry", "Tool payload must be a JSON object."
        if not str(payload.get("name", "")).strip():
            return "retry", "Tool payload missing name."
        
        args = payload.get("args", {})
        if args is None:
            payload["args"] = {}
        elif not isinstance(args, dict):
            return "retry", "Tool args must be a JSON object."
        
        return "tool", payload
    
    # Try XML-style tool: <tool name="..." path="..."><content>...</content></tool>
    if "<tool" in response and ("<final>" not in response or response.find("<tool") < response.find("<final>")):
        payload = parse_xml_tool(response)
        if payload is not None:
            return "tool", payload
        return "retry", "Malformed XML tool. Check your format."
    
    # Try final answer: <inal>...</final>
    if "<final>" in response:
        final = extract(response, "final").strip()
        if final:
            return "final", final
        return "retry", "Empty <final> answer. Provide content."
    
    # Raw text with no tags — treat as final if non-empty
    stripped = response.strip()
    if stripped:
        return "final", stripped
    
    return "retry", "Empty response. Use <tool> or <final>."


def extract(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>."""
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    if start == -1:
        return text
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return text[start:].strip()
    return text[start:end].strip()


def parse_xml_tool(raw: str):
    """Parse XML-style tool call like <tool name="patch_file" path="...">...</tool>."""
    match = re.search(r'<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>', raw, re.S)
    if not match:
        return None
    
    attrs = parse_attrs(match.group("attrs"))
    name = str(attrs.pop("name", "")).strip()
    if not name:
        return None
    
    body = match.group("body")
    args = dict(attrs)
    
    # Extract common XML content tags
    for key in ("content", "old_text", "new_text", "diff", "hypothesis", "command", "task", "pattern", "path"):
        if f"<{key}>" in body:
            args[key] = extract_raw(body, key)
    
    # Fallback: if no recognized tags, treat body as content for write_file
    body_text = body.strip("\n")
    if name == "write_file" and "content" not in args and body_text:
        args["content"] = body_text
    if name == "run_experiment" and "diff" not in args and body_text:
        # Try to extract diff from body
        if "---" in body_text:
            args["diff"] = body_text
    
    return {"name": name, "args": args}


def parse_attrs(text: str):
    """Parse attribute string like name="patch_file" path="train.py"."""
    attrs = {}
    for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
        attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
    return attrs


def extract_raw(text: str, tag: str) -> str:
    """Extract raw content between tags (preserves whitespace)."""
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    if start == -1:
        return text
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return text[start:]
    return text[start:end]




### args
def build_arg_parser():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Minimal coding agent for Ollama models.",
        )
    parser.add_argument("--cwd", default=".", help="Working directory")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama model name")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama server URL")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask")
    parser.add_argument("--max-steps", type=int, default=6)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    client = OllamaModelClient(
            host=args.host,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.ollama_timeout
        )

    workspace_context = WorkspaceContext.build(args.cwd)
    
    builder = PromptBuilder(
        workspace=workspace_context,
        goal="Minimize val_bpb (validation bits per byte). Lower is better.",
        protected_files={"prepare.py", "program.md"}
    )

    for step in range(6):
        print(f"\n{'='*50}")
        print(f"[Step {step+1}/6]")
        
        prompt = builder.prompt()
        print(f"Prompt: {len(prompt)} chars")
        
        response = client.complete(prompt, max_new_tokens=512)
        print(f"Agent: {response[:150]}...")
        
        kind, payload = parse_response(response)
        
        if kind == "final":
            print(f"Done: {payload}")
            break
        
        if kind == "retry":
            print(f"Parse error: {payload}")
            builder.history.append({"role": "system", "content": f"Error: {payload}"})
            continue
        
        if kind == "tool":
            name = payload.get("name")
            args = payload.get("args", {})
            print(f"Tool: {name}({args})")
            
            result = run_tool(name, args, str(workspace_context.cwd), "ask")
            print(f"Result: {result[:150]}...")
            
            builder.history.append({
                "role": "tool",
                "content": f"{name}: {result}"
            })
    
    print("\nLoop finished.")


if __name__ == "__main__":
    main()


