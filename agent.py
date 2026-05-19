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
    prompt = builder.prompt("Propose the first experiment to improve model performance.")
    print(prompt)
    print("\n" + "="*80 + "\n")
    print(f"Prompt length: {len(prompt)} characters")

if __name__ == "__main__":
    main()
