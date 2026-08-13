import os
import subprocess
import ollama

from groq import Groq 
from anthropic import Anthropic 
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use tools available to solve tasks. Act, don't explain."

TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command" : {"type": "string"}},
        "required" : ["command"],
        },
    }]

# Tool execution ----
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def agent_loop(messages: list):
    messages.append({"role": "system", "content": "You are a coding agent at {os.getcwd()}. Use tools available to solve tasks. Act, don't explain."})
    
    while True:
        response = ollama.chat(
            model=MODEL, 
            messages=messages,
            tools=TOOLS,
        )
        message = response['message']
        messages.append({"role": "assistant", "content": message})

        if message.get("tool_use"):
            for tool_call in message['tool_calls']:
                func_name = tool_call['function']['name']
                args = tool_call['function']['arguments']
                
                print(f"\033[33m$ {block.input['command']}\033[0m")

                output = run_bash(args['command'])


        results = []
        for block in response.content:
            if getattr(block, 'type', None) == "tool_use":
                command = getattr(block, 'input', {}).get('command')
                if not command and hasattr(block, 'arguments'):
                    command = block.arguments.get('command')

                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })


        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("Agent Loop")

    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)

        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
