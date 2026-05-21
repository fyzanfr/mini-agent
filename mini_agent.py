import json
import subprocess
from llama_cpp import Llama, LlamaGrammar

MODEL_PATH = "/home/RYVEN/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"


# -- Utility:

def slice(output, max_chars=2000):
    if not output:
        return ""
    text = str(output)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n.. [OUTPUT TRUNCATED..]"


AGENT_GRAMMAR = r"""
root ::= tool | final

tool ::= "{" space tkey space ":" space tval space "," space nkey space ":" space toolname space "," space akey space ":" space object space "}"
final ::= "{" space tkey space ":" space fval space "," space ckey space ":" space string space "}"

tkey ::= "\"type\""
tval ::= "\"tool\""
fval ::= "\"final\""
nkey ::= "\"name\""
akey ::= "\"args\""
ckey ::= "\"content\""

toolname ::= "\"list_files\"" | "\"read_file\"" | "\"write_file\"" | "\"run_shell\""

object ::= "{" space "}" | "{" space pair (space "," space pair)* space "}"
pair ::= string space ":" space value
value ::= string | number | "true" | "false" | "null" | object

string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
space ::= [ \t\n]*
"""

grammar = LlamaGrammar.from_string(AGENT_GRAMMAR)

llm = Llama(model_path=MODEL_PATH,
            n_ctx = 4096,
            verbose=False)

system_prompt = """You are an elite, local AI agent running natively on Arch Linux.
You have FULL ACCESS to the local file system. 

CRITICAL RULES:
- You MUST respond in pure, valid JSON.
- To execute a command: {"type": "tool", "name": "run_shell", "args": {"cmd": "<command>"}}
- To answer the user: {"type": "final", "content": "<your response>"}
- If the user asks for information, you MUST include the actual data from your Observation in your 'final' content. Do not use placeholders.
"""

# -- Memory:
messages = [
        {"role": "system", "content": system_prompt}
    ]




def execute_shell(cmd):
    approval = input(f"\n[SYSTEM] Allow to run: '{cmd}'? (y/n)")
    if approval.lower() != 'y':
        return "Command aborted by the user."

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return slice(result.stdout)
    else:
        return slice(result.stderr)


def read_file(path):
    pass


AVAILABLE_TOOLS = {
        "run_shell": execute_shell,
        "read_file": read_file
        }





while True:
    user_input = input("\n==>>>")
    if user_input.strip().lower() == ["exit", "quit", "bye"]:
        print("Shutting down agent.")
        break 

    messages.append({"role": "user", "content": user_input})


    while True:
        output = llm.create_chat_completion(
                messages=messages, 
                grammar=grammar, 
                max_tokens=1024
        )

        raw_text = output['choices'][0]['message']['content']

        try:
            parsed_data = json.loads(raw_text)
        except json.JSONDecodeError:
            print(f"[SYSTEM] Bad Json generation. Retrying...")

            messages.append({"role":"assistant","content":raw_text})
            messages.append({"role":"user", "content":"Your last response was invalid JSON, Try Again..."})
            continue

        if parsed_data["type"] == "tool":
            tool_name = parsed_data["name"]
            tool_args = parsed_data["args"]

            if tool_name in AVAILABLE_TOOLS:
                observation = AVAILABLE_TOOLS[tool_name](**tool_args)
            else:
                observation = f"Error: Tool {tool_name} not found"

            
            if observation is None:
                observation = "Success: Tool executed but returned no output."
            else:
                observation = str(observation)
            
            print(f"\n[DEBUG - OS Output]:\n{observation[:200]}...")


            messages.append({"role": "assistant", "content": raw_text})
            
            messages.append({
                "role": "user", 
                "content": f"Observation from {tool_name}:\n{observation}\n\nNow provide the 'final' JSON answer."
            })
            continue
        
        elif parsed_data["type"] == "final":
            content = parsed_data["content"]
            print(f"\n[AI]: {content}")
            
            messages.append({"role": "assistant", "content": raw_text})
            break
