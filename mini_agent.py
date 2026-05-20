import json
import subprocess
from llama_cpp import Llama, LlamaGrammar

MODEL_PATH = "/home/RYVEN/mini_agent/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

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
            verbose=False)

prompt = """
System: You are an AI Agent. You must output JSON. You have a tool called run_shell that takes a cmd argument.
User: What files are in my current directory?
"""

output = llm(prompt, grammar=grammar, max_tokens=200)

raw_text = output['choices'][0]['text']


def execute_shell(cmd):
    approval = input(f"\n[SYSTEM] Allow to run: '{cmd}'? (y/n)")
    if approval.lower() != 'y':
        return "Command aborted by the user."

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout
    else:
        return result.stderr

parsed_data = json.loads(raw_text)
if parsed_data["name"] == "run_shell":
    cmd = parsed_data["args"]["cmd"]
    observation = execute_shell(cmd)
    print(observation)
elif parsed_data["type"] == "final":
    print("the ai says: [content]")


