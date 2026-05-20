import json
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


# Qwen chat format
def qwen_prompt(system, user):
    return f"""<<||<|<|im_start|>system
{system}|</think>
<<||<|<|im_start|>user
{user}|</think>
<<||<|<|im_start|>assistant
"""

system = """You are a coding agent. You have these tools:
- list_files(path='.')
- read_file(path, start=1, end=50)
- write_file(path, content)
- run_shell(command, timeout=20)

RULES:
1. You MUST respond with ONLY one JSON object.
2. Use a tool if you need to inspect the workspace.
3. Do not guess about file contents.

Respond with JSON only."""

# Load model + grammar
print("Loading model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
    verbose=False,
)

print("Compiling grammar...")
try:
    grammar = LlamaGrammar.from_string(AGENT_GRAMMAR)
    print("Grammar compiled successfully!")
except Exception as e:
    print(f"Grammar failed: {e}")
    exit(1)

print("Ready.\n")

# Test: Ask it to list files
user = "What files are in the workspace?"

prompt = qwen_prompt(system, user)
print(f"Prompt length: {len(prompt)} chars")
print("Generating...")

output = llm(
    prompt,
    max_tokens=150,
    temperature=0.2,
    grammar=grammar,
    stop=["<<||<|<|"],
)

raw = output["choices"][0]["text"].strip()
print(f"\nRaw output:\n{raw}\n")

# Parse it
try:
    data = json.loads(raw)
    print(f"Parsed: {data}")
    print(f"Type: {data['type']}")
    if data['type'] == 'tool':
        print(f"Tool: {data['name']}")
        print(f"Args: {data['args']}")
except json.JSONDecodeError as e:
    print(f"Parse failed: {e}")
    print(f"Raw was: {repr(raw)}")

