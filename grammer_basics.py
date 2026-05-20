from llama_cpp import Llama, LlamaGrammar

MODEL_PATH = "/home/RYVEN/mini_agent/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

# Load model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,        # Small context for this test
    n_threads=4,       # Adjust to your CPU cores
    verbose=False,
)

# A simple prompt that asks for a tool call
prompt = """You are an agent. You can use tools.

Available tools:
- read_file(path)

Respond with a JSON object indicating which tool to use.

User: Read the file main.py

Response:"""

print("=" * 50)
print("WITHOUT GRAMMAR:")
print("=" * 50)

output = llm(
    prompt,
    max_tokens=100,
    temperature=0.7,   # Higher temp = more creative = more likely to mess up
)
raw = output["choices"][0]["text"]
print(raw)
print()

print("=" * 50)
print("WITH GRAMMAR:")
print("=" * 50)

# Define a grammar that ONLY allows: {"tool": "read_file", "args": {...}}
# or {"final": "..."}
grammar_text = r'''
root ::= tool | final
tool ::= "{" space "\"tool\"" space ":" space string space "," space "\"args\"" space ":" space object space "}"
final ::= "{" space "\"final\"" space ":" space string space "}"
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" [nrt"\\]
object ::= "{" space (pair (space "," space pair)*)? space "}"
pair ::= string space ":" space value
value ::= string | number | "true" | "false" | "null"
number ::= "-"? [0-9]+ ("." [0-9]+)?
space ::= " " | "\n" | "\t" | ""
'''

grammar = LlamaGrammar.from_string(grammar_text)

output = llm(
    prompt,
    max_tokens=100,
    temperature=0.7,   # Same high temp — but grammar overrides it
    grammar=grammar,
)
raw = output["choices"][0]["text"]
print(raw)
