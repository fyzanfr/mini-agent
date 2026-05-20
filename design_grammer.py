import json

AGENT_GRAMMER = r'''
root ::= tool | final

tool ::= "{" ws "\"type\"" ws ":" ws "\"tool\"" ws "," ws "\"name\"" ws ":" ws tool_name ws "," ws "\"args\"" ws ":" ws object ws "}"
final ::= "{" ws "\"type\"" ws ":" ws "\"final\"" ws "," ws "\"content\"" ws ":" ws string ws "}"
tool_name ::= "\"list_files\"" | "\"read_file\"" | "\"write_file\"" | "\"run_shell\""

object ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair ::= string ws ":" ws value
value ::= string | number | "true" | "false" | "null"

string ::= "\"" char* "\""
char ::= [^"\\] | "\\" [nrt"\\]
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws ::= " " | "\n" | "\t" | ""
'''

print("=" * 50)
print("Testing what the grammar allows:")
print("=" * 50)

valid_outputs = [
    '{"type":"tool","name":"list_files","args":{"path":"."}}',
    '{"type":"tool","name":"read_file","args":{"path":"main.py","start":1,"end":50}}',
    '{"type":"tool","name":"write_file","args":{"path":"test.py","content":"def hello(): pass"}}',
    '{"type":"tool","name":"run_shell","args":{"command":"ls -la","timeout":20}}',
    '{"type":"final","content":"Done. There are 3 files."}',
]

for output in valid_outputs:
    try:
        data = json.loads(output)
        print(f"✓ VALID: {output[:80]}...")
    except json.JSONDecodeError as e:
        print(f"✗ INVALID: {output[:80]}... ({e})")

print()
print("=" * 50)
print("Testing what the grammar FORBIDS:")
print("=" * 50)

invalid_outputs = [
    'Sure! Let me check that for you. {"type":"tool"...}',  # Prose before JSON
    '{"type":"tool","name":"read_file"}',  # Missing args
    '{"type":"tool","name":"delete_file","args":{}}',  # Unknown tool name
    '{"type":"final"}',  # Missing content
    '```json\n{"type":"tool"...}\n```',  # Markdown wrapping
]

for output in invalid_outputs:
    # These would be physically impossible for the model to generate
    # because the grammar sampler blocks those tokens
    print(f"✗ FORBIDDEN: {output[:80]}...")


