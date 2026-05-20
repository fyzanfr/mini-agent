# save as test_gguf.py
from llama_cpp import Llama

model_path = "/home/RYVEN/mini_agent/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

print("Loading model... (10-30 seconds first time)")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=8,      # Set to your CPU core count
    verbose=False,
)
print("Loaded!\n")

prompt = """<|<|im_start|>system
You are a helpful coding assistant.|<|im_end|>
<|<|im_start|>user
Write a Python function to reverse a string.|<|im_end|>
<|<|im_start|>assistant
"""

output = llm(prompt, max_tokens=100, temperature=0.7)
print(output["choices"][0]["text"])
