import mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load(
    "openai/gpt-oss-20b,
    tokenizer_config={"trust_remote_code": True}
)

file_path = "input.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text_prompt = f.read()

response = generate(
    model, tokenizer,
    prompt=text_prompt,
    max_tokens=128,
    verbose=True
)

print(response)
