import mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load(
    "mlx-community/gpt-oss-20b-MXFP4-Q4",
    tokenizer_config={"trust_remote_code": True}
)

mx.eval(model.parameters())

vocab_size = tokenizer.vocab_size
prompt_length = 4096

mx.random.seed(0)

dummy_tokens = mx.random.randint(0, vocab_size, (prompt_length,)).tolist()

tokenizer._eos_token_ids = {}

response = generate(
    model, 
    tokenizer, 
    prompt=dummy_tokens, 
    max_tokens=128, 
    verbose=True,
    prefill_step_size=4096 
)

print("\nActual benchmark:")
response = generate(
    model, 
    tokenizer, 
    prompt=dummy_tokens, 
    max_tokens=128, 
    verbose=True,
    prefill_step_size=4096  # Key optimization: process entire prompt in one go
)
