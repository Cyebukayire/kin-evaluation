"""Check that a model loads fully and writes sensible text, outside lm-eval.
    python scripts/debug_load.py google/gemma-4-31B-it
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name)
model, info = AutoModelForCausalLM.from_pretrained(
    name, dtype=torch.bfloat16, device_map="cuda:0", output_loading_info=True)
print("class:", type(model).__name__)
print(f"GPU memory used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
for k, v in info.items():
    print(f"{k}: {len(v)}", list(v)[:5])



prompt = "Question: Alice has 3 apples and buys 4 more. How many apples does she have?\nAnswer:"
print("bos_token:", tok.bos_token, tok.bos_token_id)

def run(label, ids, m=model):
    ids = ids.to("cuda:0")
    out = m.generate(ids, max_new_tokens=30, do_sample=False)
    print(f"\n[{label}] first tokens={tok.convert_ids_to_tokens(ids[0][:3])}")
    print(repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))

plain = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
run("no bos", plain)
run("bos by hand", torch.cat([torch.tensor([[tok.bos_token_id]]), plain], dim=1))
chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                               add_generation_prompt=True, return_tensors="pt")
run("chat template", chat if isinstance(chat, torch.Tensor) else chat["input_ids"])

del model; torch.cuda.empty_cache()
eager = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda:0",
                                             attn_implementation="eager")
run("bos by hand + eager attention",
    torch.cat([torch.tensor([[tok.bos_token_id]]), plain], dim=1), eager)