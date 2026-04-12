"""Debug exactly where the gradient chain breaks in Qwen3 attention.

The ODE correction has grad_fn. The attention_mask modification should
propagate. Let's check what Qwen3's attention does with the mask.
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Qwen3-4B (eager attention for inspection)...")
tok = AutoTokenizer.from_pretrained("/workspace/models/qwen3-4b", trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained("/workspace/models/qwen3-4b", device_map="cuda",
    torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="eager")
llm.eval()
for p in llm.parameters():
    p.requires_grad_(False)

text = "Hello world"
inputs = tok(text, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
N = input_ids.shape[1]

# Create a differentiable bias and inject via hook on layer 0 only
bias_param = torch.randn(1, 1, N, N, device="cuda", dtype=torch.bfloat16, requires_grad=True)

def hook_fn(module, args, kwargs):
    mask = kwargs.get("attention_mask", None)
    if mask is not None:
        kwargs["attention_mask"] = mask + bias_param
    return

layer0 = llm.model.layers[0]
h = layer0.register_forward_pre_hook(hook_fn, with_kwargs=True)

outputs = llm(input_ids=input_ids)
logits = outputs.logits

print(f"bias_param.requires_grad: {bias_param.requires_grad}")
print(f"logits.requires_grad: {logits.requires_grad}")
print(f"logits.grad_fn: {logits.grad_fn}")

if logits.grad_fn is not None:
    loss = logits.sum()
    loss.backward()
    print(f"bias_param.grad: {bias_param.grad is not None}")
    if bias_param.grad is not None:
        print(f"bias_param.grad.norm: {bias_param.grad.norm().item():.6f}")
        print("*** GRADIENT FLOWS through attention_mask ***")
    else:
        print("*** grad exists on logits but not on bias_param ***")
else:
    print("\n*** logits has no grad_fn ***")
    # Check: does the LLM use torch.no_grad internally?
    # Try: manually compute attention for one layer to see if mask is differentiable
    print("\nManual attention test:")
    embed = llm.model.embed_tokens(input_ids)
    print(f"  embed.grad_fn: {embed.grad_fn}")

    # Run just layer 0
    layer0_out = layer0(embed, attention_mask=None)
    print(f"  layer0_out type: {type(layer0_out)}")
    if isinstance(layer0_out, tuple):
        h_out = layer0_out[0]
    else:
        h_out = layer0_out
    print(f"  h_out.grad_fn: {h_out.grad_fn}")

h.remove()
print(f"\nGPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
