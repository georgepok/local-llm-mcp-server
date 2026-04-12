"""Verify: can CE loss gradients reach MetricNet through attention bias injection?

Tests the full gradient path:
  CE loss → logits → attention(QK^T + bias) → bias → ODE correction → MetricNet

If logits.grad_fn is not None, the path exists and CE training is possible.
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from liquid_arc.config import LiquidARCConfig
from liquid_arc.dynamics import ContinuousDynamics
from liquid_arc.context_pool import ContextPool
from liquid_arc.layer_wise_ode import LayerWiseODE, hook_llm_layers

print("Loading...")
config = LiquidARCConfig.from_yaml("/workspace/liquid-arc/configs/mind_layerwise.yaml")
tok = AutoTokenizer.from_pretrained("/workspace/models/qwen3-4b", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained("/workspace/models/qwen3-4b", device_map="cuda",
    torch_dtype=torch.bfloat16, trust_remote_code=True)
llm.eval()
for p in llm.parameters():
    p.requires_grad_(False)

# ODE in float32, train mode, requires_grad
dynamics = ContinuousDynamics(config).to("cuda").float().train()
context_pool = ContextPool(config).to("cuda").float().train()
ckpt = torch.load("/workspace/liquid-arc/output_layerwise_v3/checkpoints/step_1500.pt",
    map_location="cuda", weights_only=False)
dynamics.load_state_dict(ckpt["dynamics_state"])
context_pool.load_state_dict(ckpt["context_pool_state"])
dynamics = dynamics.float().train()
context_pool = context_pool.float().train()
dynamics.freeze_tau = False

# Verify dynamics params require grad
n_grad = sum(1 for p in dynamics.parameters() if p.requires_grad)
n_total = sum(1 for p in dynamics.parameters())
print(f"Dynamics: {n_grad}/{n_total} params require grad")

# Create ODE in training mode
layer_ode = LayerWiseODE(dynamics=dynamics, context_pool=context_pool,
    n_layers=llm.config.num_hidden_layers, d_llm=llm.config.hidden_size,
    d_ode=config.d_model, epsilon=0.5, device="cuda")
layer_ode.training_mode = True

# Register hooks (attention mode)
hooks = hook_llm_layers(llm, layer_ode, mode='attention')

# Forward pass
text = "The bridge collapsed causing traffic rerouting."
inputs = tok(text, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
labels = input_ids.clone()

layer_ode.start_forward()
outputs = llm(input_ids=input_ids)
layer_ode.end_forward()

logits = outputs.logits

# Check gradient path
print(f"\nlogits.requires_grad: {logits.requires_grad}")
print(f"logits.grad_fn: {logits.grad_fn}")

if logits.grad_fn is None:
    print("\n*** NO GRADIENT PATH — CE training not possible with this approach ***")

    # Debug: check intermediate tensors
    print("\nDebugging gradient chain...")
    # Check if bias had grad_fn
    if layer_ode.layer_biases:
        print(f"  layer_biases stored: {len(layer_ode.layer_biases)}")
    if layer_ode.correction is not None:
        print(f"  correction.requires_grad: {layer_ode.correction.requires_grad}")
        print(f"  correction.grad_fn: {layer_ode.correction.grad_fn}")
else:
    print("\n*** GRADIENT PATH EXISTS — CE training IS possible ***")

    # Compute CE loss and backprop
    ce_loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        labels[:, 1:].contiguous().view(-1))

    print(f"CE loss: {ce_loss.item():.4f}")
    print(f"CE loss grad_fn: {ce_loss.grad_fn}")

    ce_loss.backward()

    # Check which MetricNet params got gradients
    grad_params = 0
    nonzero_grads = 0
    for name, p in dynamics.named_parameters():
        if p.grad is not None:
            grad_params += 1
            if p.grad.abs().max() > 0:
                nonzero_grads += 1
                if "metric" in name or "tau" in name:
                    print(f"  {name}: grad_norm={p.grad.norm().item():.6f}")

    print(f"\nParams with grad: {grad_params}/{n_total}")
    print(f"Params with nonzero grad: {nonzero_grads}/{n_total}")

    if nonzero_grads > 0:
        print("\n*** CE → MetricNet gradient flow CONFIRMED ***")
    else:
        print("\n*** Gradients exist but are all zero — numerically dead ***")

# Cleanup
for h in hooks:
    h.remove()
print(f"\nGPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
