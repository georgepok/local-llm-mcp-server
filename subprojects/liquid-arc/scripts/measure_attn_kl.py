"""Measure how much the ODE bias actually changes attention distributions.

KL divergence between plain attention and ODE-biased attention at each layer.
If KL is near zero, the bias isn't changing attention — the perturbation is too weak.
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from liquid_arc.config import LiquidARCConfig
from liquid_arc.dynamics import ContinuousDynamics
from liquid_arc.context_pool import ContextPool
from liquid_arc.layer_wise_ode import LayerWiseODE, hook_llm_layers

config = LiquidARCConfig.from_yaml("/workspace/liquid-arc/configs/mind_layerwise.yaml")
tok = AutoTokenizer.from_pretrained("/workspace/models/qwen3-4b", trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained("/workspace/models/qwen3-4b", device_map="cuda",
    torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="eager")
llm.eval()
dynamics = ContinuousDynamics(config).to("cuda").to(torch.bfloat16).eval()
context_pool = ContextPool(config).to("cuda").to(torch.bfloat16).eval()
ckpt = torch.load("/workspace/liquid-arc/output_layerwise_v3/checkpoints/step_500.pt",
    map_location="cuda", weights_only=False)
dynamics.load_state_dict(ckpt["dynamics_state"])
context_pool.load_state_dict(ckpt["context_pool_state"])
dynamics.freeze_tau = False

text = "A bridge collapsed. Trucks were rerouted through downtown. Supply disruptions hit the warehouse."
inputs = tok(text, return_tensors="pt").to("cuda")
N = inputs["input_ids"].shape[1]

# Plain forward
with torch.no_grad():
    out_plain = llm(**inputs, output_attentions=True)
plain_attns = [a[0].float().mean(dim=0) for a in out_plain.attentions]

# ODE forward at eps=0.5
layer_ode = LayerWiseODE(dynamics=dynamics, context_pool=context_pool,
    n_layers=llm.config.num_hidden_layers, d_llm=llm.config.hidden_size,
    d_ode=config.d_model, epsilon=0.5, device="cuda")
hooks = hook_llm_layers(llm, layer_ode)
layer_ode.start_forward()
with torch.no_grad():
    out_ode = llm(**inputs, output_attentions=True)
layer_ode.end_forward()
ode_attns = [a[0].float().mean(dim=0) for a in out_ode.attentions]
for h in hooks:
    h.remove()

# Also try eps=5.0 (extreme) to see if higher eps changes anything
layer_ode2 = LayerWiseODE(dynamics=dynamics, context_pool=context_pool,
    n_layers=llm.config.num_hidden_layers, d_llm=llm.config.hidden_size,
    d_ode=config.d_model, epsilon=5.0, device="cuda")
hooks2 = hook_llm_layers(llm, layer_ode2)
layer_ode2.start_forward()
with torch.no_grad():
    out_ode5 = llm(**inputs, output_attentions=True)
layer_ode2.end_forward()
ode5_attns = [a[0].float().mean(dim=0) for a in out_ode5.attentions]
for h in hooks2:
    h.remove()

print(f"Tokens: {N}")
print(f"{'Lyr':>3} {'KL(e=0.5)':>10} {'KL(e=5.0)':>10} {'MaxD(0.5)':>10} {'MaxD(5.0)':>10}")
print(f"{'---':>3} {'----------':>10} {'----------':>10} {'----------':>10} {'----------':>10}")
for i in range(len(plain_attns)):
    p = plain_attns[i].clamp(min=1e-10)
    q = ode_attns[i].clamp(min=1e-10)
    q5 = ode5_attns[i].clamp(min=1e-10)
    kl = (p * (p.log() - q.log())).sum(dim=-1).mean().item()
    kl5 = (p * (p.log() - q5.log())).sum(dim=-1).mean().item()
    md = (p - q).abs().max().item()
    md5 = (p - q5).abs().max().item()
    print(f"{i:>3} {kl:>10.6f} {kl5:>10.6f} {md:>10.6f} {md5:>10.6f}")

# Summary
kl_05 = [(plain_attns[i].clamp(min=1e-10) * (plain_attns[i].clamp(min=1e-10).log() - ode_attns[i].clamp(min=1e-10).log())).sum(dim=-1).mean().item() for i in range(len(plain_attns))]
kl_50 = [(plain_attns[i].clamp(min=1e-10) * (plain_attns[i].clamp(min=1e-10).log() - ode5_attns[i].clamp(min=1e-10).log())).sum(dim=-1).mean().item() for i in range(len(plain_attns))]
print(f"\nMean KL(eps=0.5): {sum(kl_05)/len(kl_05):.6f}")
print(f"Mean KL(eps=5.0): {sum(kl_50)/len(kl_50):.6f}")
print(f"Max  KL(eps=0.5): {max(kl_05):.6f} at layer {kl_05.index(max(kl_05))}")
print(f"Max  KL(eps=5.0): {max(kl_50):.6f} at layer {kl_50.index(max(kl_50))}")
