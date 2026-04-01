"""Smoke test for LiquidARC Mind."""
import sys
sys.path.insert(0, '/workspace/liquid-arc')

from liquid_arc.config import LiquidARCConfig
from liquid_arc.mind import LiquidARCMind
from sentence_transformers import SentenceTransformer

print("Loading config...")
config = LiquidARCConfig.from_yaml('/workspace/liquid-arc/configs/linguistic_mind.yaml')
print(f"Config: d_model={config.d_model}, n_ode_steps={config.n_ode_steps}")

print("Loading embedder...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

print("Creating mind...")
mind = LiquidARCMind(
    checkpoint_path='/workspace/liquid-arc/output_30m/checkpoints/best.pt',
    config=config, text_embedder=embedder, device='cuda',
    max_context_events=64, freeze_dynamics=True, enable_online_learning=False,
)
print(f"Mind ready, h shape: {mind._h.shape}")

# Test 1: observe_event
print("\n--- Test 1: observe_event ---")
r1 = mind.observe_event('user_message', 'Hello, how are you doing today?')
print(f"pred_err={r1['prediction_error']:.3f}, cv={r1['cv']:.3f}, h_norm={r1['h_norm']:.3f}, events={r1['events_in_context']}")

r2 = mind.observe_event('assistant_message', 'I am doing well, thank you for asking!')
print(f"pred_err={r2['prediction_error']:.3f}, cv={r2['cv']:.3f}")

# Test 2: get_context
print("\n--- Test 2: get_context ---")
ctx = mind.get_context()
print(f"status={ctx['status']}, n_events={ctx['n_events']}")
for item in ctx['context']:
    print(f"  [{item['type']}] rel={item['relevance']:.3f}: {item['preview'][:60]}")

# Test 3: diagnostics
print("\n--- Test 3: diagnostics ---")
diag = mind.get_diagnostics()
print(f"cv={diag['metric_cv']:.3f}, tau={diag['tau_mean']:.3f}, beta={diag['beta_mean']:.3f}")

# Test 4: topic shift
print("\n--- Test 4: topic shift ---")
r3 = mind.observe_event('user_message', 'Actually let us discuss quantum computing and entanglement')
print(f"Topic shift pred_err={r3['prediction_error']:.3f} (was {r2['prediction_error']:.3f})")

# Test 5: build up context
print("\n--- Test 5: multi-event context ---")
for msg in ['The superposition principle is fascinating',
            'How does decoherence work?',
            'What about quantum error correction?',
            'Can you explain the no-cloning theorem?']:
    mind.observe_event('user_message', msg)
    mind.observe_event('assistant_message', f'Great question about {msg.split()[-1]}...')

ctx2 = mind.get_context()
print(f"n_events={ctx2['n_events']}, focus={ctx2['focus_indices']}")
for item in ctx2['context'][:3]:
    print(f"  TOP: [{item['type']}] rel={item['relevance']:.3f}: {item['preview'][:60]}")

# Test 6: goal
print("\n--- Test 6: signal_goal ---")
gr = mind.signal_goal('Understand quantum computing basics', priority=0.9)
print(f"Goal pred_err={gr['prediction_error']:.3f}")

# Test 7: reset
print("\n--- Test 7: reset ---")
mind.reset()
ctx3 = mind.get_context()
print(f"After reset: {ctx3['status']}")

print("\n=== ALL SMOKE TESTS PASSED ===")
