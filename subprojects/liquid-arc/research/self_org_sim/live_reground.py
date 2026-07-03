# Closed loop: trained Liquid HOLDS the mission -> retrieves it from a bank when the LLM has
# lost it from context -> re-grounds the LLM. Decoder-free (retrieval, not embedding->text).
# Holder runs client-side (~3M params, CPU); 30B via the persistent server (mclient).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F, numpy as np, mclient
from train_steer_controller import SteerController
torch.manual_seed(0)

data = torch.load('/home/pokazge/checkpoints/mission_seqs.pt', weights_only=False, map_location='cpu')
N = len(data)
bank_z = torch.stack([F.normalize(d['z'], dim=0) for d in data])   # [N,384] all missions = the retrieval bank
bank_txt = [d['g'] for d in data]
dev = torch.device('cpu')
ctrl = SteerController(d_llm=2048, z_goal_dim=384, d=128, K=8, use_slow=True).to(dev)
_CK = sys.argv[1] if len(sys.argv) > 1 else '/home/pokazge/checkpoints/mission_holder.pt'
print('[reground] holder =', _CK)
ctrl.load_state_dict(torch.load(_CK, map_location='cpu')['controller'])
ctrl.eval()

# held-out missions = the test split used in training (last 25%)
test_idx = list(range(int(N * 0.75), N))
DIST = ['Who won the World Cup in 2018?', 'What is a good recipe for dinner tonight?',
        'Tell me a fun fact about octopuses.', 'How does Wi-Fi work?', 'What is the capital of New Zealand?']
rng = np.random.default_rng(1)

def onmiss(text, z_true):                       # cos(mission-state of a response, true mission)
    ms = mclient.gen([{'role': 'user', 'content': text},
                      {'role': 'user', 'content': 'In ONE short line, restate the overall task and the next step.'}], 36)
    return float((mclient.encode(ms) * z_true).sum()), ms

retr_hits = 0; retr_tot = 0
base_cos = []; reg_cos = []
print('=== LIQUID HOLDS -> RETRIEVES -> RE-GROUNDS the LLM (held-out missions) ===\n')
for mi in test_idx:
    m = data[mi]; g = m['g']; z_true = F.normalize(m['z'], dim=0)
    print('MISSION[%d]: %s' % (mi, g[:90]))
    with torch.no_grad():
        ctrl.reset_episode(1, dev); ctrl.slow_step(m['seq'][0].unsqueeze(0))
    hist = [{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + g}]
    prompts = [('full', 'What is the first concrete step?'), ('full', 'Done. What should we do next?'),
               ('trunc', DIST[rng.integers(5)]), ('trunc', DIST[rng.integers(5)]),
               ('trunc', 'What should I focus on now?'), ('trunc', 'And what is the final step to finish?')]
    for t, (mode, u) in enumerate(prompts):
        ctx = hist if mode == 'full' else hist[-2:]          # truncated: mission gone from context
        r = mclient.gen(ctx + [{'role': 'user', 'content': u}], 42)
        ms = mclient.gen(ctx + [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r},
                                {'role': 'user', 'content': 'In ONE short line, restate the overall task and the next step.'}], 36)
        with torch.no_grad():
            h = ctrl.dyn_state(mclient.encode(ms).unsqueeze(0)); ctrl.h_c = h
            held = F.normalize(ctrl.g_head(h.flatten(1)).squeeze(0), dim=0)
        if mode == 'trunc':
            # (a) RETRIEVE the mission from the held belief
            j = int((bank_z @ held).argmax()); hit = (j == mi); retr_hits += hit; retr_tot += 1
            # (b) BASELINE: LLM answers with mission out of context
            bc, _ = onmiss(' '.join([c['content'] for c in ctx]) + ' User: ' + u, z_true)
            base_cos.append(bc)
            # (c) RE-GROUND: inject the RETRIEVED mission back, then answer
            reg_ctx = [{'role': 'system', 'content': 'Remember, the overall task is: ' + bank_txt[j]}] + ctx[-2:] + [{'role': 'user', 'content': u}]
            rr = mclient.gen(reg_ctx, 42)
            rms = mclient.gen(reg_ctx + [{'role': 'assistant', 'content': rr},
                              {'role': 'user', 'content': 'In ONE short line, restate the overall task and the next step.'}], 36)
            rc = float((mclient.encode(rms) * z_true).sum()); reg_cos.append(rc)
            print('  turn %d [trunc] retrieve->%s (%s)  on-mission: baseline=%.2f  re-grounded=%.2f' %
                  (t, 'CORRECT' if hit else 'idx%d' % j, bank_txt[j][:40], bc, rc))
        hist += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
    print()

print('=== SUMMARY (held-out, %d missions, %d truncated turns) ===' % (len(test_idx), retr_tot))
print('  retrieval top-1 (held belief -> correct mission of %d): %.0f%% (%d/%d)' % (N, 100 * retr_hits / retr_tot, retr_hits, retr_tot))
print('  on-mission cos  baseline (drifting) : %.3f' % np.mean(base_cos))
print('  on-mission cos  re-grounded by Liquid: %.3f  (Δ=%+.3f)' % (np.mean(reg_cos), np.mean(reg_cos) - np.mean(base_cos)))
print('=== ALL_DONE ===')
