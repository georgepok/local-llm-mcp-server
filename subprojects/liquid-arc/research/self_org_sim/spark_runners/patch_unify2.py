# Derive train_unify2.py from train_unify.py: SHARPER teacher (bigger on-step contrast -> larger
# coherent distillation gradient -> stronger learned steer). Magnitude headroom comes from --cap_rel 1.0 (CLI).
src = open('/home/pokazge/liquid-arc/research/self_org_sim/train_unify.py').read()
def rep(old, new):
    assert src.count(old) == 1, 'expected 1, got %d: %r' % (src.count(old), old[:70])
    return src.replace(old, new)
old_teacher = '        t_msgs = history + [{"role": "user", "content": f"{drift} (Stay on our task — the next step is: {wp_t}.)"}]'
new_teacher = '        t_msgs = history + [{"role": "user", "content": f"{drift}\\n\\nSet that aside and refocus on our task. The single next step is: {wp_t}. Do exactly that step now — concretely and directly, no preamble."}]'
src = rep(old_teacher, new_teacher)
open('/home/pokazge/liquid-arc/research/self_org_sim/train_unify2.py', 'w').write(src)
import ast; ast.parse(src)
print('train_unify2.py written + parses OK (sharper teacher)')
