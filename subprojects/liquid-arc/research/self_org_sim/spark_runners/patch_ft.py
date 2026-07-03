# train_unify_ft.py: FULL-TRUNCATION training. Student context = ONLY the drift turn (zero task
# content) -> the belief->LoRA is the ONLY way the mission reaches generation. Forces real
# actuation-from-belief (the prior keep_turns=1 left a context crutch -> LoRA never learned to inject).
src = open('/home/pokazge/liquid-arc/research/self_org_sim/train_unify.py').read()
def rep(old, new, n=1):
    c = src.count(old); assert c == n, 'expected %d, got %d: %r' % (n, c, old[:65])
    return src.replace(old, new)

# add --full_truncate flag
src = rep('    p.add_argument("--keep_turns", type=int, default=0)',
          '    p.add_argument("--keep_turns", type=int, default=0)\n'
          '    p.add_argument("--full_truncate", action="store_true", help="student context = ONLY the drift turn (zero task content) -> actuate from belief alone")')

# both s_msgs / msgs context builders (episode + eval_onplan) -> empty when full_truncate
old_ctx = '(history[-2*args.keep_turns:] if args.keep_turns>0 else history)'
new_ctx = '([] if args.full_truncate else (history[-2*args.keep_turns:] if args.keep_turns>0 else history))'
src = rep(old_ctx, new_ctx, n=2)

open('/home/pokazge/liquid-arc/research/self_org_sim/train_unify_ft.py', 'w').write(src)
import ast; ast.parse(src)
print('train_unify_ft.py written + parses OK (full-truncation regime)')
