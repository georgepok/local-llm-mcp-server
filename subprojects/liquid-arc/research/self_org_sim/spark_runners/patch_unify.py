# Derive train_unify.py from train_liquid_lora2.py: add belief->MISSION reconstruction (holding)
# alongside the existing distillation-CE (actuation) + contrastive (which-step). One controller, all three.
import io
src = open('/home/pokazge/liquid-arc/research/self_org_sim/train_liquid_lora2.py').read()
def rep(old, new):
    global src
    assert src.count(old) == 1, 'expected 1 occurrence, got %d for: %r' % (src.count(old), old[:60])
    src = src.replace(old, new)

# A) new arg
rep('    p.add_argument("--contrast_tau", type=float, default=0.1, help="InfoNCE temperature")',
    '    p.add_argument("--contrast_tau", type=float, default=0.1, help="InfoNCE temperature")\n'
    '    p.add_argument("--lambda_recon", type=float, default=0.0, help="weight of belief->MISSION reconstruction (holding readout)")')

# B) init rloss list
rep('    n_way = len(way_emb); ces, closs, coss = [], [], []',
    '    n_way = len(way_emb); ces, closs, coss, rloss = [], [], [], []')

# C) recon term right after the belief is built
rep('        h = controller.dyn_state(way_emb[p].unsqueeze(0), h_llm); lora.set_state(h)',
    '        h = controller.dyn_state(way_emb[p].unsqueeze(0), h_llm); lora.set_state(h)\n'
    '        if args.lambda_recon > 0:                                # HOLDING: belief reads back as the overall MISSION\n'
    '            held = F.normalize(controller.g_head(h.flatten(1)).squeeze(0), dim=0)\n'
    '            rloss.append(1.0 - (held * F.normalize(zG, dim=0)).sum())')

# D) return r_m as a 4th value
rep('    ce_m = torch.stack(ces).mean() if ces else None\n'
    '    ct_m = torch.stack(closs).mean() if closs else None\n'
    '    return ce_m, ct_m, float(np.mean(coss)) if coss else 0.0',
    '    ce_m = torch.stack(ces).mean() if ces else None\n'
    '    ct_m = torch.stack(closs).mean() if closs else None\n'
    '    r_m = torch.stack(rloss).mean() if rloss else None\n'
    '    return ce_m, ct_m, r_m, float(np.mean(coss)) if coss else 0.0')

# E) main loop: unpack 4, fold recon into loss, track its roll
rep('    best = run_eval("init"); closs = 0.0; ctsum = 0.0; cn = 0',
    '    best = run_eval("init"); closs = 0.0; ctsum = 0.0; rcsum = 0.0; cn = 0')
rep('        opt.zero_grad(); lsum = 0.0; ctl = 0.0; nb = 0',
    '        opt.zero_grad(); lsum = 0.0; ctl = 0.0; rcl = 0.0; nb = 0')
rep('            ce, ct, _ = episode(model, tok, controller, lora, enc_tok, enc_model, g, wtxt, wemb, zG, device, args, train=True)',
    '            ce, ct, rc, _ = episode(model, tok, controller, lora, enc_tok, enc_model, g, wtxt, wemb, zG, device, args, train=True)')
rep('            loss = ce if ct is None else ce + args.lambda_contrast * ct\n'
    '            (loss / args.group).backward(); lsum += float(ce.detach()); ctl += float(ct.detach()) if ct is not None else 0.0; nb += 1',
    '            loss = ce\n'
    '            if ct is not None: loss = loss + args.lambda_contrast * ct\n'
    '            if rc is not None: loss = loss + args.lambda_recon * rc\n'
    '            (loss / args.group).backward(); lsum += float(ce.detach()); ctl += float(ct.detach()) if ct is not None else 0.0; rcl += float(rc.detach()) if rc is not None else 0.0; nb += 1')
rep('        opt.step(); closs += lsum / max(1, nb); ctsum += ctl / max(1, nb); cn += 1',
    '        opt.step(); closs += lsum / max(1, nb); ctsum += ctl / max(1, nb); rcsum += rcl / max(1, nb); cn += 1')
rep('            print(f"step {step:>4}  distill_CE(roll)={closs/cn:.3f}  contrast(roll)={ctsum/cn:.3f}", flush=True); closs = 0.0; ctsum = 0.0; cn = 0',
    '            print(f"step {step:>4}  distill_CE(roll)={closs/cn:.3f}  contrast(roll)={ctsum/cn:.3f}  recon_cos(roll)={1.0-rcsum/cn:.3f}", flush=True); closs = 0.0; ctsum = 0.0; rcsum = 0.0; cn = 0')

open('/home/pokazge/liquid-arc/research/self_org_sim/train_unify.py', 'w').write(src)
import ast; ast.parse(src)
print('train_unify.py written + parses OK')
