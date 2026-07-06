import io
f='/home/pokazge/NativeEntity/native_entity.py'
s=open(f).read()
old=("        FE = int(os.environ.get('FIELD_EPOCHS', '40')); items = [(S, ri, d) for S, ri in Sep_tr for d in range(nN4) if not pair_unseen(ri, d)]\n"
     "        print('  P11-TRAIN field-train items=%d (SEEN pairs) | FIELD_EPOCHS=%d' % (len(items), FE), flush=True)")
new=("        FE = int(os.environ.get('FIELD_EPOCHS', '40')); items = [(S, ri, d) for S, ri in Sep_tr for d in range(nN4) if not pair_unseen(ri, d)]\n"
     "        if os.environ.get('BALANCE', '1') == '1':\n"
     "            mm = [it for it in items if it[1] == it[2]]; nn_ = [it for it in items if it[1] != it[2]]\n"
     "            if mm and nn_: reps = max(1, round(len(nn_) / len(mm))); items = nn_ + mm * reps\n"
     "            print('  P11-TRAIN BALANCED match-oversampled (match=%d nonmatch=%d)' % (len([it for it in items if it[1]==it[2]]), len([it for it in items if it[1]!=it[2]])), flush=True)\n"
     "        print('  P11-TRAIN field-train items=%d | FIELD_EPOCHS=%d' % (len(items), FE), flush=True)")
assert s.count(old)==1, 'old block found %d times'%s.count(old)
open(f,'w').write(s.replace(old,new))
print('PATCHED OK')
